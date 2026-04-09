import asyncio
import logging
from collections import deque
from embeddings.encoder import EmbeddingEncoder
from storage.database import DatabaseWriter

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """
    Continuous background embedding worker.
    Embeds nodes and micro nodes as fast as possible.
    Never blocks agent execution.
    Runs independently of everything else.
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        db_writer: DatabaseWriter,
        batch_size: int = 10,
        poll_interval_ms: int = 50,
    ):
        self.encoder = encoder
        self.db = db_writer
        self.batch_size = batch_size
        self.poll_interval = poll_interval_ms / 1000

        self._node_queue: deque = deque()
        self._micro_queue: deque = deque()
        self._running = False
        self._task = None
        self._embedded_count = 0
        self._loop = None

    async def start(self):
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._task = asyncio.create_task(
            self._worker_loop()
        )
        logger.info("EmbeddingWorker started.")

    async def stop(self):
        self._running = False
        # drain remaining queue
        await self._process_nodes()
        await self._process_micros()
        if self._task:
            self._task.cancel()
        logger.info(
            f"EmbeddingWorker stopped. "
            f"Embedded: {self._embedded_count}"
        )

    def queue_node(self, node):
        """
        Queue a parent node for immediate embedding.
        Called the instant a node is created.
        Non blocking.
        """
        self._node_queue.append(node)

    def queue_micro(self, micro_node):
        """
        Queue a micro node for immediate embedding.
        Called the instant a token is captured.
        Non blocking.
        """
        self._micro_queue.append(micro_node)

    async def _worker_loop(self):
        while self._running:
            await self._process_nodes()
            await self._process_micros()
            await asyncio.sleep(self.poll_interval)

    async def _process_nodes(self):
        if not self._node_queue:
            return

        batch = []
        while self._node_queue and (
            len(batch) < self.batch_size
        ):
            batch.append(self._node_queue.popleft())

        if not batch:
            return

        for node in batch:
            try:
                text = self._build_node_text(node)
                if not text:
                    continue

                loop = asyncio.get_event_loop()
                vector = await loop.run_in_executor(
                    None,
                    lambda t=text: (
                        self.encoder.embeddings
                        .embed_query(t)
                    )
                )

                if vector:
                    node.embedding_vector = vector
                    async with self.db.pool.acquire() as conn:
                        vec_str = (
                            "[" +
                            ",".join(str(x) for x in vector)
                            + "]"
                        )
                        await conn.execute(
                            """
                            UPDATE nodes
                            SET embedding_vector = $1::vector
                            WHERE node_id = $2
                            """,
                            vec_str,
                            node.node_id
                        )
                    self._embedded_count += 1

            except Exception as e:
                logger.error(
                    f"Node embedding failed: {e}"
                )

    async def _process_micros(self):
        if not self._micro_queue:
            return

        batch = []
        while self._micro_queue and (
            len(batch) < self.batch_size
        ):
            batch.append(self._micro_queue.popleft())

        if not batch:
            return

        texts = [m.content for m in batch if m.content]
        if not texts:
            return

        try:
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                None,
                lambda: (
                    self.encoder.embeddings
                    .embed_documents(texts)
                )
            )

            for micro_node, vector in zip(batch, vectors):
                micro_node.embedding_vector = vector
                micro_node.embedding_complete = True
                async with self.db.pool.acquire() as conn:
                    vec_str = (
                        "[" +
                        ",".join(str(x) for x in vector)
                        + "]"
                    )
                    await conn.execute(
                        """
                        UPDATE micro_nodes
                        SET embedding_vector = $1::vector
                        WHERE micro_id = $2
                        """,
                        vec_str,
                        micro_node.micro_id
                    )
                self._embedded_count += 1

        except Exception as e:
            logger.error(
                f"Micro embedding failed: {e}"
            )

    def _build_node_text(self, node) -> str:
        from tracer.node import NodeType
        parts = []

        if node.node_type == NodeType.REASONING:
            if node.prompt_text:
                parts.append(
                    f"PROMPT: {node.prompt_text[:500]}"
                )
            if node.response_text:
                parts.append(
                    f"RESPONSE: {node.response_text[:500]}"
                )

        elif node.node_type == NodeType.TOOL_CALL:
            if node.tool_name:
                parts.append(f"TOOL: {node.tool_name}")
            if node.input_params:
                parts.append(
                    f"INPUT: {str(node.input_params)[:300]}"
                )

        elif node.node_type == NodeType.TOOL_RESPONSE:
            if node.tool_name:
                parts.append(f"TOOL: {node.tool_name}")
            if node.raw_output:
                parts.append(
                    f"OUTPUT: {str(node.raw_output)[:300]}"
                )

        return "\n".join(parts) if parts else ""

    def get_stats(self) -> dict:
        return {
            "embedded_count": self._embedded_count,
            "node_queue_size": len(self._node_queue),
            "micro_queue_size": len(self._micro_queue),
            "running": self._running,
        }