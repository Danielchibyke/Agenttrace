import logging
from embedding_service.queue_client import EmbeddingQueueClient

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """
    Lightweight client-side worker.
    Pushes content to Redis queue.
    Actual embedding happens in the dedicated service.
    Agent never waits for embedding.
    """

    def __init__(self, *args, **kwargs):
        self.queue_client = EmbeddingQueueClient()
        self._connected = False
        self._queued_count = 0

    async def start(self):
        self._connected = self.queue_client.connect()
        if self._connected:
            logger.info(
                "EmbeddingWorker connected to Redis."
            )
        else:
            logger.warning(
                "Redis not available. "
                "Embeddings will be skipped."
            )

    async def stop(self):
        sizes = self.queue_client.get_queue_size()
        logger.info(
            f"EmbeddingWorker stopped. "
            f"Queued: {self._queued_count}. "
            f"Remaining in queue: {sizes}"
        )

    def queue_node(self, node):
        """
        Push node content to Redis immediately.
        Non blocking.
        """
        if not self._connected:
            return

        from tracer.node import NodeType

        parts = []
        if node.node_type == NodeType.REASONING:
            if node.prompt_text:
                parts.append(
                    f"PROMPT: {node.prompt_text[:500]}"
                )
            if node.response_text:
                parts.append(
                    f"RESPONSE: "
                    f"{node.response_text[:500]}"
                )
        elif node.node_type == NodeType.TOOL_CALL:
            if node.tool_name:
                parts.append(f"TOOL: {node.tool_name}")
            if node.input_params:
                parts.append(
                    f"INPUT: "
                    f"{str(node.input_params)[:300]}"
                )
        elif node.node_type == NodeType.TOOL_RESPONSE:
            if node.tool_name:
                parts.append(f"TOOL: {node.tool_name}")
            if node.raw_output:
                parts.append(
                    f"OUTPUT: "
                    f"{str(node.raw_output)[:300]}"
                )

        content = "\n".join(parts)
        if not content:
            return

        self.queue_client.push_node(
            node_id=node.node_id,
            content=content,
            node_type=node.node_type.value,
            session_id=node.session_id,
        )
        self._queued_count += 1

    def queue_micro(self, micro_node):
        """
        Push micro node content to Redis immediately.
        Non blocking.
        """
        if not self._connected:
            return
        if not micro_node.content:
            return

        self.queue_client.push_micro(
            micro_id=micro_node.micro_id,
            content=micro_node.content,
            session_id=micro_node.session_id,
            parent_node_id=micro_node.parent_node_id,
        )
        self._queued_count += 1

    def get_stats(self) -> dict:
        sizes = self.queue_client.get_queue_size()
        return {
            "queued_count": self._queued_count,
            "queue_sizes": sizes,
            "connected": self._connected,
        }