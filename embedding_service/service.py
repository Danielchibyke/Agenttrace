import sys
import os
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

import asyncio
import json
import logging
import signal
import redis.asyncio as aioredis
from embeddings.encoder import EmbeddingEncoder
from storage.database import DatabaseWriter
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EMBED] %(message)s"
)
logger = logging.getLogger(__name__)

EMBEDDING_QUEUE = "agenttrace:embed:queue"
MICRO_QUEUE = "agenttrace:embed:micro_queue"
EMBEDDING_READY = "agenttrace:embed:ready"

# batch settings
NODE_BATCH_SIZE = 5
MICRO_BATCH_SIZE = 20
POLL_INTERVAL = 0.05  # 50ms


class EmbeddingService:
    """
    Standalone embedding microservice.
    Runs as a separate process.
    Pulls from Redis queue.
    Embeds and updates PostgreSQL.
    Publishes ready events for WebSocket.
    """

    def __init__(self):
        self.encoder = EmbeddingEncoder()
        self.db = DatabaseWriter()
        self.redis = None
        self.pubsub_redis = None
        self._running = False
        self._embedded_nodes = 0
        self._embedded_micros = 0

    async def start(self):
        await self.db.connect()
        self.redis = await aioredis.from_url(
            "redis://localhost:6379",
            decode_responses=True,
        )
        self.pubsub_redis = await aioredis.from_url(
            "redis://localhost:6379",
            decode_responses=True,
        )
        self._running = True
        logger.info(
            "Embedding service started. "
            "Waiting for work..."
        )

    async def stop(self):
        self._running = False
        if self.redis:
            await self.redis.aclose()
        if self.pubsub_redis:
            await self.pubsub_redis.aclose()
        await self.db.close()
        logger.info(
            f"Embedding service stopped. "
            f"Nodes: {self._embedded_nodes} "
            f"Micros: {self._embedded_micros}"
        )

    async def run(self):
        await self.start()

        loop = asyncio.get_event_loop()

        def handle_shutdown(sig):
            logger.info(f"Shutdown signal received.")
            loop.create_task(self.stop())
            loop.stop()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        # run both queues concurrently
        await asyncio.gather(
            self._process_node_queue(),
            self._process_micro_queue(),
        )

    async def _process_node_queue(self):
        """Process parent node embeddings."""
        while self._running:
            try:
                batch = []
                while len(batch) < NODE_BATCH_SIZE:
                    item = await self.redis.lpop(
                        EMBEDDING_QUEUE
                    )
                    if not item:
                        break
                    try:
                        batch.append(json.loads(item))
                    except Exception:
                        continue

                if batch:
                    await self._embed_nodes(batch)
                else:
                    await asyncio.sleep(POLL_INTERVAL)

            except Exception as e:
                logger.error(
                    f"Node queue error: {e}"
                )
                await asyncio.sleep(0.1)

    async def _process_micro_queue(self):
        """Process micro node token embeddings."""
        while self._running:
            try:
                batch = []
                while len(batch) < MICRO_BATCH_SIZE:
                    item = await self.redis.lpop(
                        MICRO_QUEUE
                    )
                    if not item:
                        break
                    try:
                        batch.append(json.loads(item))
                    except Exception:
                        continue

                if batch:
                    await self._embed_micros(batch)
                else:
                    await asyncio.sleep(POLL_INTERVAL)

            except Exception as e:
                logger.error(
                    f"Micro queue error: {e}"
                )
                await asyncio.sleep(0.1)

    async def _embed_nodes(self, batch: list):
        """Embed a batch of parent nodes."""
        for item in batch:
            try:
                content = item.get("content", "")
                if not content:
                    continue

                loop = asyncio.get_event_loop()
                vector = await loop.run_in_executor(
                    None,
                    lambda c=content: (
                        self.encoder.embeddings
                        .embed_query(c)
                    )
                )

                if not vector:
                    continue

                vec_str = (
                    "[" +
                    ",".join(str(x) for x in vector) +
                    "]"
                )

                async with self.db.pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE nodes
                        SET embedding_vector = $1::vector
                        WHERE node_id = $2
                        """,
                        vec_str,
                        item["node_id"],
                    )

                # publish ready event
                await self.pubsub_redis.publish(
                    EMBEDDING_READY,
                    json.dumps({
                        "type": "node",
                        "node_id": item["node_id"],
                        "session_id": item["session_id"],
                    })
                )

                self._embedded_nodes += 1

                if self._embedded_nodes % 10 == 0:
                    logger.info(
                        f"Embedded {self._embedded_nodes} "
                        f"nodes, "
                        f"{self._embedded_micros} micros"
                    )

            except Exception as e:
                logger.error(
                    f"Node embed error: {e}"
                )

    async def _embed_micros(self, batch: list):
        """Embed a batch of micro nodes."""
        contents = [
            item.get("content", "")
            for item in batch
        ]
        contents = [c for c in contents if c]

        if not contents:
            return

        try:
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                None,
                lambda: (
                    self.encoder.embeddings
                    .embed_documents(contents)
                )
            )

            for item, vector in zip(batch, vectors):
                if not vector:
                    continue

                vec_str = (
                    "[" +
                    ",".join(str(x) for x in vector) +
                    "]"
                )

                async with self.db.pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE micro_nodes
                        SET embedding_vector = $1::vector
                        WHERE micro_id = $2
                        """,
                        vec_str,
                        item["micro_id"],
                    )

                # publish ready event
                await self.pubsub_redis.publish(
                    EMBEDDING_READY,
                    json.dumps({
                        "type": "micro",
                        "micro_id": item["micro_id"],
                        "session_id": item["session_id"],
                    })
                )

                self._embedded_micros += 1

        except Exception as e:
            logger.error(f"Micro embed error: {e}")


if __name__ == "__main__":
    service = EmbeddingService()
    asyncio.run(service.run())