import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from tracer.node import Node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncWriteQueue:
    """
    Bridges in-memory tree to persistent database.
    Never blocks agent execution.
    Handles retries, backoff, and emergency file overflow.
    """

    def __init__(
        self,
        db_writer,
        flush_interval_ms: int = 500,
        max_queue_size: int = 1000,
        max_retries: int = 3,
        overflow_path: str = "overflow"
    ):
        self.db_writer = db_writer
        self.flush_interval = flush_interval_ms / 1000
        self.max_queue_size = max_queue_size
        self.max_retries = max_retries
        self.overflow_path = Path(overflow_path)
        self.overflow_path.mkdir(exist_ok=True)

        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._failed_count = 0
        self._success_count = 0

    def push(self, node: Node):
        """
        Called by tracer core after every node capture.
        Non blocking. If queue is full writes to overflow file.
        """
        try:
            self._queue.put_nowait(node)
        except asyncio.QueueFull:
            logger.warning(
                f"Queue full. Writing node {node.node_id} to overflow file."
            )
            self._write_overflow(node)

    async def start(self):
        """Start the background flush loop."""
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())
        logger.info("AsyncWriteQueue started.")

    async def stop(self):
        """
        Gracefully stop the queue.
        Flushes all remaining nodes before stopping.
        """
        self._running = False
        await self._drain()
        if self._task:
            self._task.cancel()
        logger.info(
            f"AsyncWriteQueue stopped. "
            f"Success: {self._success_count} "
            f"Failed: {self._failed_count}"
        )

    async def _flush_loop(self):
        """
        Background loop that flushes nodes to database
        every flush_interval seconds.
        """
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._drain()

    async def _drain(self):
        """Flush all current nodes in queue to database."""
        nodes = []

        while not self._queue.empty():
            try:
                node = self._queue.get_nowait()
                nodes.append(node)
            except asyncio.QueueEmpty:
                break

        if not nodes:
            return

        await self._write_batch(nodes)

    async def _write_batch(self, nodes: list[Node]):
        """
        Write a batch of nodes to database with retry and backoff.
        Falls back to overflow file if all retries fail.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                await self.db_writer.write_nodes(nodes)
                self._success_count += len(nodes)
                return
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    f"DB write attempt {attempt} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)

        logger.error(
            f"All {self.max_retries} retries failed. "
            f"Writing {len(nodes)} nodes to overflow."
        )
        for node in nodes:
            self._write_overflow(node)
            self._failed_count += len(nodes)

    def _write_overflow(self, node: Node):
        """
        Emergency fallback. Writes node to local file
        when database is unreachable and queue is full.
        """
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_path = self.overflow_path / f"overflow_{timestamp}.jsonl"
            with open(file_path, "a") as f:
                f.write(json.dumps(node.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Overflow write failed: {e}")

    def status(self) -> dict:
        return {
            "queue_size": self._queue.qsize(),
            "running": self._running,
            "success_count": self._success_count,
            "failed_count": self._failed_count,
            "overflow_path": str(self.overflow_path),
        }