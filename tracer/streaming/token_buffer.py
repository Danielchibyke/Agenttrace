import asyncio
import logging
from collections import deque
from typing import Callable, Optional
from tracer.streaming.micro_node import MicroNode

logger = logging.getLogger(__name__)


class TokenBuffer:
    """
    Fast in-memory buffer for incoming tokens.
    Sits between token capture and the two tracks.
    Never blocks the token stream.

    Track A — dispatches to visualizer immediately
    Track B — batches for embedding every batch_interval seconds
    """

    def __init__(
        self,
        session_id: str,
        batch_size: int = 20,
        batch_interval_ms: int = 100,
        on_visualizer_dispatch=None,
        on_batch_ready=None,
        db_writer=None,
    ):
        self.session_id = session_id
        self.batch_size = batch_size
        self.batch_interval = batch_interval_ms / 1000
        self.on_visualizer_dispatch = on_visualizer_dispatch
        self.on_batch_ready = on_batch_ready
        self._db_writer = db_writer

        self._buffer: deque = deque()
        self._pending_batch: list[MicroNode] = []
        self._all_micro_nodes: list[MicroNode] = []
        self._running = False
        self._task = None
        self._lock = asyncio.Lock()
        self._loop = None
    async def start(self):
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._task = asyncio.create_task(self._batch_loop())
        logger.info("TokenBuffer started.")

    async def stop(self):
        self._running = False
        # flush remaining
        await self._flush_batch()
        if self._task:
            self._task.cancel()
        logger.info(
            f"TokenBuffer stopped. "
            f"Total micro nodes: {len(self._all_micro_nodes)}"
        )

    def push(self, micro_node: MicroNode):
        """
        Called for every token captured.
        Non blocking. Thread safe.
        Works from both sync and async contexts.
        """
        self._all_micro_nodes.append(micro_node)

        try:
            loop = self._loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._async_push(micro_node), loop
                )
            else:
                self._pending_batch.append(micro_node)
        except Exception as e:
            logger.error(f"Push failed: {e}")
            self._pending_batch.append(micro_node)

    async def _async_push(self, micro_node: MicroNode):
        """
        Async push — runs in the main event loop.
        Writes skeleton immediately to database.
        Queues for batch embedding.
        """
        async with self._lock:
            self._buffer.append(micro_node)

        # write skeleton immediately if db_writer available
        if hasattr(self, '_db_writer') and self._db_writer:
            try:
                await self._db_writer.upsert_micro_node(
                    micro_node
                )
            except Exception as e:
                logger.error(
                    f"Micro skeleton write failed: {e}"
                )

    def _dispatch_visualizer(self, micro_node: MicroNode):
        """Fire and forget to visualizer."""
        micro_node.dispatched_to_visualizer = True
        if self.on_visualizer_dispatch:
            asyncio.ensure_future(
                self.on_visualizer_dispatch(micro_node)
            )

    async def _batch_loop(self):
        """
        Background loop for Track B.
        Collects tokens and fires batch callback
        every batch_interval seconds or when batch_size reached.
        """
        while self._running:
            await asyncio.sleep(self.batch_interval)
            await self._flush_batch()

    async def _flush_batch(self):
        """Collect buffered tokens and fire batch callback."""
        async with self._lock:
            if not self._buffer:
                return

            batch = []
            while self._buffer and len(batch) < self.batch_size:
                batch.append(self._buffer.popleft())

            if batch and self.on_batch_ready:
                try:
                    await self.on_batch_ready(batch)
                except Exception as e:
                    logger.error(f"Batch callback failed: {e}")

    def get_all_micro_nodes(self) -> list[MicroNode]:
        return list(self._all_micro_nodes)

    def pending_count(self) -> int:
        return len(self._buffer)