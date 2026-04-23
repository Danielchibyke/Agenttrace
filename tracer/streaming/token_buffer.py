import asyncio
import logging
import uuid
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
        parent_node_id: str = None,  # ADDED: Required parent node ID
        batch_size: int = 20,
        batch_interval_ms: int = 100,
        on_visualizer_dispatch=None,
        on_batch_ready=None,
        db_writer=None,
        embedding_worker=None
    ):
        self.session_id = session_id
        self.parent_node_id = parent_node_id  # ADDED: Store parent node ID
        self.batch_size = batch_size
        self.batch_interval = batch_interval_ms / 1000
        self.on_visualizer_dispatch = on_visualizer_dispatch
        self.on_batch_ready = on_batch_ready
        self._db_writer = db_writer
        self._embedding_worker = embedding_worker

        self._buffer: deque = deque()
        self._pending_batch: list[MicroNode] = []
        self._all_micro_nodes: list[MicroNode] = []
        self._running = False
        self._task = None
        self._lock = asyncio.Lock()
        self._loop = None
        self._parent_ensured = False  # ADDED: Track if parent exists
        self._pending_parent_creation = False  # ADDED: Prevent duplicate parent creation

    async def start(self):
        self._running = True
        self._loop = asyncio.get_event_loop()
        
        # ENSURE PARENT NODE EXISTS BEFORE PROCESSING ANY MICRO NODES
        await self._ensure_parent_node()
        
        self._task = asyncio.create_task(self._batch_loop())
        logger.info(f"TokenBuffer started with parent node: {self.parent_node_id}")

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

    async def _ensure_parent_node(self):
        """
        Ensure the parent node exists in the database before any micro nodes are written.
        """
        if self._parent_ensured or self._pending_parent_creation:
            return
            
        self._pending_parent_creation = True
        
        try:
            if not self.parent_node_id:
                # Generate a parent node ID if none provided
                self.parent_node_id = str(uuid.uuid4())
                logger.warning(f"No parent_node_id provided, generated: {self.parent_node_id}")
            
            if self._db_writer:
                # Check if parent exists
                async with self._db_writer.pool.acquire() as conn:
                    exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM nodes WHERE node_id = $1)",
                        self.parent_node_id
                    )
                    
                    if not exists:
                        # Create the parent node
                        await conn.execute(
                            """
                            INSERT INTO nodes (node_id, session_id, node_type, content, status, created_at)
                            VALUES ($1, $2, $3, $4, $5, NOW())
                            ON CONFLICT (node_id) DO NOTHING
                            """,
                            self.parent_node_id,
                            self.session_id,
                            "llm_streaming",
                            "LLM streaming response",
                            "streaming"
                        )
                        logger.info(f"Created parent node: {self.parent_node_id}")
                    else:
                        logger.info(f"Parent node exists: {self.parent_node_id}")
                        
                self._parent_ensured = True
        except Exception as e:
            logger.error(f"Failed to ensure parent node: {e}")
            # Even if it fails, mark as ensured to prevent infinite loops
            self._parent_ensured = True
        finally:
            self._pending_parent_creation = False

    def push(self, micro_node: MicroNode):
        """
        Called for every token captured.
        Non blocking. Thread safe.
        Works from both sync and async contexts.
        """
        # Set the parent_node_id on the micro_node if not already set
        if not micro_node.parent_node_id and self.parent_node_id:
            micro_node.parent_node_id = self.parent_node_id
            
        self._all_micro_nodes.append(micro_node)

        try:
            loop = self._loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._async_push(micro_node), loop
                )
            else:
                # If loop not running, queue for later
                if not micro_node.parent_node_id and self.parent_node_id:
                    micro_node.parent_node_id = self.parent_node_id
                self._pending_batch.append(micro_node)
        except Exception as e:
            logger.error(f"Push failed: {e}")
            if not micro_node.parent_node_id and self.parent_node_id:
                micro_node.parent_node_id = self.parent_node_id
            self._pending_batch.append(micro_node)

    async def _async_push(self, micro_node: MicroNode):
        """
        Async push — runs in the main event loop.
        Writes skeleton immediately to database.
        Queues for batch embedding.
        """
        # Ensure parent node ID is set
        if not micro_node.parent_node_id and self.parent_node_id:
            micro_node.parent_node_id = self.parent_node_id
            
        async with self._lock:
            self._buffer.append(micro_node)

        # Write skeleton immediately if db_writer available
        if self._db_writer:
            try:
                # Ensure parent exists before writing
                await self._ensure_parent_node()
                
                # Now write the micro node
                await self._db_writer.upsert_micro_node(micro_node)
            except Exception as e:
                logger.error(f"Micro skeleton write failed: {e}")
                # Queue for retry
                self._pending_batch.append(micro_node)
                
        # Push to Redis embedding queue immediately
        if self._embedding_worker:
            try:
                self._embedding_worker.queue_micro(micro_node)
            except Exception as e:
                logger.error(f"Failed to queue micro node for embedding: {e}")

    def _dispatch_visualizer(self, micro_node: MicroNode):
        """Fire and forget to visualizer."""
        micro_node.dispatched_to_visualizer = True
        if self.on_visualizer_dispatch:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.on_visualizer_dispatch(micro_node))
                else:
                    asyncio.run(self.on_visualizer_dispatch(micro_node))
            except Exception as e:
                logger.error(f"Visualizer dispatch failed: {e}")

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
        # Also process any pending micro nodes
        if self._pending_batch:
            async with self._lock:
                for node in self._pending_batch:
                    self._buffer.append(node)
                self._pending_batch.clear()
                
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
        return len(self._buffer) + len(self._pending_batch)