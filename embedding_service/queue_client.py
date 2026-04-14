import json
import logging
import redis
from typing import Optional

logger = logging.getLogger(__name__)

EMBEDDING_QUEUE = "agenttrace:embed:queue"
EMBEDDING_READY = "agenttrace:embed:ready"
MICRO_QUEUE = "agenttrace:embed:micro_queue"


class EmbeddingQueueClient:
    """
    Used by the agent process to push content
    to the embedding service queue.
    Non blocking — fire and forget.
    Agent never waits for embedding.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
    ):
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
        )
        self._connected = False

    def connect(self) -> bool:
        try:
            self._client.ping()
            self._connected = True
            logger.info("EmbeddingQueueClient connected.")
            return True
        except Exception as e:
            logger.error(
                f"Redis connection failed: {e}"
            )
            return False

    def push_node(
        self,
        node_id: str,
        content: str,
        node_type: str,
        session_id: str,
    ):
        """
        Push a parent node for embedding.
        Returns immediately — never blocks.
        """
        if not self._connected:
            return
        try:
            payload = json.dumps({
                "type": "node",
                "node_id": node_id,
                "content": content,
                "node_type": node_type,
                "session_id": session_id,
            })
            self._client.rpush(EMBEDDING_QUEUE, payload)
        except Exception as e:
            logger.error(f"Push node failed: {e}")

    def push_micro(
        self,
        micro_id: str,
        content: str,
        session_id: str,
        parent_node_id: str,
    ):
        """
        Push a micro node token for embedding.
        Returns immediately — never blocks.
        """
        if not self._connected:
            return
        try:
            payload = json.dumps({
                "type": "micro",
                "micro_id": micro_id,
                "content": content,
                "session_id": session_id,
                "parent_node_id": parent_node_id,
            })
            self._client.rpush(MICRO_QUEUE, payload)
        except Exception as e:
            logger.error(f"Push micro failed: {e}")

    def get_queue_size(self) -> dict:
        return {
            "node_queue": self._client.llen(
                EMBEDDING_QUEUE
            ),
            "micro_queue": self._client.llen(
                MICRO_QUEUE
            ),
        }