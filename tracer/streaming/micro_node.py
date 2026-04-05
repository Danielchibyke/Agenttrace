import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class MicroNode:
    """
    Represents a single token or execution substep.
    Child of a parent Node in the causal tree.
    Much lighter than Node — optimized for high volume capture.
    """

    parent_node_id: str
    session_id: str
    task_id: str
    content: str
    index: int

    micro_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = field(
        default_factory=datetime.utcnow
    )
    token_index: int = 0
    logprob: Optional[float] = None
    embedding_vector: Optional[list] = None

    # track A — sent to visualizer immediately
    dispatched_to_visualizer: bool = False

    # track B — embedding status
    embedding_queued: bool = False
    embedding_complete: bool = False

    def to_dict(self) -> dict:
        return {
            "micro_id": self.micro_id,
            "parent_node_id": self.parent_node_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "content": self.content,
            "index": self.index,
            "token_index": self.token_index,
            "timestamp": self.timestamp.isoformat(),
            "logprob": self.logprob,
            "embedding_vector": self.embedding_vector,
        }

    def to_visualizer_event(self) -> dict:
        """
        Lightweight payload for Track A.
        Sent immediately to frontend without embedding.
        Position computed later when embedding arrives.
        """
        return {
            "micro_id": self.micro_id,
            "parent_node_id": self.parent_node_id,
            "session_id": self.session_id,
            "content": self.content,
            "index": self.index,
            "token_index": self.token_index,
            "timestamp": self.timestamp.isoformat(),
            "has_embedding": False,
            "x": None,
            "y": None,
            "z": None,
        }