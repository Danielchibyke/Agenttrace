import uuid
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any

class NodeType(Enum):
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"

@dataclass
class Node:
    node_type: NodeType
    session_id: str
    task_id: str

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    step_number: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: Optional[float] = None

    # reasoning fields
    prompt_text: Optional[str] = None
    response_text: Optional[str] = None
    model_name: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None

    # tool call fields
    tool_name: Optional[str] = None
    input_params: Optional[dict] = None
    raw_output: Optional[Any] = None
    status: Optional[str] = None
    error_message: Optional[str] = None

    # replay fields
    conversation_snapshot: Optional[list] = None
    system_prompt_snapshot: Optional[str] = None
    memory_snapshot: Optional[dict] = None
    agent_state_snapshot: Optional[dict] = None
    tool_config_snapshot: Optional[dict] = None

    # HD space field
    embedding_vector: Optional[list] = None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "step_number": self.step_number,
            "node_type": self.node_type.value,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "model_name": self.model_name,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tool_name": self.tool_name,
            "input_params": self.input_params,
            "raw_output": str(self.raw_output) if self.raw_output else None,
            "status": self.status,
            "error_message": self.error_message,
            "conversation_snapshot": self.conversation_snapshot,
            "system_prompt_snapshot": self.system_prompt_snapshot,
            "memory_snapshot": self.memory_snapshot,
            "agent_state_snapshot": self.agent_state_snapshot,
            "tool_config_snapshot": self.tool_config_snapshot,
            "embedding_vector": self.embedding_vector,
        }