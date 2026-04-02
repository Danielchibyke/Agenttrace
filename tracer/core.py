import uuid
from datetime import datetime
from tracer.node import Node, NodeType
from tracer.tree import ExecutionTree

class TracerCore:
    """
    Framework agnostic tracer core.
    Receives Node objects from any adapter and manages
    the execution tree and write queue.
    Does not know or care which framework is running.
    """

    def __init__(self, task_id: str = None, write_queue=None):
        self.session_id = str(uuid.uuid4())
        self.task_id = task_id or str(uuid.uuid4())
        self.tree = ExecutionTree(
            session_id=self.session_id,
            task_id=self.task_id
        )
        self.write_queue = write_queue
        self._current_parent_id = None

    def record_node(self, node: Node) -> Node:
        node.session_id = self.session_id
        node.task_id = self.task_id
        node.parent_id = self._current_parent_id
        node.timestamp = datetime.utcnow()

        self.tree.add_node(node)
        self._current_parent_id = node.node_id

        if self.write_queue:
            self.write_queue.push(node)

        return node

    def create_reasoning_node(
        self,
        prompt_text: str,
        model_name: str = None,
        conversation_snapshot: list = None,
        system_prompt_snapshot: str = None,
    ) -> Node:
        return Node(
            node_type=NodeType.REASONING,
            session_id=self.session_id,
            task_id=self.task_id,
            prompt_text=prompt_text,
            model_name=model_name,
            conversation_snapshot=conversation_snapshot,
            system_prompt_snapshot=system_prompt_snapshot,
        )

    def create_tool_call_node(
        self,
        tool_name: str,
        input_params: dict,
        tool_config_snapshot: dict = None,
    ) -> Node:
        return Node(
            node_type=NodeType.TOOL_CALL,
            session_id=self.session_id,
            task_id=self.task_id,
            tool_name=tool_name,
            input_params=input_params,
            tool_config_snapshot=tool_config_snapshot,
        )

    def create_tool_response_node(
        self,
        tool_name: str,
        raw_output: str,
        status: str = "success",
        error_message: str = None,
    ) -> Node:
        return Node(
            node_type=NodeType.TOOL_RESPONSE,
            session_id=self.session_id,
            task_id=self.task_id,
            tool_name=tool_name,
            raw_output=raw_output,
            status=status,
            error_message=error_message,
        )

    def set_parent(self, node_id: str):
        """
        Manually set the current parent node.
        Used when reasoning leads to reasoning
        instead of a tool call.
        """
        self._current_parent_id = node_id

    def get_summary(self) -> dict:
        return self.tree.summary()

    def get_tree(self) -> ExecutionTree:
        return self.tree