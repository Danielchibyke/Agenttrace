from typing import Optional
from tracer.node import Node

class ExecutionTree:
    def __init__(self, session_id: str, task_id: str):
        self.session_id = session_id
        self.task_id = task_id
        self.nodes: dict[str, Node] = {}
        self.root_id: Optional[str] = None
        self.current_node_id: Optional[str] = None
        self.step_counter: int = 0

    def add_node(self, node: Node) -> Node:
        node.step_number = self.step_counter
        self.step_counter += 1

        if self.root_id is None:
            self.root_id = node.node_id

        self.nodes[node.node_id] = node
        self.current_node_id = node.node_id
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_current_node(self) -> Optional[Node]:
        if self.current_node_id:
            return self.nodes.get(self.current_node_id)
        return None

    def get_children(self, node_id: str) -> list[Node]:
        return [
            node for node in self.nodes.values()
            if node.parent_id == node_id
        ]

    def get_path_to_root(self, node_id: str) -> list[Node]:
        path = []
        current = self.nodes.get(node_id)
        while current:
            path.append(current)
            if current.parent_id:
                current = self.nodes.get(current.parent_id)
            else:
                break
        return list(reversed(path))

    def get_depth(self, node_id: str) -> int:
        return len(self.get_path_to_root(node_id)) - 1

    def get_all_nodes(self) -> list[Node]:
        return list(self.nodes.values())

    def get_reasoning_hop_count(self) -> int:
        return sum(
            1 for node in self.nodes.values()
            if node.node_type.value == "reasoning"
        )

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "total_nodes": len(self.nodes),
            "reasoning_hops": self.get_reasoning_hop_count(),
            "root_id": self.root_id,
            "current_node_id": self.current_node_id,
        }