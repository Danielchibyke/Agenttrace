from abc import ABC, abstractmethod
from tracer.node import Node

class BaseAdapter(ABC):
    """
    Every framework adapter must implement this interface.
    The adapter's only job is to translate framework-specific
    events into universal Node objects and pass them to the
    tracer core.
    """

    def __init__(self, tracer_core):
        self.core = tracer_core

    @abstractmethod
    def on_reasoning_start(self, prompt: str, **kwargs) -> Node:
        """
        Called when the LLM starts generating a reasoning step.
        Must return a Node with node_type REASONING.
        """
        pass

    @abstractmethod
    def on_reasoning_end(self, node: Node, response: str, **kwargs) -> Node:
        """
        Called when the LLM finishes generating a reasoning step.
        Must update and return the node with response and latency.
        """
        pass

    @abstractmethod
    def on_tool_start(self, tool_name: str, input_params: dict, **kwargs) -> Node:
        """
        Called when the agent starts executing a tool.
        Must return a Node with node_type TOOL_CALL.
        """
        pass

    @abstractmethod
    def on_tool_end(self, node: Node, output: str, **kwargs) -> Node:
        """
        Called when a tool finishes executing.
        Must update and return the node with output and latency.
        """
        pass

    @abstractmethod
    def on_error(self, node: Node, error: Exception, **kwargs) -> Node:
        """
        Called when any step fails.
        Must update and return the node with error details.
        """
        pass