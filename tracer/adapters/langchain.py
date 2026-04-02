import time
from typing import Any, Union
from langchain.callbacks.base import BaseCallbackHandler
from tracer.adapters.base import BaseAdapter
from tracer.node import Node, NodeType
from tracer.core import TracerCore

class LangChainAdapter(BaseCallbackHandler, BaseAdapter):
    """
    Translates LangChain callback events into universal
    Node objects and passes them to TracerCore.
    """

    def __init__(self, tracer_core: TracerCore):
        BaseCallbackHandler.__init__(self)
        BaseAdapter.__init__(self, tracer_core)
        self._active_reasoning_node: Node = None
        self._active_tool_node: Node = None
        self._reasoning_start_time: float = None
        self._tool_start_time: float = None

    # --- reasoning events ---

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        self._reasoning_start_time = time.time()
        prompt_text = prompts[0] if prompts else ""
        model_name = serialized.get("name", "unknown")

        node = self.core.create_reasoning_node(
            prompt_text=prompt_text,
            model_name=model_name,
            conversation_snapshot=prompts,
        )
        self._active_reasoning_node = self.core.record_node(node)

    def on_llm_end(self, response, **kwargs):
        if not self._active_reasoning_node:
            return

        latency = (time.time() - self._reasoning_start_time) * 1000
        response_text = ""
        tokens_in = None
        tokens_out = None

        if response.generations:
            response_text = response.generations[0][0].text

        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            tokens_in = usage.get("prompt_tokens")
            tokens_out = usage.get("completion_tokens")

        self._active_reasoning_node.response_text = response_text
        self._active_reasoning_node.latency_ms = latency
        self._active_reasoning_node.tokens_in = tokens_in
        self._active_reasoning_node.tokens_out = tokens_out

        self.on_reasoning_end(
            self._active_reasoning_node,
            response_text
        )

    def on_llm_error(self, error: Exception, **kwargs):
        if self._active_reasoning_node:
            self.on_error(self._active_reasoning_node, error)

    # --- tool events ---

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        self._tool_start_time = time.time()
        tool_name = serialized.get("name", "unknown")

        node = self.core.create_tool_call_node(
            tool_name=tool_name,
            input_params={"input": input_str},
        )
        self._active_tool_node = self.core.record_node(node)

    def on_tool_end(self, output: str, **kwargs):
        if not self._active_tool_node:
            return

        latency = (time.time() - self._tool_start_time) * 1000
        self._active_tool_node.latency_ms = latency
        self._active_tool_node.raw_output = output
        self._active_tool_node.status = "success"

        response_node = self.core.create_tool_response_node(
            tool_name=self._active_tool_node.tool_name,
            raw_output=output,
            status="success",
        )
        self.core.record_node(response_node)

    def on_tool_error(self, error: Exception, **kwargs):
        if self._active_tool_node:
            self._active_tool_node.status = "error"
            self.on_error(self._active_tool_node, error)

    # --- base adapter implementation ---

    def on_reasoning_start(self, prompt: str, **kwargs) -> Node:
        pass

    def on_reasoning_end(self, node: Node, response: str, **kwargs) -> Node:
        return node

    def on_error(self, node: Node, error: Exception, **kwargs) -> Node:
        node.status = "error"
        node.error_message = str(error)
        return node