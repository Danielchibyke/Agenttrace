import time
import asyncio
import logging
from langchain_classic.callbacks.base import BaseCallbackHandler
from tracer.adapters.base import BaseAdapter
from tracer.node import Node, NodeType
from tracer.core import TracerCore
from tracer.streaming.token_buffer import TokenBuffer
from tracer.streaming.micro_node import MicroNode

logger = logging.getLogger(__name__)


class LangChainAdapter(BaseCallbackHandler, BaseAdapter):

    def __init__(
        self,
        tracer_core: TracerCore,
        token_buffer: TokenBuffer = None,
        embedding_worker=None,
    ):
        BaseCallbackHandler.__init__(self)
        BaseAdapter.__init__(self, tracer_core)
        self.token_buffer = token_buffer
        self.embedding_worker = embedding_worker
        self._active_reasoning_node: Node = None
        self._active_tool_node: Node = None
        self._reasoning_start_time: float = None
        self._tool_start_time: float = None
        self._token_index: int = 0

    def _upsert_now(self, node: Node):
        """
        Write node to database immediately.
        Also queue for immediate background embedding.
        """
        if self.core.write_queue:
            self.core.write_queue.push(node)
        if self.embedding_worker:
            self.embedding_worker.queue_node(node)
    # --- reasoning events ---

    def on_llm_start(
        self, serialized: dict, prompts: list, **kwargs
    ):
        self._reasoning_start_time = time.time()
        self._token_index = 0
        prompt_text = prompts[0] if prompts else ""
        model_name = serialized.get("name", "unknown")

        node = self.core.create_reasoning_node(
            prompt_text=prompt_text,
            model_name=model_name,
            conversation_snapshot=prompts,
        )
        self._active_reasoning_node = (
            self.core.record_node(node)
        )

        # write skeleton immediately — no details yet
        self._upsert_now(self._active_reasoning_node)

    def on_llm_new_token(self, token: str, **kwargs):
        if not self._active_reasoning_node:
            return
        if not token:
            return

        micro_node = MicroNode(
            parent_node_id=(
                self._active_reasoning_node.node_id
            ),
            session_id=self.core.session_id,
            task_id=self.core.task_id,
            content=token,
            index=self._token_index,
            token_index=self._token_index,
        )
        self._token_index += 1

        if self.token_buffer:
            self.token_buffer.push(micro_node)
        # queue for immediate embedding
        if self.embedding_worker:
            self.embedding_worker.queue_micro(micro_node)    

    def on_llm_end(self, response, **kwargs):
        if not self._active_reasoning_node:
            return

        latency = (
            time.time() - self._reasoning_start_time
        ) * 1000
        response_text = ""
        tokens_in = None
        tokens_out = None

        if response.generations:
            response_text = (
                response.generations[0][0].text
            )
        if response.llm_output:
            usage = response.llm_output.get(
                "token_usage", {}
            )
            tokens_in = usage.get("prompt_tokens")
            tokens_out = usage.get("completion_tokens")

        # update node with completed details
        self._active_reasoning_node.response_text = (
            response_text
        )
        self._active_reasoning_node.latency_ms = latency
        self._active_reasoning_node.tokens_in = tokens_in
        self._active_reasoning_node.tokens_out = tokens_out
        self._active_reasoning_node.status = "success"

        # push updated node — upsert merges details
        self._upsert_now(self._active_reasoning_node)

        self.on_reasoning_end(
            self._active_reasoning_node,
            response_text
        )

    def on_llm_error(self, error: Exception, **kwargs):
        if self._active_reasoning_node:
            self._active_reasoning_node.status = "error"
            self._active_reasoning_node.error_message = (
                str(error)
            )
            self._upsert_now(self._active_reasoning_node)

    # --- tool events ---

    def on_tool_start(
        self, serialized: dict, input_str: str, **kwargs
    ):
        print(f"[TOOL START FIRED] {serialized} {kwargs}")
        self._tool_start_time = time.time()
        

        tool_name = (
            kwargs.get("name") or
            serialized.get("name") or
            (
                serialized.get("id", ["unknown"])[-1]
                if serialized.get("id") else None
            ) or
            "unknown_tool"
        )
        node = self.core.create_tool_call_node(
            tool_name=tool_name,
            input_params={"input": input_str},
        )
        self._active_tool_node = (
            self.core.record_node(node)
        )

        # write skeleton immediately
        self._upsert_now(self._active_tool_node)

    def on_tool_end(self, output: str, **kwargs):
        print(f"[TOOL END FIRED] {output[:50]}")
        if not self._active_tool_node:
            return

        latency = (
            time.time() - self._tool_start_time
        ) * 1000
        self._active_tool_node.latency_ms = latency
        self._active_tool_node.raw_output = output
        self._active_tool_node.status = "success"

        # push completed tool call
        self._upsert_now(self._active_tool_node)

        # create and immediately write response node
        response_node = self.core.create_tool_response_node(
            tool_name=self._active_tool_node.tool_name,
            raw_output=output,
            status="success",
        )
        self.core.record_node(response_node)
        self._upsert_now(response_node)

    def on_tool_error(self, error: Exception, **kwargs):
        if self._active_tool_node:
            self._active_tool_node.status = "error"
            self._active_tool_node.error_message = (
                str(error)
            )
            self._upsert_now(self._active_tool_node)

    # --- base adapter implementation ---

    def on_reasoning_start(
        self, prompt: str, **kwargs
    ) -> Node:
        pass

    def on_reasoning_end(
        self, node: Node, response: str, **kwargs
    ) -> Node:
        return node

    def on_error(
        self, node: Node, error: Exception, **kwargs
    ) -> Node:
        node.status = "error"
        node.error_message = str(error)
        return node