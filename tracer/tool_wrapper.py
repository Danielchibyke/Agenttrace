import time
import logging
from langchain_classic.tools import BaseTool
from tracer.core import TracerCore

logger = logging.getLogger(__name__)


class TracedTool(BaseTool):
    """
    Wraps any LangChain tool to capture execution
    as nodes. Preserves all tool metadata so
    LangChain can still convert it to OpenAI format.
    """
    wrapped_tool: object = None
    tracer_core: object = None
    embedding_worker: object = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, *args, **kwargs):
        start_time = time.time()

        tool_input = (
            args[0] if args
            else kwargs.get("input", str(kwargs))
        )

        node = self.tracer_core.create_tool_call_node(
            tool_name=self.name,
            input_params=(
                {"input": tool_input}
                if isinstance(tool_input, str)
                else tool_input
            ),
        )
        self.tracer_core.record_node(node)

        if self.tracer_core.write_queue:
            self.tracer_core.write_queue.push(node)

        if self.embedding_worker:
            self.embedding_worker.queue_node(node)

        try:
            output = self.wrapped_tool._run(
                *args, **kwargs
            )
            latency = (time.time() - start_time) * 1000

            node.raw_output = output
            node.status = "success"
            node.latency_ms = latency

            if self.tracer_core.write_queue:
                self.tracer_core.write_queue.push(node)

            if self.embedding_worker:
                self.embedding_worker.queue_node(node)

            response_node = (
                self.tracer_core.create_tool_response_node(
                    tool_name=self.name,
                    raw_output=output,
                    status="success",
                )
            )
            self.tracer_core.record_node(response_node)

            if self.tracer_core.write_queue:
                self.tracer_core.write_queue.push(
                    response_node
                )

            if self.embedding_worker:
                self.embedding_worker.queue_node(
                    response_node
                )

            return output

        except Exception as e:
            node.status = "error"
            node.error_message = str(e)
            node.latency_ms = (
                time.time() - start_time
            ) * 1000
            if self.tracer_core.write_queue:
                self.tracer_core.write_queue.push(node)
            raise

    async def _arun(self, *args, **kwargs):
        start_time = time.time()

        tool_input = (
            args[0] if args
            else kwargs.get("input", str(kwargs))
        )

        node = self.tracer_core.create_tool_call_node(
            tool_name=self.name,
            input_params=(
                {"input": tool_input}
                if isinstance(tool_input, str)
                else tool_input
            ),
        )
        self.tracer_core.record_node(node)

        if self.tracer_core.write_queue:
            self.tracer_core.write_queue.push(node)

        if self.embedding_worker:
            self.embedding_worker.queue_node(node)

        try:
            # call the original tool's run method directly
            # bypassing _arun entirely to avoid config issues
            output = self.wrapped_tool._run(
                *args,
                **{
                    k: v for k, v in kwargs.items()
                    if k not in [
                        "config", "run_manager",
                        "callbacks"
                    ]
                }
            )

            latency = (time.time() - start_time) * 1000
            node.raw_output = output
            node.status = "success"
            node.latency_ms = latency

            if self.tracer_core.write_queue:
                self.tracer_core.write_queue.push(node)

            if self.embedding_worker:
                self.embedding_worker.queue_node(node)

            response_node = (
                self.tracer_core.create_tool_response_node(
                    tool_name=self.name,
                    raw_output=output,
                    status="success",
                )
            )
            self.tracer_core.record_node(response_node)

            if self.tracer_core.write_queue:
                self.tracer_core.write_queue.push(
                    response_node
                )

            if self.embedding_worker:
                self.embedding_worker.queue_node(
                    response_node
                )

            return output

        except Exception as e:
            node.status = "error"
            node.error_message = str(e)
            node.latency_ms = (
                time.time() - start_time
            ) * 1000
            if self.tracer_core.write_queue:
                self.tracer_core.write_queue.push(node)
            raise

def wrap_tools(
    tools: list,
    tracer_core: TracerCore,
    embedding_worker=None,
) -> list:
    """
    Wrap a list of tools for tracing.
    Preserves all tool metadata and schema.
    """
    wrapped = []
    for t in tools:
        traced = TracedTool(
            name=t.name,
            description=t.description,
            args_schema=t.args_schema,
            wrapped_tool=t,
            tracer_core=tracer_core,
            embedding_worker=embedding_worker,
        )
        wrapped.append(traced)
    return wrapped