import time
import asyncio
import logging
from langchain_classic.tools import BaseTool
from tracer.node import NodeType
from tracer.core import TracerCore

logger = logging.getLogger(__name__)


def wrap_tool(tool: BaseTool, tracer_core: TracerCore, embedding_worker=None) -> BaseTool:
    """
    Wraps a LangChain tool to capture execution
    as nodes regardless of how the agent invokes it.
    Works with any agent type.
    """
    original_run = tool._run
    original_arun = tool._arun

    def traced_run(tool_input, **kwargs):
        start_time = time.time()

        # create tool call node immediately
        node = tracer_core.create_tool_call_node(
            tool_name=tool.name,
            input_params={"input": tool_input},
        )
        tracer_core.record_node(node)

        if tracer_core.write_queue:
            tracer_core.write_queue.push(node)

        if embedding_worker:
            embedding_worker.queue_node(node)

        try:
            output = original_run(tool_input, **kwargs)
            latency = (time.time() - start_time) * 1000

            # update with result
            node.raw_output = output
            node.status = "success"
            node.latency_ms = latency

            if tracer_core.write_queue:
                tracer_core.write_queue.push(node)

            if embedding_worker:
                embedding_worker.queue_node(node)

            # create response node
            response_node = (
                tracer_core.create_tool_response_node(
                    tool_name=tool.name,
                    raw_output=output,
                    status="success",
                )
            )
            tracer_core.record_node(response_node)

            if tracer_core.write_queue:
                tracer_core.write_queue.push(response_node)

            if embedding_worker:
                embedding_worker.queue_node(response_node)

            return output

        except Exception as e:
            node.status = "error"
            node.error_message = str(e)
            node.latency_ms = (
                time.time() - start_time
            ) * 1000

            if tracer_core.write_queue:
                tracer_core.write_queue.push(node)

            raise

    async def traced_arun(tool_input, **kwargs):
        start_time = time.time()

        node = tracer_core.create_tool_call_node(
            tool_name=tool.name,
            input_params={"input": tool_input},
        )
        tracer_core.record_node(node)

        if tracer_core.write_queue:
            tracer_core.write_queue.push(node)

        if embedding_worker:
            embedding_worker.queue_node(node)

        try:
            if original_arun:
                output = await original_arun(
                    tool_input, **kwargs
                )
            else:
                output = original_run(
                    tool_input, **kwargs
                )

            latency = (time.time() - start_time) * 1000
            node.raw_output = output
            node.status = "success"
            node.latency_ms = latency

            if tracer_core.write_queue:
                tracer_core.write_queue.push(node)

            if embedding_worker:
                embedding_worker.queue_node(node)

            response_node = (
                tracer_core.create_tool_response_node(
                    tool_name=tool.name,
                    raw_output=output,
                    status="success",
                )
            )
            tracer_core.record_node(response_node)

            if tracer_core.write_queue:
                tracer_core.write_queue.push(response_node)

            if embedding_worker:
                embedding_worker.queue_node(response_node)

            return output

        except Exception as e:
            node.status = "error"
            node.error_message = str(e)
            node.latency_ms = (
                time.time() - start_time
            ) * 1000

            if tracer_core.write_queue:
                tracer_core.write_queue.push(node)

            raise

    tool._run = traced_run
    tool._arun = traced_arun
    return tool


def wrap_tools(
    tools: list,
    tracer_core: TracerCore,
    embedding_worker=None,
) -> list:
    """Wrap a list of tools for tracing."""
    return [
        wrap_tool(t, tracer_core, embedding_worker)
        for t in tools
    ]