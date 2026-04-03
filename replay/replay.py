import asyncio
import logging
import os
from typing import Optional, AsyncGenerator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from storage.database import DatabaseWriter
from tracer.node import NodeType

load_dotenv()
logger = logging.getLogger(__name__)


class ReplayEngine:
    """
    Enables two replay modes for any captured session:

    OBSERVE MODE  — scrub to any node and inspect full
                    agent state at that exact moment.
                    Read only. No re-execution.

    EXECUTE MODE  — roll back to any node and re-run
                    the agent forward from that exact
                    state. Full execution resumes.
    """

    def __init__(self, db_writer: DatabaseWriter):
        self.db = db_writer

    # -------------------------------------------------
    # OBSERVE MODE
    # -------------------------------------------------

    async def observe(
        self,
        session_id: str,
        node_id: str = None
    ) -> dict:
        """
        Scrub to any point in a session and inspect
        the full agent state at that moment.

        If node_id is None returns the full session.
        If node_id is provided returns state up to
        and including that node.
        """
        nodes = await self.db.get_session_nodes(session_id)

        if not nodes:
            return {"error": f"No nodes found for session {session_id}"}

        if node_id is None:
            return self._build_observation(nodes, nodes[-1]["node_id"])

        node_ids = [n["node_id"] for n in nodes]
        if node_id not in node_ids:
            return {"error": f"Node {node_id} not found in session"}

        return self._build_observation(nodes, node_id)

    def _build_observation(
        self,
        all_nodes: list[dict],
        target_node_id: str
    ) -> dict:
        """
        Build a full state snapshot at a target node.
        Includes everything the agent knew at that point.
        """
        target_index = next(
            i for i, n in enumerate(all_nodes)
            if n["node_id"] == target_node_id
        )

        nodes_up_to_target = all_nodes[:target_index + 1]
        target_node = all_nodes[target_index]

        reasoning_nodes = [
            n for n in nodes_up_to_target
            if n["node_type"] == NodeType.REASONING.value
        ]
        tool_call_nodes = [
            n for n in nodes_up_to_target
            if n["node_type"] == NodeType.TOOL_CALL.value
        ]
        error_nodes = [
            n for n in nodes_up_to_target
            if n.get("status") == "error"
        ]

        return {
            "session_id": target_node["session_id"],
            "task_id": target_node["task_id"],
            "target_node_id": target_node_id,
            "target_node_type": target_node["node_type"],
            "step_number": target_node["step_number"],
            "timestamp": str(target_node["timestamp"]),

            # full state at this point
            "conversation_snapshot": target_node.get(
                "conversation_snapshot"
            ),
            "system_prompt_snapshot": target_node.get(
                "system_prompt_snapshot"
            ),
            "agent_state_snapshot": target_node.get(
                "agent_state_snapshot"
            ),

            # execution summary up to this point
            "summary": {
                "total_steps": len(nodes_up_to_target),
                "reasoning_steps": len(reasoning_nodes),
                "tool_calls": len(tool_call_nodes),
                "errors": len(error_nodes),
                "last_tool_used": tool_call_nodes[-1]["tool_name"]
                    if tool_call_nodes else None,
                "last_reasoning": reasoning_nodes[-1]["response_text"]
                    if reasoning_nodes else None,
            },

            # causal chain to this node
            "path": [
                {
                    "node_id": n["node_id"],
                    "node_type": n["node_type"],
                    "step_number": n["step_number"],
                    "tool_name": n.get("tool_name"),
                    "status": n.get("status"),
                    "latency_ms": n.get("latency_ms"),
                }
                for n in nodes_up_to_target
            ]
        }

    # -------------------------------------------------
    # EXECUTE MODE
    # -------------------------------------------------

    async def re_execute(
        self,
        session_id: str,
        node_id: str,
        tools: list[BaseTool],
        tracer_core=None
    ) -> AsyncGenerator[dict, None]:
        """
        Roll back to any node and re-run the agent
        forward from that exact state.

        Reconstructs full conversation history,
        system prompt, and agent state from the
        snapshot stored at that node.

        Yields execution events as they happen
        so you can watch the re-execution live.
        """
        node = await self.db.get_node(node_id)

        if not node:
            yield {"error": f"Node {node_id} not found"}
            return

        # reconstruct agent context from snapshot
        system_prompt = node.get("system_prompt_snapshot") or (
            "You are a helpful AI assistant."
        )
        conversation_snapshot = node.get("conversation_snapshot") or []
        agent_state = node.get("agent_state_snapshot") or {}

        logger.info(
            f"Re-executing from node {node_id} "
            f"at step {node['step_number']}"
        )

        yield {
            "event": "re_execute_start",
            "from_node_id": node_id,
            "from_step": node["step_number"],
            "reconstructed_context_length": len(conversation_snapshot),
            "agent_state": agent_state,
        }

        # rebuild conversation history from snapshot
        messages = self._reconstruct_messages(
            conversation_snapshot,
            system_prompt
        )

        # build prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # build llm
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # attach tracer if provided
        callbacks = []
        if tracer_core:
            from tracer.adapters.langchain import LangChainAdapter
            callbacks.append(LangChainAdapter(tracer_core))

        # build and run agent
        agent = create_openai_tools_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            callbacks=callbacks,
            verbose=True,
            return_intermediate_steps=True
        )

        # get the original task goal from agent state
        task_goal = agent_state.get("task_goal", "Continue the task.")

        try:
            result = await executor.ainvoke(
                {
                    "input": task_goal,
                    "chat_history": messages,
                },
                config={"callbacks": callbacks}
            )

            yield {
                "event": "re_execute_complete",
                "output": result.get("output"),
                "intermediate_steps": len(
                    result.get("intermediate_steps", [])
                ),
            }

        except Exception as e:
            logger.error(f"Re-execution failed: {e}")
            yield {
                "event": "re_execute_error",
                "error": str(e),
            }

    def _reconstruct_messages(
        self,
        conversation_snapshot: list,
        system_prompt: str
    ) -> list:
        """
        Rebuild LangChain message objects from
        the stored conversation snapshot.
        """
        messages = []
        for msg in conversation_snapshot:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        return messages