import asyncio
import json
import logging
import os
from typing import Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tracer.node import Node, NodeType

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

class EmbeddingEncoder:
    """
    Converts raw node data into HD vectors.
    Runs as a second pass after capture so it never
    slows down agent execution.
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def encode_node(self, node: Node) -> list[float]:
        """
        Encode a single node into a HD vector.
        Strategy depends on node type.
        """
        text = self._build_encoding_text(node)
        if not text:
            return None
        return await self._embed(text)

    async def encode_fused_pair(
        self,
        reasoning_node: Node,
        action_node: Node
    ) -> list[float]:
        """
        Fuse a reasoning node and its resulting action node
        into a single HD vector.
        This is the core unit of your HD space —
        intent plus action together.
        """
        reasoning_text = self._build_encoding_text(reasoning_node)
        action_text = self._build_encoding_text(action_node)

        fused_text = f"""
REASONING:
{reasoning_text}

ACTION TAKEN:
{action_text}
        """.strip()

        return await self._embed(fused_text)

    async def encode_batch(
        self,
        nodes: list[Node]
    ) -> dict[str, list[float]]:
        """
        Encode a batch of nodes concurrently.
        Returns dict of node_id to embedding vector.
        """
        tasks = {
            node.node_id: self.encode_node(node)
            for node in nodes
        }

        results = {}
        completed = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True
        )

        for node_id, result in zip(tasks.keys(), completed):
            if isinstance(result, Exception):
                logger.error(
                    f"Encoding failed for node {node_id}: {result}"
                )
                results[node_id] = None
            else:
                results[node_id] = result

        return results

    def _build_encoding_text(self, node: Node) -> Optional[str]:
        """
        Build the text representation of a node
        for embedding. Different strategy per node type.
        """
        if node.node_type == NodeType.REASONING:
            parts = []
            if node.prompt_text:
                parts.append(f"PROMPT: {node.prompt_text}")
            if node.response_text:
                parts.append(f"RESPONSE: {node.response_text}")
            return "\n\n".join(parts) if parts else None

        elif node.node_type == NodeType.TOOL_CALL:
            parts = []
            if node.tool_name:
                parts.append(f"TOOL: {node.tool_name}")
            if node.input_params:
                parts.append(
                    f"INPUT: {json.dumps(node.input_params, indent=2)}"
                )
            return "\n\n".join(parts) if parts else None

        elif node.node_type == NodeType.TOOL_RESPONSE:
            parts = []
            if node.tool_name:
                parts.append(f"TOOL: {node.tool_name}")
            if node.raw_output:
                output_str = str(node.raw_output)
                # truncate very long outputs
                if len(output_str) > 2000:
                    output_str = output_str[:2000] + "...[truncated]"
                parts.append(f"OUTPUT: {output_str}")
            if node.status:
                parts.append(f"STATUS: {node.status}")
            if node.error_message:
                parts.append(f"ERROR: {node.error_message}")
            return "\n\n".join(parts) if parts else None

        return None

    async def _embed(self, text: str) -> list[float]:
        """
        Call OpenAI embedding API.
        Returns a 1536 dimensional vector.
        """
        try:
            response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            raise