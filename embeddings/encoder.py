import asyncio
import json
import logging
import os
from typing import Optional
from dotenv import load_dotenv
from tracer.node import Node, NodeType

load_dotenv()
logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Converts raw node data into HD vectors.
    Uses Gemini text-embedding-004 by default.
    Falls back to Ollama nomic-embed-text if no API key.
    """

    def __init__(self):
        self._setup_embeddings()

    def _setup_embeddings(self):
        api_key = os.getenv("GOOGLE_API_KEY")

        if api_key:
            try:
                from langchain_google_genai import (
                    GoogleGenerativeAIEmbeddings
                )
                self.embeddings = (
                    GoogleGenerativeAIEmbeddings(
                        model="gemini-embedding-001",
                        google_api_key=api_key,
                        dim=3072
                    )
                )
                self._provider = "gemini"
                self._dim = 3072
                logger.info(
                    "EmbeddingEncoder using Gemini "
                    "text-embedding-004."
                )
                return
            except Exception as e:
                logger.warning(
                    f"Gemini embedding init failed: {e}. "
                    f"Falling back to Ollama."
                )

        from langchain_ollama import OllamaEmbeddings
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        self._provider = "ollama"
        self._dim = 768
        logger.info(
            "EmbeddingEncoder using Ollama "
            "nomic-embed-text."
        )

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def dim(self) -> int:
        return self._dim

    async def encode_node(
        self, node: Node
    ) -> Optional[list[float]]:
        text = self._build_encoding_text(node)
        if not text:
            return None
        return await self._embed(text)

    async def encode_fused_pair(
        self,
        reasoning_node: Node,
        action_node: Node
    ) -> Optional[list[float]]:
        reasoning_text = self._build_encoding_text(
            reasoning_node
        )
        action_text = self._build_encoding_text(
            action_node
        )
        if not reasoning_text and not action_text:
            return None

        fused_text = f"""
REASONING:
{reasoning_text or ''}

ACTION TAKEN:
{action_text or ''}
        """.strip()

        return await self._embed(fused_text)

    async def encode_batch(
        self,
        nodes: list[Node]
    ) -> dict[str, Optional[list[float]]]:
        tasks = {
            node.node_id: self.encode_node(node)
            for node in nodes
        }

        results = {}
        completed = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True
        )

        for node_id, result in zip(
            tasks.keys(), completed
        ):
            if isinstance(result, Exception):
                logger.error(
                    f"Encoding failed for "
                    f"node {node_id}: {result}"
                )
                results[node_id] = None
            else:
                results[node_id] = result

        return results

    def _build_encoding_text(
        self, node: Node
    ) -> Optional[str]:
        if node.node_type == NodeType.REASONING:
            parts = []
            if node.prompt_text:
                parts.append(
                    f"PROMPT: {node.prompt_text[:500]}"
                )
            if node.response_text:
                parts.append(
                    f"RESPONSE: {node.response_text[:500]}"
                )
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
                if len(output_str) > 2000:
                    output_str = (
                        output_str[:2000] + "...[truncated]"
                    )
                parts.append(f"OUTPUT: {output_str}")
            if node.status:
                parts.append(f"STATUS: {node.status}")
            if node.error_message:
                parts.append(f"ERROR: {node.error_message}")
            return "\n\n".join(parts) if parts else None

        return None

    async def _embed(
        self, text: str
    ) -> Optional[list[float]]:
        try:
            loop = asyncio.get_event_loop()
            vector = await loop.run_in_executor(
                None,
                self.embeddings.embed_query,
                text
            )
            return vector
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise