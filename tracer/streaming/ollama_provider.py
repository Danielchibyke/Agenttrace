import logging
from typing import AsyncGenerator, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from tracer.streaming.provider_base import BaseStreamingProvider
from tracer.streaming.micro_node import MicroNode
from tracer.streaming.token_buffer import TokenBuffer

logger = logging.getLogger(__name__)


class OllamaStreamingProvider(BaseStreamingProvider):
    """
    Ollama streaming adapter.
    Captures each token as it streams from the local model.
    """

    def __init__(
        self,
        token_buffer: TokenBuffer,
        parent_node_id: str,
        session_id: str,
        task_id: str,
        model: str = "qwen2.5:0.5b",
        temperature: float = 0,
    ):
        super().__init__(
            token_buffer, parent_node_id,
            session_id, task_id
        )
        self.model = model
        self.temperature = temperature
        self._full_response = ""
        self._client = ChatOllama(
            model=model,
            temperature=temperature,
        )

    async def stream(
        self,
        messages: list[dict],
        **kwargs
    ) -> AsyncGenerator[MicroNode, None]:
        """
        Stream tokens from Ollama.
        Each token becomes a MicroNode pushed to buffer.
        """
        self._full_response = ""
        self._token_index = 0

        # convert messages to langchain format
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        try:
            async for chunk in self._client.astream(lc_messages):
                token_text = chunk.content
                if not token_text:
                    continue

                self._full_response += token_text

                micro_node = self._create_micro_node(
                    content=token_text,
                    logprob=None,
                )

                # push to buffer — triggers Track A and Track B
                self.buffer.push(micro_node)

                yield micro_node

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise

    async def get_full_response(self) -> str:
        return self._full_response