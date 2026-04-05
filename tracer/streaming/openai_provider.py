import logging
import os
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI
from tracer.streaming.provider_base import BaseStreamingProvider
from tracer.streaming.micro_node import MicroNode
from tracer.streaming.token_buffer import TokenBuffer

logger = logging.getLogger(__name__)


class OpenAIStreamingProvider(BaseStreamingProvider):
    """
    OpenAI streaming adapter.
    Captures each token including logprobs if available.
    """

    def __init__(
        self,
        token_buffer: TokenBuffer,
        parent_node_id: str,
        session_id: str,
        task_id: str,
        model: str = "gpt-4o",
        temperature: float = 0,
        capture_logprobs: bool = True,
    ):
        super().__init__(
            token_buffer, parent_node_id,
            session_id, task_id
        )
        self.model = model
        self.temperature = temperature
        self.capture_logprobs = capture_logprobs
        self._full_response = ""
        self._client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def stream(
        self,
        messages: list[dict],
        **kwargs
    ) -> AsyncGenerator[MicroNode, None]:
        """
        Stream tokens from OpenAI.
        Captures logprobs when available.
        """
        self._full_response = ""
        self._token_index = 0

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=self.temperature,
                logprobs=self.capture_logprobs,
                top_logprobs=1 if self.capture_logprobs else None,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                token_text = delta.content
                if not token_text:
                    continue

                self._full_response += token_text

                # extract logprob if available
                logprob = None
                if (self.capture_logprobs and
                        chunk.choices[0].logprobs and
                        chunk.choices[0].logprobs.content):
                    logprob = (
                        chunk.choices[0]
                        .logprobs.content[0].logprob
                    )

                micro_node = self._create_micro_node(
                    content=token_text,
                    logprob=logprob,
                )

                self.buffer.push(micro_node)
                yield micro_node

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    async def get_full_response(self) -> str:
        return self._full_response