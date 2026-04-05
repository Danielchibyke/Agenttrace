from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from tracer.streaming.micro_node import MicroNode
from tracer.streaming.token_buffer import TokenBuffer


class BaseStreamingProvider(ABC):
    """
    Every LLM provider adapter must implement this interface.
    Translates provider-specific streaming events into
    universal MicroNode objects and pushes them to TokenBuffer.
    """

    def __init__(
        self,
        token_buffer: TokenBuffer,
        parent_node_id: str,
        session_id: str,
        task_id: str,
    ):
        self.buffer = token_buffer
        self.parent_node_id = parent_node_id
        self.session_id = session_id
        self.task_id = task_id
        self._token_index = 0

    def _create_micro_node(
        self,
        content: str,
        logprob: Optional[float] = None,
    ) -> MicroNode:
        node = MicroNode(
            parent_node_id=self.parent_node_id,
            session_id=self.session_id,
            task_id=self.task_id,
            content=content,
            index=self._token_index,
            token_index=self._token_index,
            logprob=logprob,
        )
        self._token_index += 1
        return node

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        **kwargs
    ) -> AsyncGenerator[MicroNode, None]:
        """
        Stream tokens from the LLM provider.
        Must yield MicroNode for each token received.
        Must also push each MicroNode to self.buffer.
        """
        pass

    @abstractmethod
    async def get_full_response(self) -> str:
        """
        Return the complete assembled response text
        after streaming is complete.
        """
        pass