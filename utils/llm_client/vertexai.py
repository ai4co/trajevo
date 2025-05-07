import logging
import os

from typing import Optional

from .base import BaseClient

try:
    from langchain_google_vertexai import ChatVertexAI
except ImportError:
    ChatVertexAI = "langchain_google_vertexai"

try:
    from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
except ImportError:
    Choice = "openai"  # dummy
    ChatCompletionMessage = "openai"

logger = logging.getLogger(__name__)


class VertexAIClient(BaseClient):
    ClientClass = ChatVertexAI

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 6,
    ) -> None:
        super().__init__(model, temperature)

        if isinstance(self.ClientClass, str):
            logger.fatal(
                f"Package `{self.ClientClass}` is required. Install with `pip install langchain-google-vertexai`"
            )
            exit(-1)

        if isinstance(Choice, str):
            logger.fatal(
                "Package `openai` is required due to `Choice` and `ChatCompletionMessage` classes"
            )
            exit(-1)

        # Check if GOOGLE_APPLICATION_CREDENTIALS is set
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            logger.fatal(
                "GOOGLE_APPLICATION_CREDENTIALS is not set. Please set it to the path of the service account key file https://python.langchain.com/docs/integrations/llms/google_vertex_ai_palm/."
            )

        self.client = self.ClientClass(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        # Convert messages to the format expected by VertexAI
        messages_mod = []
        for message in messages:
            if message["role"] == "system":
                messages_mod.append(("system", message["content"]))
            else:
                messages_mod.append((message["role"], message["content"]))

        response = self.client.invoke(messages_mod)

        # Convert response to OpenAI format
        choice = Choice(
            index=0,
            message=ChatCompletionMessage(role="assistant", content=response.content),
            finish_reason="stop",
        )
        logger.info("Response has been successfully generated!")

        return [choice]
