import logging

from typing import Optional

from .base import BaseClient

try:
    from anthropic import AnthropicBedrock
except ImportError:
    AnthropicBedrock = "anthropic"

try:
    from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
except ImportError:
    Choice = "openai"  # dummy
    ChatCompletionMessage = "openai"


logger = logging.getLogger(__name__)


class BedrockClient(BaseClient):

    ClientClass = AnthropicBedrock

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        max_tokens: Optional[int] = 10000,
    ) -> None:
        super().__init__(model, temperature)

        if isinstance(self.ClientClass, str):
            logger.fatal(
                f"Package `{self.ClientClass}` is required. Install with `pip install anthropic[bedrock]`"
            )
            exit(-1)

        if isinstance(Choice, str):
            logger.fatal(
                "Package `openai` is required due to `Choice` and `ChatCompletionMessage` classes"
            )
            exit(-1)

        self.client = self.ClientClass(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
        )
        self.max_tokens = max_tokens

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):

        messages_mod = []
        for message in messages:
            if message["role"] == "system":
                messages_mod.append(
                    {
                        "role": "user",
                        "content": "This is the system prompt:\n" + message["content"],
                    }
                )
            else:
                messages_mod.append(
                    {"role": message["role"], "content": message["content"]}
                )

        response = self.client.messages.create(
            model=self.model,
            messages=messages_mod,
            temperature=min(
                temperature, 1.0
            ),  # https://docs.anthropic.com/en/api/messages?q=temperature#body-temperature
            max_tokens=self.max_tokens,
            stream=False,
        )

        choice = Choice(
            index=0,
            message=ChatCompletionMessage(
                role="assistant", content=response.content[0].text
            ),
            finish_reason="stop",
        )
        response = [choice]
        return response
