"""Small OpenAI API wrapper used by the hyperparameter optimizer."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class OpenAIClientError(Exception):
    """Raised when an OpenAI request fails."""


class OptimizerUnavailableError(OpenAIClientError):
    """Raised when OpenAI integration is not available in this environment."""


@runtime_checkable
class ChatCompletionClient(Protocol):
    """Protocol implemented by optimizer LLM clients."""

    def complete_json(self, prompt: str) -> str:
        """Return a JSON-oriented completion for the supplied prompt."""


class OpenAIClient:
    """OpenAI chat-completion client for optimizer analysis."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the OpenAI SDK client.

        Args:
            api_key: OpenAI API key loaded from environment configuration.
            model: Chat-completion model name.
            timeout_seconds: Request timeout in seconds.

        Raises:
            OptimizerUnavailableError: If the OpenAI SDK is not installed.
        """

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OptimizerUnavailableError(
                "The openai package is required for live optimizer analysis."
            ) from exc

        if not api_key.strip():
            raise OptimizerUnavailableError("OpenAI API key is not configured.")

        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self._model = model

    def complete_json(self, prompt: str) -> str:
        """Generate a JSON response for a hyperparameter optimization prompt."""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful quantitative ML engineer. "
                            "Return only valid JSON matching the requested schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("OpenAI optimizer request failed: %s", type(exc).__name__)
            raise OpenAIClientError("OpenAI optimizer request failed.") from exc

        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise OpenAIClientError("OpenAI returned an empty optimizer response.")
        return content


__all__ = [
    "ChatCompletionClient",
    "OpenAIClient",
    "OpenAIClientError",
    "OptimizerUnavailableError",
]
