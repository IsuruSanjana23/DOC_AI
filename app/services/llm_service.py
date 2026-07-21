from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings
from app.services.exceptions import LLMServiceError
from app.services.prompt_builder import Prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """The immutable result of an LLM generation call.

    :param text: The generated response text.
    :param model: The model that produced the response.
    :param usage: Optional token usage details from the API
        (e.g., ``{"prompt_tokens": ..., "completion_tokens": ...}``).
    """

    text: str
    model: str
    usage: dict[str, Any] | None = None


class BaseLLMService(ABC):
    """Abstract interface for LLM providers.

    Every concrete LLM service must implement :meth:`generate` which
    accepts a :class:`Prompt` and returns a :class:`LLMResponse`.

    The :class:`Prompt` carries separate ``system`` and user-content
    fields so that providers with dedicated system-message slots
    (OpenAI, Anthropic, DeepSeek via LiteLLM) can format the request
    optimally.
    """

    @abstractmethod
    async def generate(self, prompt: Prompt) -> LLMResponse:
        """Send *prompt* to the LLM and return the generated response.

        :param prompt: The fully constructed prompt, including system
            instructions, context, and question.
        :returns: An immutable :class:`LLMResponse`.

        :raises LLMServiceError: On any API, network, or response-
            parsing failure.
        """
        ...


class DeepSeekLLMService(BaseLLMService):
    """LLM service for DeepSeek (via LiteLLM proxy).

    Communicates with an OpenAI-compatible chat completions endpoint.
    The LiteLLM proxy translates the OpenAI-format request to the
    DeepSeek model.

    Configuration is read from :class:`app.core.config.Settings`:
        ``llm_api_key``, ``llm_base_url``, ``llm_model``,
        ``llm_timeout_seconds``, ``llm_max_retries``.

    Usage::

        service = DeepSeekLLMService()
        response = await service.generate(prompt)
        print(response.text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Initialise the DeepSeek LLM service.

        All parameters default to values from ``app.core.config.settings``.
        Explicit arguments take precedence, enabling per-instance overrides
        (e.g., different model for testing).

        :param api_key: API key for the LiteLLM endpoint.
        :param base_url: Base URL of the OpenAI-compatible endpoint.
        :param model: Model name to use (passed through LiteLLM).
        :param timeout_seconds: Maximum wait time per request.
        :param max_retries: Number of retries on transient failures.
        """
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._base_url = (
            base_url if base_url is not None else settings.llm_base_url
        ).rstrip("/")
        self._model = model if model is not None else settings.llm_model
        self._timeout = (
            timeout_seconds if timeout_seconds is not None
            else settings.llm_timeout_seconds
        )
        self._max_retries = (
            max_retries if max_retries is not None
            else settings.llm_max_retries
        )

        if not self._api_key:
            raise LLMServiceError(
                "LLM API key is not configured. "
                "Set the LITELLM_API_KEY environment variable."
            )

        logger.debug(
            "DeepSeekLLMService initialized — "
            "base_url=%s model=%s timeout=%ds max_retries=%d",
            self._base_url,
            self._model,
            self._timeout,
            self._max_retries,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    async def generate(self, prompt: Prompt) -> LLMResponse:
        """Send the prompt to the DeepSeek LLM and return the response.

        :param prompt: The structured prompt to send.
        :returns: The generated response.

        :raises LLMServiceError: On API errors, network failures,
            timeouts, or malformed responses.
        """
        messages = self._build_messages(prompt)

        logger.info(
            "LLM request — model=%s messages=%d "
            "system_len=%d user_len=%d",
            self._model,
            len(messages),
            len(messages[0]["content"]),
            len(messages[1]["content"]),
        )

        payload = {
            "model": self._model,
            "messages": messages,
        }

        text, usage = await self._send_request(payload)

        logger.info(
            "LLM response — model=%s response_len=%d",
            self._model,
            len(text),
        )

        return LLMResponse(text=text, model=self._model, usage=usage)

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _build_messages(prompt: Prompt) -> list[dict[str, str]]:
        """Convert a :class:`Prompt` into the messages array for the API.

        The system instructions go to the ``system`` role.
        The context (if any) and question go to the ``user`` role.
        """
        user_parts = []
        if prompt.context:
            user_parts.append(prompt.context)
        user_parts.append(prompt.question)

        return [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    async def _send_request(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        """POST the payload to the chat completions endpoint.

        Implements retry logic for transient failures (5xx, rate limits).
        Raises :class:`LLMServiceError` on permanent failures.

        :param payload: The request body (model + messages).
        :returns: A tuple of (response_text, usage_dict_or_none).
        """
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_exception: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self._timeout),
                ) as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload,
                    )

                if response.is_success:
                    return self._parse_response(response.json())

                if response.status_code in (429, 502, 503, 504):
                    logger.warning(
                        "LLM transient error (attempt %d/%d) — "
                        "status=%d",
                        attempt,
                        self._max_retries,
                        response.status_code,
                    )
                    last_exception = LLMServiceError(
                        "API returned status "
                        f"{response.status_code}: {response.text}"
                    )
                    continue

                raise LLMServiceError(
                    f"LLM API error (status={response.status_code}): "
                    f"{response.text}"
                )

            except httpx.TimeoutException as e:
                logger.warning(
                    "LLM timeout (attempt %d/%d)", attempt, self._max_retries
                )
                last_exception = LLMServiceError(
                    f"Request timed out after {self._timeout}s"
                )
                last_exception.__cause__ = e

            except httpx.RequestError as e:
                logger.warning(
                    "LLM network error (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    e,
                )
                last_exception = LLMServiceError(f"Network error: {e}")
                last_exception.__cause__ = e

        raise LLMServiceError(
            f"LLM request failed after {self._max_retries} retries"
        ) from last_exception

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        """Extract the response text and usage from the API JSON response.

        :param data: The parsed JSON response from the API.
        :returns: A tuple of (response_text, usage_or_none).

        :raises LLMServiceError: If the response structure is unexpected.
        """
        try:
            choices = data["choices"]
            if not choices:
                raise LLMServiceError("API returned empty choices array")
            text = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMServiceError(
                f"Unexpected API response structure: {e}"
            ) from e

        usage: dict[str, Any] | None = data.get("usage")
        return text, usage
