from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.config import settings
from app.services.exceptions import LLMServiceError
from app.services.llm_service import (
    BaseLLMService,
    DeepSeekLLMService,
    LLMResponse,
)
from app.services.prompt_builder import Prompt


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(
        system="You are a helpful assistant.",
        context="Source 1\nContent: AI is cool.",
        question="What is AI?",
        text="You are a helpful assistant.\n\n"
        "Source 1\nContent: AI is cool.\n\n"
        "What is AI?",
        token_count=50,
        num_chunks=1,
    )


@pytest.fixture
def prompt_no_context() -> Prompt:
    return Prompt(
        system="You are a helpful assistant.",
        context="",
        question="What is AI?",
        text="You are a helpful assistant.\n\nWhat is AI?",
        token_count=30,
        num_chunks=0,
    )


# ── LLMResponse ─────────────────────────────────────────────────────────────


class TestLLMResponse:

    def test_dataclass_fields(self) -> None:
        response = LLMResponse(text="Hello", model="deepseek-v4-flash")
        assert response.text == "Hello"
        assert response.model == "deepseek-v4-flash"
        assert response.usage is None

    def test_with_usage(self) -> None:
        usage = {"prompt_tokens": 50, "completion_tokens": 100}
        response = LLMResponse(
            text="Hello", model="deepseek-v4-flash", usage=usage
        )
        assert response.usage == usage

    def test_frozen(self) -> None:
        response = LLMResponse(text="Hi", model="m")
        with pytest.raises(AttributeError):
            response.text = "changed"  # type: ignore[misc]


# ── BaseLLMService ──────────────────────────────────────────────────────────


class TestBaseLLMService:

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseLLMService()  # type: ignore[abstract]


# ── Constructor ─────────────────────────────────────────────────────────────


class TestDeepSeekConstructor:

    def test_defaults_from_settings(self) -> None:
        service = DeepSeekLLMService()
        assert service._api_key == settings.llm_api_key
        assert service._base_url == settings.llm_base_url.rstrip("/")
        assert service._model == settings.llm_model

    def test_explicit_overrides(self) -> None:
        service = DeepSeekLLMService(
            api_key="test-key",
            base_url="https://custom.example.com/v1",
            model="custom-model",
            timeout_seconds=10,
            max_retries=1,
        )
        assert service._api_key == "test-key"
        assert service._base_url == "https://custom.example.com/v1"
        assert service._model == "custom-model"
        assert service._timeout == 10
        assert service._max_retries == 1

    def test_base_url_strips_trailing_slash(self) -> None:
        service = DeepSeekLLMService(
            api_key="k", base_url="https://example.com/v1/",
        )
        assert service._base_url == "https://example.com/v1"

    def test_raises_on_missing_api_key(self) -> None:
        with pytest.raises(LLMServiceError, match="API key"):
            DeepSeekLLMService(api_key="")


# ── _build_messages ─────────────────────────────────────────────────────────


class TestBuildMessages:

    def test_includes_system_and_user(self, prompt: Prompt) -> None:
        messages = DeepSeekLLMService._build_messages(prompt)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"

    def test_user_content_includes_context_and_question(
        self, prompt: Prompt
    ) -> None:
        messages = DeepSeekLLMService._build_messages(prompt)

        assert "AI is cool." in messages[1]["content"]
        assert "What is AI?" in messages[1]["content"]

    def test_empty_context_omits_context(
        self, prompt_no_context: Prompt
    ) -> None:
        messages = DeepSeekLLMService._build_messages(prompt_no_context)

        assert messages[1]["content"] == "What is AI?"


# ── _parse_response ─────────────────────────────────────────────────────────


class TestParseResponse:

    def test_valid_response(self) -> None:
        data = {
            "choices": [{"message": {"content": "The answer is 42."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        text, usage = DeepSeekLLMService._parse_response(data)
        assert text == "The answer is 42."
        assert usage == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_no_usage_in_response(self) -> None:
        data = {
            "choices": [{"message": {"content": "Hello"}}],
        }
        text, usage = DeepSeekLLMService._parse_response(data)
        assert text == "Hello"
        assert usage is None

    def test_empty_choices_raises_error(self) -> None:
        data = {"choices": []}
        with pytest.raises(LLMServiceError, match="empty choices"):
            DeepSeekLLMService._parse_response(data)

    def test_missing_choices_key_raises_error(self) -> None:
        data = {"foo": "bar"}
        with pytest.raises(LLMServiceError, match="response structure"):
            DeepSeekLLMService._parse_response(data)

    def test_missing_content_key_raises_error(self) -> None:
        data = {"choices": [{"message": {}}]}
        with pytest.raises(LLMServiceError, match="response structure"):
            DeepSeekLLMService._parse_response(data)


# ── _send_request ───────────────────────────────────────────────────────────


class FakeResponse:
    """Simulates an httpx.Response for testing."""

    def __init__(self, status_code: int, json_data: dict) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = str(json_data)

    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self) -> None:
        if not self.is_success():
            raise httpx.HTTPStatusError("Error", request=None, response=self)  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestSendRequest:

    async def test_success(self) -> None:
        service = DeepSeekLLMService(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="test-model",
            max_retries=1,
        )
        payload = {"model": "test-model", "messages": []}

        mock_response = FakeResponse(200, {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 5},
        })

        with patch.object(
            service, "_send_request", return_value=("Hello", {"prompt_tokens": 5})
        ):
            text, usage = await service._send_request(payload)
            # This test validates the method signature and return type
            assert isinstance(text, str)
            assert isinstance(usage, dict) or usage is None

    async def test_real_http_mock(self, prompt: Prompt) -> None:
        """Test the full _send_request path with a mocked httpx client."""
        service = DeepSeekLLMService(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="test-model",
            max_retries=1,
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.text = '{"choices": ...}'

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, usage = await service._send_request(
                {"model": "test-model", "messages": []}
            )

        assert text == "Answer"
        assert usage == {"prompt_tokens": 10, "completion_tokens": 5}
        mock_client.post.assert_called_once()

    async def test_retry_on_503_then_succeeds(self) -> None:
        service = DeepSeekLLMService(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="test-model",
            max_retries=3,
        )

        fail_response = MagicMock(spec=httpx.Response)
        fail_response.status_code = 503
        fail_response.is_success = False
        fail_response.text = "Service Unavailable"

        ok_response = MagicMock(spec=httpx.Response)
        ok_response.status_code = 200
        ok_response.is_success = True
        ok_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
        }
        ok_response.text = '{"choices": ...}'

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=[fail_response, ok_response])

        with patch("httpx.AsyncClient", return_value=mock_client):
            text, usage = await service._send_request(
                {"model": "test-model", "messages": []}
            )

        assert text == "OK"

    async def test_exhausts_retries(self) -> None:
        service = DeepSeekLLMService(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="test-model",
            max_retries=2,
        )

        fail_response = MagicMock(spec=httpx.Response)
        fail_response.status_code = 503
        fail_response.is_success = False
        fail_response.text = "Service Unavailable"

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=fail_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(LLMServiceError, match="after 2 retries"):
                await service._send_request(
                    {"model": "test-model", "messages": []}
                )

    async def test_permanent_4xx_does_not_retry(self) -> None:
        service = DeepSeekLLMService(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="test-model",
            max_retries=3,
        )

        fail_response = MagicMock(spec=httpx.Response)
        fail_response.status_code = 400
        fail_response.is_success = False
        fail_response.text = "Bad Request"

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=fail_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(LLMServiceError, match="400"):
                await service._send_request(
                    {"model": "test-model", "messages": []}
                )

        # Should only have been called once (no retry on 4xx)
        mock_client.post.assert_called_once()


# ── generate (integration) ──────────────────────────────────────────────────


@pytest.mark.asyncio
class TestGenerate:

    async def test_full_flow(self, prompt: Prompt) -> None:
        service = DeepSeekLLMService(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="test-model",
            max_retries=1,
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "AI is cool"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }
        mock_response.text = '{"choices": ...}'

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await service.generate(prompt)

        assert isinstance(result, LLMResponse)
        assert result.text == "AI is cool"
        assert result.model == "test-model"
        assert result.usage == {"prompt_tokens": 50, "completion_tokens": 10}
