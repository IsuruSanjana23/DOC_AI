from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.retriever import RetrieverError
from app.repositories.chunk_repository import SearchResult
from app.services.exceptions import LLMServiceError, PromptBuilderError
from app.services.rag_service import RAGResponse, RAGService, RAGServiceError


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_search_result(
    text: str = "Sample chunk text for testing purposes.",
    score: float = 0.95,
    page_number: int | None = 1,
    document_title: str = "test.pdf",
) -> SearchResult:
    chunk = MagicMock()
    chunk.text = text
    chunk.page_number = page_number
    doc = MagicMock()
    doc.original_filename = document_title
    chunk.document = doc
    return SearchResult(chunk=chunk, score=score)


def _make_results(n: int = 3) -> list[SearchResult]:
    return [
        _make_search_result(
            text=f"Chunk {i} text content for testing similarity search.",
            score=0.95 - i * 0.05,
            page_number=i + 1,
            document_title="test.pdf",
        )
        for i in range(n)
    ]


def _make_mocks(results: list[SearchResult] | None = None):
    if results is None:
        results = _make_results()

    retriever = MagicMock()
    retriever.retrieve.return_value = results

    prompt_builder = MagicMock()
    prompt = MagicMock()
    prompt.text = "Full assembled prompt text"
    prompt.token_count = 600
    prompt.num_chunks = len(results)
    prompt.system = "System instructions"
    prompt.context = "Context block"
    prompt.question = "What is the question?"
    prompt_builder.build.return_value = prompt

    llm_service = AsyncMock()
    response = MagicMock()
    response.text = "This is the generated answer from the LLM."
    response.model = "deepseek-v4-flash"
    response.usage = {
        "prompt_tokens": 600,
        "completion_tokens": 120,
        "total_tokens": 720,
    }
    llm_service.generate.return_value = response

    return retriever, prompt_builder, llm_service, prompt


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRAGResponse:

    def test_dataclass_fields(self):
        prompt = MagicMock()
        response = RAGResponse(
            query="test query",
            answer="test answer",
            sources=[],
            token_usage=None,
            prompt=prompt,
        )
        assert response.query == "test query"
        assert response.answer == "test answer"
        assert response.sources == []
        assert response.token_usage is None
        assert response.prompt is prompt

    def test_frozen(self):
        prompt = MagicMock()
        response = RAGResponse(
            query="q", answer="a", sources=[], token_usage=None, prompt=prompt
        )
        with pytest.raises(AttributeError):
            response.answer = "new answer"


class TestRAGServiceConstructor:

    def test_stores_dependencies(self):
        retriever = MagicMock()
        builder = MagicMock()
        llm = MagicMock()
        service = RAGService(retriever, builder, llm)
        assert service._retriever is retriever
        assert service._prompt_builder is builder
        assert service._llm_service is llm


class TestAnswer:

    @pytest.mark.asyncio
    async def test_returns_rag_response(self):
        retriever, builder, llm, prompt = _make_mocks()
        service = RAGService(retriever, builder, llm)

        result = await service.answer("test query")

        assert isinstance(result, RAGResponse)
        assert result.query == "test query"

    @pytest.mark.asyncio
    async def test_passes_query_to_retriever(self):
        retriever, builder, llm, prompt = _make_mocks()
        service = RAGService(retriever, builder, llm)

        await service.answer("What is DI?", top_k=7, min_score=0.3)

        retriever.retrieve.assert_called_once_with(
            "What is DI?", top_k=7
        )

    @pytest.mark.asyncio
    async def test_passes_results_to_prompt_builder(self):
        results = _make_results(2)
        retriever, builder, llm, prompt = _make_mocks(results)
        service = RAGService(retriever, builder, llm)

        await service.answer("test query")

        builder.build.assert_called_once_with("test query", results)

    @pytest.mark.asyncio
    async def test_passes_prompt_to_llm(self):
        retriever, builder, llm, prompt = _make_mocks()
        service = RAGService(retriever, builder, llm)

        await service.answer("test query")

        llm.generate.assert_called_once_with(prompt)

    @pytest.mark.asyncio
    async def test_includes_answer_text(self):
        retriever, builder, llm, prompt = _make_mocks()
        service = RAGService(retriever, builder, llm)

        result = await service.answer("test query")

        assert result.answer == "This is the generated answer from the LLM."

    @pytest.mark.asyncio
    async def test_includes_sources_with_metadata(self):
        results = _make_results(2)
        retriever, builder, llm, prompt = _make_mocks(results)
        service = RAGService(retriever, builder, llm)

        result = await service.answer("test query")

        assert len(result.sources) == 2
        assert result.sources[0]["source_number"] == 1
        assert result.sources[0]["document_title"] == "test.pdf"
        assert result.sources[0]["page_number"] == 1
        assert "Chunk 0" in result.sources[0]["text_preview"]
        assert result.sources[0]["relevance_score"] == 0.95
        assert result.sources[1]["source_number"] == 2
        assert result.sources[1]["relevance_score"] == 0.90

    @pytest.mark.asyncio
    async def test_includes_token_usage(self):
        retriever, builder, llm, prompt = _make_mocks()
        service = RAGService(retriever, builder, llm)

        result = await service.answer("test query")

        assert result.token_usage == {
            "prompt_tokens": 600,
            "completion_tokens": 120,
            "total_tokens": 720,
        }

    @pytest.mark.asyncio
    async def test_token_usage_none_when_llm_returns_none(self):
        retriever, builder, llm, prompt = _make_mocks()
        llm.generate.return_value.usage = None
        service = RAGService(retriever, builder, llm)

        result = await service.answer("test query")

        assert result.token_usage is None

    @pytest.mark.asyncio
    async def test_includes_prompt(self):
        retriever, builder, llm, prompt = _make_mocks()
        service = RAGService(retriever, builder, llm)

        result = await service.answer("test query")

        assert result.prompt is prompt

    @pytest.mark.asyncio
    async def test_retrieval_failure_raises_rag_service_error(self):
        retriever, builder, llm, prompt = _make_mocks()
        retriever.retrieve.side_effect = RetrieverError("DB connection lost")
        service = RAGService(retriever, builder, llm)

        with pytest.raises(RAGServiceError, match="Retrieval failed"):
            await service.answer("test query")

    @pytest.mark.asyncio
    async def test_prompt_builder_failure_raises_rag_service_error(self):
        retriever, builder, llm, prompt = _make_mocks()
        builder.build.side_effect = PromptBuilderError("Empty query")
        service = RAGService(retriever, builder, llm)

        with pytest.raises(RAGServiceError, match="Prompt building failed"):
            await service.answer("test query")

    @pytest.mark.asyncio
    async def test_llm_failure_raises_rag_service_error(self):
        retriever, builder, llm, prompt = _make_mocks()
        llm.generate.side_effect = LLMServiceError("API timeout")
        service = RAGService(retriever, builder, llm)

        with pytest.raises(RAGServiceError, match="LLM generation failed"):
            await service.answer("test query")

    @pytest.mark.asyncio
    async def test_uses_default_top_k_when_not_provided(self):
        retriever, builder, llm, prompt = _make_mocks()
        service = RAGService(retriever, builder, llm)

        await service.answer("test query")

        retriever.retrieve.assert_called_once_with("test query", top_k=5)

    @pytest.mark.asyncio
    async def test_source_text_preview_limited_to_200_chars(self):
        long_text = "A" * 500
        results = [_make_search_result(text=long_text, score=0.9)]
        retriever, builder, llm, prompt = _make_mocks(results)
        service = RAGService(retriever, builder, llm)

        result = await service.answer("test query")

        assert len(result.sources[0]["text_preview"]) == 200
        assert result.sources[0]["text_preview"] == "A" * 200

    @pytest.mark.asyncio
    async def test_relevance_scores_rounded_to_4_decimals(self):
        results = [_make_search_result(score=0.123456)]
        retriever, builder, llm, prompt = _make_mocks(results)
        service = RAGService(retriever, builder, llm)

        result = await service.answer("test query")

        assert result.sources[0]["relevance_score"] == 0.1235


class TestRAGServiceError:

    def test_is_exception(self):
        assert issubclass(RAGServiceError, Exception)

    def test_carries_message(self):
        err = RAGServiceError("something broke")
        assert str(err) == "something broke"
