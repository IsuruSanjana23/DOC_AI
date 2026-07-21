from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.repositories.chunk_repository import SearchResult
from app.services.exceptions import PromptBuilderError
from app.services.prompt_builder import (
    DEFAULT_CHUNK_TEMPLATE,
    DEFAULT_CONTEXT_TEMPLATE,
    DEFAULT_QUESTION_TEMPLATE,
    DEFAULT_SYSTEM_TEMPLATE,
    Prompt,
    PromptBuilder,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def make_chunk(
    text: str = "Sample chunk content.",
    page_number: int | None = 5,
    document_title: str | None = "test.pdf",
) -> MagicMock:
    chunk = MagicMock(spec=DocumentChunk)
    chunk.text = text
    chunk.page_number = page_number
    chunk.document_id = uuid4()

    if document_title is not None:
        doc = MagicMock(spec=Document)
        doc.original_filename = document_title
        chunk.document = doc
    else:
        chunk.document = None

    return chunk


def make_result(
    text: str = "Sample chunk content.",
    page_number: int | None = 5,
    document_title: str | None = "test.pdf",
    score: float = 0.95,
) -> SearchResult:
    chunk = make_chunk(
        text=text,
        page_number=page_number,
        document_title=document_title,
    )
    return SearchResult(chunk=chunk, score=score)  # type: ignore[arg-type]


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


# ── Constructor ─────────────────────────────────────────────────────────────


class TestPromptBuilderConstructor:

    def test_default_templates(self) -> None:
        builder = PromptBuilder()
        assert builder._system_template == DEFAULT_SYSTEM_TEMPLATE
        assert builder._context_template == DEFAULT_CONTEXT_TEMPLATE
        assert builder._chunk_template == DEFAULT_CHUNK_TEMPLATE
        assert builder._question_template == DEFAULT_QUESTION_TEMPLATE

    def test_custom_templates(self) -> None:
        custom_system = "Custom system"
        custom_context = "Custom context: {context_blocks}"
        custom_chunk = "Chunk {source_number}: {text}"
        custom_question = "Q: {query}"

        builder = PromptBuilder(
            system_template=custom_system,
            context_template=custom_context,
            chunk_template=custom_chunk,
            question_template=custom_question,
        )
        assert builder._system_template == custom_system
        assert builder._context_template == custom_context
        assert builder._chunk_template == custom_chunk
        assert builder._question_template == custom_question


# ── Happy path ──────────────────────────────────────────────────────────────


class TestBuild:

    def test_returns_prompt_dataclass(self, builder: PromptBuilder) -> None:
        result = make_result()
        prompt = builder.build("What is DI?", [result])

        assert isinstance(prompt, Prompt)

    def test_contains_system_instructions(self, builder: PromptBuilder) -> None:
        result = make_result()
        prompt = builder.build("What is DI?", [result])

        assert "You are a precise, factual AI assistant" in prompt.text
        assert prompt.system == DEFAULT_SYSTEM_TEMPLATE.strip()

    def test_contains_formatted_context(self, builder: PromptBuilder) -> None:
        result = make_result(
            text="Dependency injection is a design pattern.",
            page_number=42,
            document_title="FastAPI Guide.pdf",
        )
        prompt = builder.build("What is DI?", [result])

        assert "Source 1" in prompt.context
        assert "FastAPI Guide.pdf" in prompt.context
        assert "Page: 42" in prompt.context
        assert "Dependency injection is a design pattern." in prompt.context

    def test_contains_formatted_question(self, builder: PromptBuilder) -> None:
        result = make_result()
        prompt = builder.build("What is DI?", [result])

        assert "What is DI?" in prompt.question
        assert "Answer:" in prompt.question

    def test_multiple_chunks_numbered_sequentially(
        self, builder: PromptBuilder
    ) -> None:
        results = [
            make_result(text="Chunk A", document_title="Doc 1"),
            make_result(text="Chunk B", document_title="Doc 2"),
            make_result(text="Chunk C", document_title="Doc 3"),
        ]
        prompt = builder.build("test", results)

        assert "Source 1" in prompt.context
        assert "Source 2" in prompt.context
        assert "Source 3" in prompt.context

    def test_chunks_separated_by_blank_line(
        self, builder: PromptBuilder
    ) -> None:
        results = [
            make_result(text="First chunk"),
            make_result(text="Second chunk"),
        ]
        prompt = builder.build("test", results)

        assert "First chunk\n\n--------------\nSource 2" in prompt.context

    def test_num_chunks_populated(self, builder: PromptBuilder) -> None:
        results = [make_result() for _ in range(4)]
        prompt = builder.build("test", results)

        assert prompt.num_chunks == 4

    def test_token_count_populated(self, builder: PromptBuilder) -> None:
        result = make_result(text="Short text.")
        prompt = builder.build("Hi?", [result])

        assert prompt.token_count > 0
        assert prompt.token_count == len(prompt.text)

    def test_assembled_text_contains_all_sections(
        self, builder: PromptBuilder
    ) -> None:
        result = make_result(text="Some context.")
        prompt = builder.build("A question?", [result])

        assert "You are a precise" in prompt.text
        assert "Some context." in prompt.text
        assert "A question?" in prompt.text

    def test_question_appears_after_context(
        self, builder: PromptBuilder
    ) -> None:
        result = make_result(text="CONTEXT_BLOCK")
        prompt = builder.build("QUERY_TEXT", [result])

        context_pos = prompt.text.index("CONTEXT_BLOCK")
        question_pos = prompt.text.index("QUERY_TEXT")
        assert context_pos < question_pos


# ── Input validation ────────────────────────────────────────────────────────


class TestInputValidation:

    def test_empty_query_raises_error(self, builder: PromptBuilder) -> None:
        with pytest.raises(PromptBuilderError, match="non-empty"):
            builder.build("", [make_result()])

    def test_whitespace_query_raises_error(self, builder: PromptBuilder) -> None:
        with pytest.raises(PromptBuilderError, match="non-empty"):
            builder.build("   ", [make_result()])

    def test_query_is_stripped(self, builder: PromptBuilder) -> None:
        result = make_result()
        prompt = builder.build("  What is DI?  ", [result])

        assert "What is DI?" in prompt.question
        assert "  What is DI?  " not in prompt.question


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_results_produces_no_context(
        self, builder: PromptBuilder
    ) -> None:
        prompt = builder.build("Hello?", [])

        assert prompt.context == ""
        assert "Context:" not in prompt.text

    def test_missing_document_title_uses_fallback(
        self, builder: PromptBuilder
    ) -> None:
        chunk = make_chunk(document_title=None)
        result = SearchResult(chunk=chunk, score=0.9)  # type: ignore[arg-type]
        prompt = builder.build("test", [result])

        assert "Unknown Document" in prompt.context

    def test_none_page_number_uses_fallback(
        self, builder: PromptBuilder
    ) -> None:
        result = make_result(page_number=None)
        prompt = builder.build("test", [result])

        assert "Page: N/A" in prompt.context

    def test_empty_chunk_text_raises_error(
        self, builder: PromptBuilder
    ) -> None:
        result = make_result(text="")
        with pytest.raises(PromptBuilderError, match="empty text"):
            builder.build("test", [result])


# ── Template overrides ──────────────────────────────────────────────────────


class TestTemplateOverrides:

    def test_custom_system_template(self) -> None:
        builder = PromptBuilder(
            system_template="You are a {role}.",
        )
        prompt = builder.build("Hi?", [make_result()])

        assert "You are a {role}." == prompt.system
        assert "precise" not in prompt.system

    def test_custom_chunk_template(self) -> None:
        builder = PromptBuilder(
            chunk_template="[{source_number}] {text}",
        )
        result = make_result(text="Hello world")
        prompt = builder.build("Hi?", [result])

        assert "[1] Hello world" in prompt.context

    def test_custom_question_template(self) -> None:
        builder = PromptBuilder(
            question_template="{query}???",
        )
        prompt = builder.build("What", [make_result()])

        assert "What???" in prompt.question


# ── PromptBuilderError ──────────────────────────────────────────────────────


class TestPromptBuilderError:

    def test_is_exception(self) -> None:
        assert issubclass(PromptBuilderError, Exception)

    def test_carries_message(self) -> None:
        error = PromptBuilderError("Something went wrong")
        assert str(error) == "Something went wrong"
