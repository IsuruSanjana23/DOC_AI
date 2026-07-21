from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from app.models.chunk import DocumentChunk
from app.rag.chunker import TextChunk
from app.rag.embedder import BaseEmbedder, EmbeddedChunk, EmbeddingError
from app.rag.retriever import Retriever, RetrieverError
from app.repositories.chunk_repository import ChunkRepository, SearchResult


class FakeEmbedder(BaseEmbedder):

    def __init__(self) -> None:
        self.last_texts: list[str] = []

    def embed(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        self.last_texts = [c.text for c in chunks]
        return [
            EmbeddedChunk.from_chunk(c, np.array([0.1, 0.2, 0.3]))
            for c in chunks
        ]


@pytest.fixture
def repository() -> MagicMock:
    return MagicMock(spec=ChunkRepository)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_chunk(**kwargs: object) -> MagicMock:
    chunk = MagicMock(spec=DocumentChunk)
    chunk.document_id = kwargs.get("document_id", uuid4())
    chunk.chunk_index = kwargs.get("chunk_index", 0)
    chunk.text = kwargs.get("text", "")
    chunk.embedding = kwargs.get("embedding", [0.1, 0.2, 0.3])
    return chunk


class TestRetrieverDependencies:

    def test_requires_repository(self) -> None:
        embedder = FakeEmbedder()
        with pytest.raises(TypeError):
            Retriever(embedder=embedder)  # type: ignore[call-arg]

    def test_requires_embedder(self) -> None:
        repo = MagicMock(spec=ChunkRepository)
        with pytest.raises(TypeError):
            Retriever(repository=repo)  # type: ignore[call-arg]

    def test_injects_repository(self) -> None:
        repo = MagicMock(spec=ChunkRepository)
        embedder = FakeEmbedder()
        retriever = Retriever(repository=repo, embedder=embedder)
        assert retriever._repository is repo
        assert retriever._embedder is embedder


class TestRetriever:

    def test_retrieve_uses_embedder_to_encode_query(
        self,
        repository: MagicMock,
        embedder: FakeEmbedder,
    ) -> None:
        fake_chunk = make_chunk(text="AI is cool")
        repository.search_similar.return_value = [
            SearchResult(chunk=fake_chunk, score=0.9),
        ]

        retriever = Retriever(repository=repository, embedder=embedder)
        results = retriever.retrieve("what is AI")

        assert embedder.last_texts == ["what is AI"]
        assert len(results) == 1
        assert results[0].chunk is fake_chunk

    def test_retrieve_returns_search_results(
        self,
        repository: MagicMock,
        embedder: FakeEmbedder,
    ) -> None:
        repository.search_similar.return_value = [
            SearchResult(chunk=make_chunk(text="a"), score=0.95),
            SearchResult(chunk=make_chunk(text="b"), score=0.85),
        ]

        retriever = Retriever(repository=repository, embedder=embedder)
        results = retriever.retrieve("test")

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score == 0.95
        assert results[1].score == 0.85

    def test_retrieve_passes_top_k_to_repository(
        self,
        repository: MagicMock,
        embedder: FakeEmbedder,
    ) -> None:
        repository.search_similar.return_value = []

        retriever = Retriever(
            repository=repository,
            embedder=embedder,
            top_k=10,
        )
        retriever.retrieve("test")

        repository.search_similar.assert_called_once()
        _args, kwargs = repository.search_similar.call_args
        assert kwargs.get("top_k") == 10

    def test_retrieve_overrides_default_top_k(
        self,
        repository: MagicMock,
        embedder: FakeEmbedder,
    ) -> None:
        repository.search_similar.return_value = []

        retriever = Retriever(
            repository=repository,
            embedder=embedder,
            top_k=3,
        )
        retriever.retrieve("test", top_k=7)

        _args, kwargs = repository.search_similar.call_args
        assert kwargs.get("top_k") == 7

    def test_retrieve_forwards_query_vector(
        self,
        repository: MagicMock,
        embedder: FakeEmbedder,
    ) -> None:
        repository.search_similar.return_value = []

        retriever = Retriever(repository=repository, embedder=embedder)
        retriever.retrieve("test query")

        args, _kwargs = repository.search_similar.call_args
        vector = args[0]
        assert isinstance(vector, list)
        assert vector == [0.1, 0.2, 0.3]

    def test_retrieve_empty_query(
        self,
        repository: MagicMock,
        embedder: FakeEmbedder,
    ) -> None:
        repository.search_similar.return_value = []

        retriever = Retriever(repository=repository, embedder=embedder)
        results = retriever.retrieve("")

        assert results == []
        assert embedder.last_texts == [""]


class TestRetrieverErrors:

    def test_retriever_error_is_exception(self) -> None:
        assert issubclass(RetrieverError, Exception)

    def test_wraps_embedding_error(
        self,
        repository: MagicMock,
    ) -> None:
        class FailingEmbedder(BaseEmbedder):
            def embed(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
                raise EmbeddingError("model not loaded")

        embedder = FailingEmbedder()
        retriever = Retriever(repository=repository, embedder=embedder)

        with pytest.raises(RetrieverError) as exc_info:
            retriever.retrieve("test query")

        assert "Failed to embed query" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, EmbeddingError)

    def test_wraps_repository_error(
        self,
        embedder: FakeEmbedder,
    ) -> None:
        repository = MagicMock(spec=ChunkRepository)
        repository.search_similar.side_effect = RuntimeError("connection lost")

        retriever = Retriever(repository=repository, embedder=embedder)

        with pytest.raises(RetrieverError) as exc_info:
            retriever.retrieve("test query")

        assert "Vector search failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_wraps_unexpected_embedding_error(
        self,
        repository: MagicMock,
    ) -> None:
        class ExplodingEmbedder(BaseEmbedder):
            def embed(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
                raise RuntimeError("segfault in tokenizer")

        embedder = ExplodingEmbedder()
        retriever = Retriever(repository=repository, embedder=embedder)

        with pytest.raises(RetrieverError) as exc_info:
            retriever.retrieve("test query")

        assert "Unexpected embedding error" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
