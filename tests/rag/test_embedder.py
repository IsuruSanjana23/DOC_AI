from __future__ import annotations

import logging
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag.chunker import TextChunk
from app.rag.embedder import (
    BaseEmbedder,
    EmbeddedChunk,
    EmbeddingError,
    SentenceTransformerEmbedder,
)


class FakeEmbedder(BaseEmbedder):
    """Returns deterministic vectors for testing."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk.from_chunk(
                chunk,
                np.full(self.dim, float(chunk.chunk_index + 1)),
            )
            for chunk in chunks
        ]


class TestEmbeddedChunk:

    def test_from_chunk_copies_fields(self) -> None:
        chunk = TextChunk(chunk_index=5, text="hello world", page_number=2)
        vector = np.array([0.1, 0.2, 0.3])

        result = EmbeddedChunk.from_chunk(chunk, vector)

        assert result.chunk_index == 5
        assert result.text == "hello world"
        assert result.page_number == 2
        np.testing.assert_array_equal(result.vector, [0.1, 0.2, 0.3])

    def test_from_chunk_does_not_mutate_original(self) -> None:
        chunk = TextChunk(chunk_index=0, text="test", page_number=None)
        vector = np.array([0.5, 0.6])

        result = EmbeddedChunk.from_chunk(chunk, vector)

        result.chunk_index = 99
        assert chunk.chunk_index == 0

    def test_from_chunk_accepts_none_page(self) -> None:
        chunk = TextChunk(chunk_index=1, text="no page", page_number=None)
        result = EmbeddedChunk.from_chunk(chunk, np.array([0.0]))

        assert result.page_number is None


class TestBaseEmbedder:

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbedder()  # type: ignore[abstract]


class TestFakeEmbedder:

    def test_embed_returns_correct_count(self) -> None:
        chunks = [
            TextChunk(0, "a", 1),
            TextChunk(1, "b", 1),
            TextChunk(2, "c", 2),
        ]
        embedder = FakeEmbedder(dim=4)

        results = embedder.embed(chunks)

        assert len(results) == 3

    def test_embed_preserves_order(self) -> None:
        chunks = [
            TextChunk(0, "first", 1),
            TextChunk(1, "second", 1),
        ]
        embedder = FakeEmbedder(dim=2)

        results = embedder.embed(chunks)

        assert results[0].text == "first"
        assert results[1].text == "second"
        assert results[0].chunk_index == 0
        assert results[1].chunk_index == 1

    def test_embed_empty_list(self) -> None:
        embedder = FakeEmbedder(dim=4)
        assert embedder.embed([]) == []

    def test_embed_vector_dimension(self) -> None:
        chunks = [TextChunk(0, "test", 1)]
        embedder = FakeEmbedder(dim=7)

        results = embedder.embed(chunks)

        assert len(results[0].vector) == 7

    def test_embed_interface_matches_base(self) -> None:
        assert isinstance(FakeEmbedder(), BaseEmbedder)

    def test_embed_query_returns_vector(self) -> None:
        embedder = FakeEmbedder(dim=4)
        result = embedder.embed_query("hello")
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_embed_query_delegates_to_embed(self) -> None:
        embedder = FakeEmbedder(dim=3)
        result = embedder.embed_query("test query")
        np.testing.assert_array_equal(result, np.full(3, 1.0))


class TestSentenceTransformerEmbedder:

    def test_init_does_not_load_model(self) -> None:
        embedder = SentenceTransformerEmbedder(
            model_name="fake",
            batch_size=16,
            device="cpu",
        )
        assert embedder._model is None
        assert embedder.model_name == "fake"
        assert embedder.batch_size == 16
        assert embedder.device == "cpu"

    def test_embed_empty_list(self) -> None:
        embedder = SentenceTransformerEmbedder()
        assert embedder.embed([]) == []

    @patch(
        "app.rag.embedder.SentenceTransformer",
        autospec=True,
    )
    def test_embed_calls_encode_with_correct_args(
        self, mock_st: MagicMock
    ) -> None:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        embedder = SentenceTransformerEmbedder(
            model_name="mock-model",
            batch_size=64,
            device="cpu",
        )
        chunks = [
            TextChunk(0, "text a", 1),
            TextChunk(1, "text b", 2),
        ]

        results = embedder.embed(chunks)

        mock_instance.encode.assert_called_once_with(
            ["text a", "text b"],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        assert len(results) == 2
        np.testing.assert_array_equal(results[0].vector, [0.1, 0.2])
        np.testing.assert_array_equal(results[1].vector, [0.3, 0.4])

    @patch(
        "app.rag.embedder.SentenceTransformer",
        autospec=True,
    )
    def test_embed_preserves_metadata(
        self, mock_st: MagicMock
    ) -> None:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([[0.5, 0.6]])

        embedder = SentenceTransformerEmbedder()
        chunks = [TextChunk(7, "metadata test", 3)]

        results = embedder.embed(chunks)

        assert results[0].chunk_index == 7
        assert results[0].text == "metadata test"
        assert results[0].page_number == 3

    @patch(
        "app.rag.embedder.SentenceTransformer",
        autospec=True,
    )
    def test_embed_logs_on_success(
        self, mock_st: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array(
            [[0.1, 0.2, 0.3, 0.4]]
        )

        embedder = SentenceTransformerEmbedder(
            model_name="test-model",
            batch_size=16,
        )
        caplog.set_level(logging.INFO)
        embedder.embed([TextChunk(0, "log test", 1)])

        records = [r for r in caplog.records if r.name == "app.rag.embedder"]
        info_messages = [r.message for r in records if r.levelname == "INFO"]
        assert any("Embedded 1 chunks" in m for m in info_messages)
        assert any("dim=4" in m for m in info_messages)
        assert any("test-model" in m for m in info_messages)
        assert any("batch_size=16" in m for m in info_messages)

    @patch(
        "app.rag.embedder.SentenceTransformer",
        autospec=True,
    )
    def test_embed_raises_on_encode_failure(
        self, mock_st: MagicMock
    ) -> None:
        mock_instance = mock_st.return_value
        mock_instance.encode.side_effect = RuntimeError("GPU OOM")

        embedder = SentenceTransformerEmbedder()
        chunks = [TextChunk(0, "fail", 1)]

        with pytest.raises(EmbeddingError, match="GPU OOM"):
            embedder.embed(chunks)

    @patch(
        "app.rag.embedder.SentenceTransformer",
        autospec=True,
    )
    def test_embed_raises_on_model_load_failure(
        self, mock_st: MagicMock
    ) -> None:
        mock_st.side_effect = FileNotFoundError("model not found")

        embedder = SentenceTransformerEmbedder(
            model_name="nonexistent",
        )

        with pytest.raises(EmbeddingError, match="Failed to load model"):
            embedder.model

    @patch(
        "app.rag.embedder.SentenceTransformer",
        autospec=True,
    )
    def test_model_property_caches_instance(
        self, mock_st: MagicMock
    ) -> None:
        mock_instance = mock_st.return_value

        embedder = SentenceTransformerEmbedder()

        m1 = embedder.model
        m2 = embedder.model

        assert m1 is m2
        mock_st.assert_called_once()

    def test_normalized_vectors_have_unit_magnitude(self) -> None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        vectors = model.encode(
            ["test sentence"],
            normalize_embeddings=True,
        )

        mag = math.sqrt(float((vectors[0] ** 2).sum()))
        assert abs(mag - 1.0) < 1e-6

    @patch(
        "app.rag.embedder.SentenceTransformer",
        autospec=True,
    )
    def test_model_is_lazy(self, mock_st: MagicMock) -> None:
        embedder = SentenceTransformerEmbedder()
        assert embedder._model is None
        _ = embedder.model
        assert embedder._model is not None

    @patch(
        "app.rag.embedder.SentenceTransformer",
        autospec=True,
    )
    def test_embed_query_calls_embed(self, mock_st: MagicMock) -> None:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([[0.5, 0.6]])

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed_query("what is AI")

        np.testing.assert_array_equal(result, [0.5, 0.6])
        mock_instance.encode.assert_called_once()

    @patch(
        "app.rag.embedder.SentenceTransformer",
        autospec=True,
    )
    def test_embed_query_normalized(self, mock_st: MagicMock) -> None:
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = np.array([[0.6, 0.8]])

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed_query("test")

        mag = math.sqrt(float((result ** 2).sum()))
        assert abs(mag - 1.0) < 1e-6
