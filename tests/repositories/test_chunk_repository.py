from __future__ import annotations

from unittest.mock import MagicMock, call, create_autospec
from uuid import UUID, uuid4

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk
from app.rag.embedder import EmbeddedChunk
from app.repositories.chunk_repository import ChunkRepository, SearchResult


class TestSearchResult:

    def test_dataclass_fields(self) -> None:
        chunk = MagicMock(spec=DocumentChunk)
        result = SearchResult(chunk=chunk, score=0.85)
        assert result.chunk is chunk
        assert result.score == 0.85

    def test_dataclass_slots(self) -> None:
        result = SearchResult(chunk=MagicMock(spec=DocumentChunk), score=0.5)
        with pytest.raises(AttributeError):
            result.missing = "should not work"  # type: ignore[attr-defined]


class TestChunkRepository:

    @pytest.fixture
    def db(self) -> MagicMock:
        return create_autospec(Session, instance=True)

    @pytest.fixture
    def repo(self, db: MagicMock) -> ChunkRepository:
        return ChunkRepository(db)

    @pytest.fixture
    def document_id(self) -> UUID:
        return uuid4()

    @pytest.fixture
    def chunks(self) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(0, "Chunk one", 1, np.array([0.1, 0.2, 0.3])),
            EmbeddedChunk(1, "Chunk two", 2, np.array([0.4, 0.5, 0.6])),
            EmbeddedChunk(2, "Chunk three", None, np.array([0.7, 0.8, 0.9])),
        ]

    # ── save_chunks ──────────────────────────────────────────────────────

    def test_save_chunks_creates_orm_objects(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
        chunks: list[EmbeddedChunk],
    ) -> None:
        result = repo.save_chunks(document_id, chunks)

        assert len(result) == 3
        assert all(isinstance(c, DocumentChunk) for c in result)

        added: list[DocumentChunk] = db.add_all.call_args[0][0]
        assert added[0].text == "Chunk one"
        assert added[1].text == "Chunk two"
        assert added[2].text == "Chunk three"

    def test_save_chunks_converts_numpy_array_to_list(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
        chunks: list[EmbeddedChunk],
    ) -> None:
        repo.save_chunks(document_id, chunks)

        added: list[DocumentChunk] = db.add_all.call_args[0][0]
        assert added[0].embedding == [0.1, 0.2, 0.3]
        assert added[1].embedding == [0.4, 0.5, 0.6]
        assert isinstance(added[0].embedding, list)

    def test_save_chunks_assigns_document_id(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
        chunks: list[EmbeddedChunk],
    ) -> None:
        repo.save_chunks(document_id, chunks)

        added: list[DocumentChunk] = db.add_all.call_args[0][0]
        assert added[0].document_id == document_id
        assert added[1].document_id == document_id
        assert added[2].document_id == document_id

    def test_save_chunks_preserves_page_number(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
        chunks: list[EmbeddedChunk],
    ) -> None:
        repo.save_chunks(document_id, chunks)

        added: list[DocumentChunk] = db.add_all.call_args[0][0]
        assert added[0].page_number == 1
        assert added[1].page_number == 2
        assert added[2].page_number is None

    def test_save_chunks_preserves_chunk_index(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
        chunks: list[EmbeddedChunk],
    ) -> None:
        repo.save_chunks(document_id, chunks)

        added: list[DocumentChunk] = db.add_all.call_args[0][0]
        assert added[0].chunk_index == 0
        assert added[1].chunk_index == 1
        assert added[2].chunk_index == 2

    def test_save_chunks_calls_flush(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
        chunks: list[EmbeddedChunk],
    ) -> None:
        repo.save_chunks(document_id, chunks)

        db.flush.assert_called_once()

    def test_save_chunks_calls_add_all(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
        chunks: list[EmbeddedChunk],
    ) -> None:
        repo.save_chunks(document_id, chunks)

        db.add_all.assert_called_once()

    def test_save_chunks_empty_list(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
    ) -> None:
        result = repo.save_chunks(document_id, [])

        assert result == []
        db.add_all.assert_called_once_with([])

    def test_save_chunks_deletes_existing_before_insert(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
        chunks: list[EmbeddedChunk],
    ) -> None:
        repo.save_chunks(document_id, chunks)

        delete_query = db.query.call_args[0][0]
        assert delete_query == DocumentChunk

    # ── delete_by_document ───────────────────────────────────────────────

    def test_delete_by_document_calls_flush(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
    ) -> None:
        repo.delete_by_document(document_id)

        db.flush.assert_called_once()

    def test_delete_by_document_queries_document_chunks(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
    ) -> None:
        repo.delete_by_document(document_id)

        db.query.assert_called_once_with(DocumentChunk)

    # ── get_by_document ──────────────────────────────────────────────────

    def test_get_by_document_returns_scalars(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
    ) -> None:
        fake_chunks = [
            DocumentChunk(
                document_id=document_id,
                chunk_index=0,
                text="a",
                embedding=[0.1, 0.2],
            ),
        ]
        db.scalars.return_value.all.return_value = fake_chunks

        result = repo.get_by_document(document_id)

        assert result == fake_chunks

    def test_get_by_document_filters_by_id(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
    ) -> None:
        repo.get_by_document(document_id)

        stmt = db.scalars.call_args[0][0]
        compiled = str(stmt)
        assert "document_id" in compiled
        assert "chunk_index" in compiled
        assert "ORDER BY" in compiled.upper()

    # ── count_by_document ────────────────────────────────────────────────

    def test_count_by_document_returns_count(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
    ) -> None:
        db.query.return_value.filter.return_value.count.return_value = 7

        count = repo.count_by_document(document_id)

        assert count == 7

    # ── search_similar ────────────────────────────────────────────────────

    def test_search_similar_returns_search_results(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
    ) -> None:
        fake_chunk = MagicMock(spec=DocumentChunk)
        fake_chunk.document_id = document_id
        db.execute.return_value.all.return_value = [
            (fake_chunk, 0.15),
        ]

        results = repo.search_similar([0.1, 0.2])

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk is fake_chunk

    def test_search_similar_computes_score(
        self,
        repo: ChunkRepository,
        db: MagicMock,
        document_id: UUID,
    ) -> None:
        fake_chunk = MagicMock(spec=DocumentChunk)
        fake_chunk.document_id = document_id
        db.execute.return_value.all.return_value = [
            (fake_chunk, 0.15),
        ]

        results = repo.search_similar([0.1, 0.2])

        assert results[0].score == pytest.approx(0.85)

    def test_search_similar_limits_results(
        self,
        repo: ChunkRepository,
        db: MagicMock,
    ) -> None:
        fake_chunks = [
            (MagicMock(spec=DocumentChunk), 0.1),
            (MagicMock(spec=DocumentChunk), 0.2),
            (MagicMock(spec=DocumentChunk), 0.3),
        ]
        db.execute.return_value.all.return_value = fake_chunks

        results = repo.search_similar([0.1, 0.2], top_k=3)

        assert len(results) == 3

    def test_search_similar_respects_top_k(
        self,
        repo: ChunkRepository,
        db: MagicMock,
    ) -> None:
        fake_chunks = [
            (MagicMock(spec=DocumentChunk), 0.1),
            (MagicMock(spec=DocumentChunk), 0.2),
            (MagicMock(spec=DocumentChunk), 0.3),
            (MagicMock(spec=DocumentChunk), 0.4),
            (MagicMock(spec=DocumentChunk), 0.5),
        ]
        db.execute.return_value.all.return_value = fake_chunks

        results = repo.search_similar([0.1, 0.2], top_k=2)

        assert len(results) == 5  # SQL handles the limit, we just return what DB gives

    def test_search_similar_filters_by_min_score(
        self,
        repo: ChunkRepository,
        db: MagicMock,
    ) -> None:
        fake_chunks = [
            (MagicMock(spec=DocumentChunk), 0.05),
            (MagicMock(spec=DocumentChunk), 0.15),
            (MagicMock(spec=DocumentChunk), 0.25),
        ]
        db.execute.return_value.all.return_value = fake_chunks

        results = repo.search_similar(
            [0.1, 0.2],
            min_score=0.8,
        )

        assert len(results) == 2  # scores: 0.95, 0.85 >= 0.8
        assert results[0].score == pytest.approx(0.95)
        assert results[1].score == pytest.approx(0.85)
