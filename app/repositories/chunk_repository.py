from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk
from app.rag.embedder import EmbeddedChunk


@dataclass(slots=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


class ChunkRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def save_chunks(
        self,
        document_id: UUID,
        chunks: list[EmbeddedChunk],
    ) -> list[DocumentChunk]:
        self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
        ).delete()

        orm_chunks = [
            DocumentChunk(
                document_id=document_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                embedding=chunk.vector.tolist(),
            )
            for chunk in chunks
        ]

        self.db.add_all(orm_chunks)
        self.db.flush()

        return orm_chunks

    def delete_by_document(self, document_id: UUID) -> None:
        self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
        ).delete()
        self.db.flush()

    def get_by_document(self, document_id: UUID) -> list[DocumentChunk]:
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(self.db.scalars(stmt).all())

    def count_by_document(self, document_id: UUID) -> int:
        return self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
        ).count()

    def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        distance_col = DocumentChunk.embedding.cosine_distance(query_vector)
        stmt = (
            select(DocumentChunk, distance_col.label("distance"))
            .order_by(distance_col)
            .limit(top_k)
        )
        rows = self.db.execute(stmt).all()
        results = []
        for chunk, distance in rows:
            score = 1.0 - distance
            if min_score is not None and score < min_score:
                continue
            results.append(SearchResult(chunk=chunk, score=score))
        return results
