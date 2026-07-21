import logging

from app.rag.embedder import BaseEmbedder, EmbeddingError
from app.repositories.chunk_repository import ChunkRepository, SearchResult

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Raised when the retrieval process fails.

    This is the sole exception type callers should catch when using
    :class:`Retriever`. Lower-level exceptions (embedding failures,
    database errors) are wrapped with exception chaining so the root
    cause is preserved in ``__cause__``.
    """


class Retriever:

    def __init__(
        self,
        repository: ChunkRepository,
        embedder: BaseEmbedder,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        collection_id: str | None = None,
    ) -> list[SearchResult]:
        effective_top_k = top_k or self._top_k
        query_length = len(query)
        logger.debug(
            "Retrieve request — query_length=%d top_k=%d min_score=%s",
            query_length,
            effective_top_k,
            self._min_score,
        )

        try:
            query_vector = self._embedder.embed_query(query).tolist()
        except EmbeddingError as e:
            logger.warning(
                "Embedding failed for query length %d: %s",
                query_length,
                e,
            )
            raise RetrieverError(
                f"Failed to embed query (length={query_length}): {e}"
            ) from e
        except Exception as e:
            logger.exception(
                "Unexpected embedding error for query length %d",
                query_length,
            )
            raise RetrieverError(
                f"Unexpected embedding error (length={query_length}): {e}"
            ) from e

        logger.debug("Embedding succeeded — vector dimension=%d", len(query_vector))

        try:
            from uuid import UUID
            cid = UUID(collection_id) if collection_id else None
            results = self._repository.search_similar(
                query_vector,
                top_k=effective_top_k,
                min_score=self._min_score,
                collection_id=cid,
            )
        except Exception as e:
            logger.exception(
                "Vector search failed — query_length=%d top_k=%d",
                query_length,
                effective_top_k,
            )
            raise RetrieverError(
                f"Vector search failed (top_k={effective_top_k}): {e}"
            ) from e

        logger.info(
            "Retrieve succeeded — query_length=%d top_k=%d results=%d",
            query_length,
            effective_top_k,
            len(results),
        )
        return results
