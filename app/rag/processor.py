import logging
from pathlib import Path

from app.rag.chunker import Chunker, TextChunk
from app.rag.embedder import BaseEmbedder, EmbeddedChunk, EmbeddingError
from app.rag.extractor import ExtractedPage, PDFExtractionError, PDFExtractor

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Raised when the full document processing pipeline fails."""


class DocumentProcessor:

    def __init__(
        self,
        extractor: PDFExtractor,
        chunker: Chunker,
        embedder: BaseEmbedder,
    ) -> None:
        self._extractor = extractor
        self._chunker = chunker
        self._embedder = embedder

    def process(self, path: Path) -> list[EmbeddedChunk]:
        logger.info("Starting document processing: %s", path)

        pages = self._extract(path)
        chunks = self._chunk(pages)
        embedded = self._embed(chunks)

        logger.info(
            "Document processed: %s (%d pages, %d chunks, %d embeddings)",
            path,
            len(pages),
            len(chunks),
            len(embedded),
        )

        return embedded

    def _extract(self, path: Path) -> list[ExtractedPage]:
        try:
            pages = self._extractor.extract(path)
            logger.info("Extracted %d pages from %s", len(pages), path)
            return pages
        except PDFExtractionError:
            raise
        except Exception as e:
            raise DocumentProcessingError(
                f"Unexpected error during PDF extraction: {path}"
            ) from e

    def _chunk(
        self, pages: list[ExtractedPage]
    ) -> list[TextChunk]:
        try:
            chunks = self._chunker.chunk_pages(pages)
            logger.info("Chunked into %d chunks", len(chunks))
            return chunks
        except Exception as e:
            raise DocumentProcessingError(
                f"Unexpected error during chunking: {e}"
            ) from e

    def _embed(
        self, chunks: list[TextChunk]
    ) -> list[EmbeddedChunk]:
        try:
            embedded = self._embedder.embed(chunks)
            logger.info(
                "Generated %d embeddings", len(embedded)
            )
            return embedded
        except EmbeddingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(
                f"Unexpected error during embedding: {e}"
            ) from e
