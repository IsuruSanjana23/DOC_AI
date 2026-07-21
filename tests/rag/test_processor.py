from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.rag.chunker import Chunker, TextChunk
from app.rag.embedder import BaseEmbedder, EmbeddedChunk, EmbeddingError
from app.rag.extractor import ExtractedPage, PDFExtractionError, PDFExtractor
from app.rag.processor import DocumentProcessingError, DocumentProcessor


class TestDocumentProcessor:

    @pytest.fixture
    def extractor(self) -> MagicMock:
        mock = MagicMock(spec=PDFExtractor)
        mock.extract.return_value = [
            ExtractedPage(1, "First page content."),
            ExtractedPage(2, "Second page content."),
        ]
        return mock

    @pytest.fixture
    def chunker(self) -> Chunker:
        return Chunker(chunk_size=1000, chunk_overlap=200)

    @pytest.fixture
    def embedder(self) -> MagicMock:
        mock = MagicMock(spec=BaseEmbedder)
        mock.embed.return_value = [
            EmbeddedChunk(0, "First page content.", 1, np.array([0.1, 0.2])),
            EmbeddedChunk(1, "Second page content.", 2, np.array([0.3, 0.4])),
        ]
        return mock

    @pytest.fixture
    def processor(
        self,
        extractor: MagicMock,
        chunker: Chunker,
        embedder: MagicMock,
    ) -> DocumentProcessor:
        return DocumentProcessor(
            extractor=extractor,
            chunker=chunker,
            embedder=embedder,
        )

    def test_process_returns_embedded_chunks(
        self,
        processor: DocumentProcessor,
        extractor: MagicMock,
        embedder: MagicMock,
    ) -> None:
        result = processor.process(Path("doc.pdf"))

        assert len(result) == 2
        assert isinstance(result[0], EmbeddedChunk)
        assert isinstance(result[1], EmbeddedChunk)

        extractor.extract.assert_called_once_with(Path("doc.pdf"))
        embedder.embed.assert_called_once()

    def test_process_calls_extract_then_chunk_then_embed(
        self,
        extractor: MagicMock,
        chunker: Chunker,
        embedder: MagicMock,
    ) -> None:
        call_order: list[str] = []

        extractor.extract.side_effect = lambda p: (
            call_order.append("extract")
            or [ExtractedPage(1, "Content.")]
        )

        embedder.embed.side_effect = lambda c: (
            call_order.append("embed")
            or [EmbeddedChunk(0, "Content.", 1, np.array([0.1, 0.2]))]
        )

        processor = DocumentProcessor(extractor, chunker, embedder)
        processor.process(Path("doc.pdf"))

        assert call_order == ["extract", "embed"]

    def test_process_forwards_chunked_text_to_embedder(
        self,
        processor: DocumentProcessor,
        embedder: MagicMock,
    ) -> None:
        processor.process(Path("doc.pdf"))

        actual_chunks: list[TextChunk] = embedder.embed.call_args[0][0]
        assert len(actual_chunks) == 2
        assert actual_chunks[0].text == "First page content."
        assert actual_chunks[1].text == "Second page content."

    def test_process_preserves_page_numbers_in_chunks(
        self,
        processor: DocumentProcessor,
        embedder: MagicMock,
    ) -> None:
        processor.process(Path("doc.pdf"))

        actual_chunks: list[TextChunk] = embedder.embed.call_args[0][0]
        assert actual_chunks[0].page_number == 1
        assert actual_chunks[1].page_number == 2

    def test_process_handles_single_page(
        self,
        extractor: MagicMock,
        chunker: Chunker,
        embedder: MagicMock,
    ) -> None:
        extractor.extract.return_value = [
            ExtractedPage(1, "Only page."),
        ]
        embedder.embed.return_value = [
            EmbeddedChunk(0, "Only page.", 1, np.array([0.5])),
        ]
        processor = DocumentProcessor(extractor, chunker, embedder)

        result = processor.process(Path("doc.pdf"))

        assert len(result) == 1
        assert result[0].text == "Only page."

    def test_process_handles_empty_pages(
        self,
        extractor: MagicMock,
        chunker: Chunker,
        embedder: MagicMock,
    ) -> None:
        extractor.extract.return_value = []
        embedder.embed.return_value = []
        processor = DocumentProcessor(extractor, chunker, embedder)

        result = processor.process(Path("doc.pdf"))

        assert result == []

    def test_process_raises_on_missing_file(
        self,
        extractor: MagicMock,
        chunker: Chunker,
        embedder: MagicMock,
    ) -> None:
        extractor.extract.side_effect = PDFExtractionError("File not found")
        processor = DocumentProcessor(extractor, chunker, embedder)

        with pytest.raises(PDFExtractionError, match="File not found"):
            processor.process(Path("missing.pdf"))

    def test_process_raises_on_embedding_failure(
        self,
        extractor: MagicMock,
        chunker: Chunker,
        embedder: MagicMock,
    ) -> None:
        embedder.embed.side_effect = EmbeddingError("Model OOM")
        processor = DocumentProcessor(extractor, chunker, embedder)

        with pytest.raises(EmbeddingError, match="Model OOM"):
            processor.process(Path("doc.pdf"))

    def test_process_wraps_unexpected_extraction_error(
        self,
        extractor: MagicMock,
        chunker: Chunker,
        embedder: MagicMock,
    ) -> None:
        extractor.extract.side_effect = PermissionError("Access denied")
        processor = DocumentProcessor(extractor, chunker, embedder)

        with pytest.raises(DocumentProcessingError) as exc_info:
            processor.process(Path("doc.pdf"))

        assert isinstance(exc_info.value.__cause__, PermissionError)

    def test_process_wraps_unexpected_embedding_error(
        self,
        extractor: MagicMock,
        chunker: Chunker,
        embedder: MagicMock,
    ) -> None:
        embedder.embed.side_effect = ValueError("Invalid batch size")
        processor = DocumentProcessor(extractor, chunker, embedder)

        with pytest.raises(DocumentProcessingError) as exc_info:
            processor.process(Path("doc.pdf"))

        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_process_logs_success(
        self,
        processor: DocumentProcessor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        processor.process(Path("doc.pdf"))

        records = [r for r in caplog.records if r.name == "app.rag.processor"]
        info_messages = [r.message for r in records if r.levelname == "INFO"]
        assert any("Starting document processing" in m for m in info_messages)
        assert any("Document processed" in m for m in info_messages)

    def test_process_logs_each_step(
        self,
        processor: DocumentProcessor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)

        processor.process(Path("doc.pdf"))

        records = [
            r for r in caplog.records
            if r.name == "app.rag.processor"
        ]
        info = [r.message for r in records if r.levelname == "INFO"]
        step_logs = [m for m in info if "Extracted" in m or "Chunked" in m or "Generated" in m]
        assert len(step_logs) == 3

    def test_process_stops_on_extraction_failure(
        self,
        processor: DocumentProcessor,
        extractor: MagicMock,
        embedder: MagicMock,
    ) -> None:
        extractor.extract.side_effect = PDFExtractionError("Boom")

        with pytest.raises(PDFExtractionError):
            processor.process(Path("doc.pdf"))

        embedder.embed.assert_not_called()
