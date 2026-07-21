import logging
import re
from dataclasses import dataclass

from app.rag.extractor import ExtractedPage

logger = logging.getLogger(__name__)

SEPARATORS = ["\n\n", "\n", r"\. ", r"! ", r"\? ", ", ", " "]


@dataclass(slots=True)
class TextChunk:
    chunk_index: int
    text: str
    page_number: int | None


class Chunker:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[ExtractedPage]) -> list[TextChunk]:
        chunks: list[TextChunk] = []

        for page in pages:
            page_chunks = self._build_chunks(page.text, page.page_number)
            chunks.extend(page_chunks)

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        logger.info(
            "Chunked %d pages into %d chunks (size=%d, overlap=%d)",
            len(pages), len(chunks), self.chunk_size, self.chunk_overlap,
        )
        return chunks

    def chunk_text(self, text: str) -> list[TextChunk]:
        chunks = self._build_chunks(text, page_number=None)
        logger.info(
            "Chunked text (%d chars) into %d chunks (size=%d, overlap=%d)",
            len(text), len(chunks), self.chunk_size, self.chunk_overlap,
        )
        return chunks

    def _build_chunks(
        self,
        text: str,
        page_number: int | None,
    ) -> list[TextChunk]:
        if not text.strip():
            return []

        segments = self._split_into_segments(text, separator_idx=0)
        return self._merge_into_chunks(segments, page_number)

    def _split_into_segments(self, text: str, separator_idx: int) -> list[str]:
        if separator_idx >= len(SEPARATORS):
            return self._hard_split_segments(text)

        raw_separator = SEPARATORS[separator_idx]
        parts = re.split(f"({raw_separator})", text)

        segments: list[str] = []
        buffer = ""
        for i in range(0, len(parts), 2):
            chunk = parts[i]
            sep = parts[i + 1] if i + 1 < len(parts) else ""
            piece = chunk + sep

            if not piece:
                continue

            if len(piece) <= self.chunk_size:
                if buffer:
                    segments.append(buffer + piece)
                    buffer = ""
                else:
                    segments.append(piece)
            else:
                if buffer:
                    segments.append(buffer)
                    buffer = ""
                segments.extend(
                    self._split_into_segments(piece, separator_idx + 1)
                )

        return segments

    def _hard_split_segments(self, text: str) -> list[str]:
        return [
            text[i:i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]

    def _merge_into_chunks(
        self,
        segments: list[str],
        page_number: int | None,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        current = ""

        for seg in segments:
            candidate = f"{current}\n{seg}" if current else seg

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(
                    TextChunk(
                        chunk_index=len(chunks),
                        text=current,
                        page_number=page_number,
                    )
                )

                overlap = self._take_overlap(current)
                current = f"{overlap}\n{seg}" if overlap else seg

        if current:
            chunks.append(
                TextChunk(
                    chunk_index=len(chunks),
                    text=current,
                    page_number=page_number,
                )
            )

        return chunks

    def _take_overlap(self, text: str) -> str:
        if self.chunk_overlap <= 0 or len(text) <= self.chunk_overlap:
            return text

        raw = text[-self.chunk_overlap:]

        newline_idx = raw.find("\n")
        if newline_idx > 0:
            return raw[newline_idx + 1:]

        space_idx = raw.find(" ")
        if space_idx > 0:
            return raw[space_idx + 1:]

        return raw
