import logging
from dataclasses import dataclass
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    pass


@dataclass(slots=True)
class ExtractedPage:
    page_number: int
    text: str


class PDFExtractor:

    def extract(self, file_path: Path) -> list[ExtractedPage]:
        if not file_path.exists():
            msg = f"PDF file not found: {file_path}"
            logger.error(msg)
            raise PDFExtractionError(msg)

        try:
            with fitz.open(file_path) as document:
                pages: list[ExtractedPage] = []
                for index, page in enumerate(document):
                    text = page.get_text().strip()
                    if text:
                        pages.append(
                            ExtractedPage(
                                page_number=index + 1,
                                text=text,
                            )
                        )
                return pages
        except PDFExtractionError:
            raise
        except Exception as e:
            logger.exception("Failed to extract text from %s", file_path)
            raise PDFExtractionError(
                f"PDF extraction failed for {file_path}: {e}"
            ) from e
