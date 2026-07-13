"""PDF document reader using PyMuPDF.

Responsibilities:
  - Validate the file is a real, readable PDF.
  - Extract raw text page-by-page.
  - Return both the full concatenated text and a per-page breakdown
    so the chunking service has page-boundary metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from app.domain.exceptions.base import InvalidDocumentException, PDFExtractionException

logger = logging.getLogger(__name__)

_MIN_TEXT_LENGTH = 50  # characters — below this the PDF is likely scanned/image-only


@dataclass(frozen=True)
class ExtractedPage:
    """Text content of a single PDF page."""

    page_number: int  # 1-indexed
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    """Result of reading a PDF — full text plus per-page breakdown."""

    pages: list[ExtractedPage]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)


class DocumentReader:
    """Reads PDF files and returns structured text content."""

    def read(self, file_path: Path) -> ExtractedDocument:
        """Extract all text from the PDF at the given path.

        Raises:
            InvalidDocumentException: if the file does not exist or is not a PDF.
            PDFExtractionException: if PyMuPDF cannot open or read the file.
        """
        if not file_path.exists():
            raise InvalidDocumentException(f"File not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise InvalidDocumentException(
                f"Unsupported file type '{file_path.suffix}'. Only PDF files are accepted."
            )

        logger.debug("Reading PDF: %s", file_path.name)

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            raise PDFExtractionException(
                f"Cannot open PDF '{file_path.name}': {exc}"
            ) from exc

        try:
            pages: list[ExtractedPage] = []
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text("text")
                pages.append(ExtractedPage(page_number=page_index + 1, text=text))

            result = ExtractedDocument(pages=pages)

            if len(result.full_text.strip()) < _MIN_TEXT_LENGTH:
                raise InvalidDocumentException(
                    f"PDF '{file_path.name}' contains insufficient extractable text. "
                    "The document may be image-based or scanned. "
                    "Please provide a text-based PDF."
                )

            logger.info(
                "PDF read: %s | pages=%d | chars=%d",
                file_path.name,
                result.page_count,
                len(result.full_text),
            )
            return result

        except InvalidDocumentException:
            raise
        except Exception as exc:
            raise PDFExtractionException(
                f"Failed to extract text from '{file_path.name}': {exc}"
            ) from exc
        finally:
            doc.close()
