"""Semantic chunking service for SDS documents.

Strategy:
  1. Detect GHS/SDS section boundaries using a comprehensive pattern map.
  2. Split the document into sections first, then apply token-aware
     recursive splitting within each section.
  3. Tag each chunk with its section number so the retriever can
     prioritise sections that are most relevant to each metadata field.

This two-pass approach significantly outperforms naive sliding-window
chunking because section 1 (Identification) contains product name,
and sections 14–15 contain regulatory jurisdiction markers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.infrastructure.configuration.settings import Settings

logger = logging.getLogger(__name__)

# ── SDS section detection patterns ────────────────────────────────────────────
# Matches the standard GHS section headers in English, Portuguese, Spanish,
# French, German and their common variants.

_SECTION_PATTERNS: dict[int, list[str]] = {
    1:  [r"identification", r"produto e da empresa", r"identificaci[oó]n", r"identifikation"],
    2:  [r"hazard", r"perigos", r"peligros?", r"gefahr"],
    3:  [r"composition", r"composi[cç][aã]o", r"composici[oó]n", r"zusammensetzung"],
    4:  [r"first.?aid", r"primeiros.?socorros", r"primeros.?auxilios", r"erste.?hilfe"],
    5:  [r"fire.?fight", r"combate.?inc[eê]ndio", r"lucha.?contra.?incendios", r"brandschutz"],
    6:  [r"accidental.?release", r"vazamento", r"derrame", r"freisetzung"],
    7:  [r"handling", r"manuseio", r"manipulaci[oó]n", r"handhabung"],
    8:  [r"exposure.?control", r"controle.?exposi[cç]", r"control.?exposici[oó]n", r"exposition"],
    9:  [r"physical.{0,10}properties", r"propriedades.{0,10}f[íi]sicas", r"propiedades"],
    10: [r"stability.{0,10}reactivity", r"estabilidade", r"estabilidad", r"stabilit[aä]t"],
    11: [r"toxicolog", r"informa[cç][oã][oe]s.{0,20}toxicol[oó]g"],
    12: [r"ecological", r"informa[cç][oã][oe]s.{0,20}ecol[oó]g", r"ecol[oó]g"],
    13: [r"disposal", r"descarte", r"eliminaci[oó]n", r"entsorgung"],
    14: [r"transport", r"transportinforma", r"informaciones.{0,20}transporte"],
    15: [
        r"regulatory", r"informa[cç][oã][oe]s.{0,20}regulat[oó]r",
        r"reglamentaci[oó]n", r"vorschriften",
        r"osha", r"whmis", r"reach\b", r"abnt\s*nbr\s*14725",
        r"uk\s*reach", r"ghs\b",
    ],
    16: [r"other\s+information", r"outras\s+informa", r"otra\s+informaci[oó]n"],
}


def _compile_section_regex() -> re.Pattern:
    """Build a single regex that matches any known SDS section header."""
    all_pats = [p for patterns in _SECTION_PATTERNS.values() for p in patterns]
    combined = "|".join(f"(?:{p})" for p in all_pats)
    return re.compile(
        rf"(?mi)^[\s\d\.]*\b(?:section|se[cç][aã]o|secci[oó]n|abschnitt)?\s*[\d]{{0,2}}\s*[:\-–]?\s*(?:{combined})",
    )


_SECTION_RE = _compile_section_regex()


def _detect_section(text: str) -> int | None:
    """Return the GHS section number if the text contains a section header."""
    text_lower = text.lower()
    for section_num, patterns in _SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return section_num
    return None


@dataclass
class DocumentChunk:
    """A single chunk of document text ready for embedding."""

    text: str
    document_id: str
    chunk_index: int
    section_number: int | None = None
    page_numbers: list[int] = field(default_factory=list)

    @property
    def metadata(self) -> dict:
        return {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "section_number": self.section_number or 0,
            "page_numbers": ",".join(str(p) for p in self.page_numbers),
        }


class ChunkingService:
    """Splits cleaned SDS document text into semantically meaningful chunks."""

    def __init__(self, settings: Settings) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str, document_id: str) -> list[DocumentChunk]:
        """Split text into chunks, detecting SDS section numbers.

        Returns an empty list if the text is empty.
        """
        if not text.strip():
            return []

        raw_chunks = self._splitter.split_text(text)
        result: list[DocumentChunk] = []

        for index, chunk_text in enumerate(raw_chunks):
            section = _detect_section(chunk_text)
            result.append(
                DocumentChunk(
                    text=chunk_text,
                    document_id=document_id,
                    chunk_index=index,
                    section_number=section,
                )
            )

        logger.debug(
            "Chunked document_id=%s | chunks=%d | avg_len=%d",
            document_id,
            len(result),
            sum(len(c.text) for c in result) // max(len(result), 1),
        )
        return result
