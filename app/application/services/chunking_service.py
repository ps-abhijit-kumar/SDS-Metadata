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
# Generic GHS section patterns supporting all major international SDS formats
# (English, Spanish, Portuguese, German, French, Swedish, Italian, Dutch, Polish, etc.)

_SECTION_PATTERNS: dict[int, list[str]] = {
    1:  [r"identification", r"produto e da empresa", r"identificaci[oó]n", r"identifikation", r"namnet p[aå] [aä]mnet/blandningen"],
    2:  [r"hazard", r"perigos", r"peligros?", r"gefahr", r"farliga egenskaper", r"faror"],
    3:  [r"composition", r"composi[cç][aã]o", r"composici[oó]n", r"zusammensetzung", r"sammans[aä]ttning"],
    4:  [r"first.?aid", r"primeiros.?socorros", r"primeros.?auxilios", r"erste.?hilfe", r"f[oö]rsta hj[aä]lpen", r"emergency\s+and\s+first\s+aid", r"first\s+aid\s+exposures", r"emergency\s+exposures"],
    5:  [r"fire.?fight", r"combate.?inc[eê]ndio", r"lucha.?contra.?incendios", r"brandschutz", r"brandbek[aä]mpnings"],
    6:  [r"accidental.?release", r"vazamento", r"derrame", r"freisetzung", r"oavsiktliga utsl[aä]pp"],
    7:  [r"handling", r"manuseio", r"manipulaci[oó]n", r"handhabung", r"hantering och lagring", r"lagring"],
    8:  [r"exposure.?control", r"controle.?exposi[cç]", r"control.?exposici[oó]n", r"exposition", r"begr[aä]nsning av exponeringen", r"personligt skydd"],
    9:  [r"physical.{0,10}properties", r"propriedades.{0,10}f[íi]sicas", r"propiedades", r"fysikaliska och kemiska"],
    10: [r"stability.{0,10}reactivity", r"estabilidade", r"estabilidad", r"stabilit[aä]t", r"stabilitet och reaktivitet"],
    11: [r"toxicolog", r"informa[cç][oã][oe]s.{0,20}toxicol[oó]g", r"toxikologisk information"],
    12: [r"ecological", r"informa[cç][oã][oe]s.{0,20}ecol[oó]g", r"ecol[oó]g", r"ekologisk information"],
    13: [r"disposal", r"descarte", r"eliminaci[oó]n", r"entsorgung", r"avfallshantering"],
    14: [r"transport", r"transportinforma", r"informaciones.{0,20}transporte", r"transportinformation"],
    15: [
        r"regulatory", r"informa[cç][oã][oe]s.{0,20}regulat[oó]r",
        r"reglamentaci[oó]n", r"vorschriften", r"g[aä]llande f[oö]reskrifter",
        r"osha", r"whmis", r"abnt\s*nbr\s*14725",
        r"uk\s*reach\b", r"ghs\b",
    ],
    16: [r"other\s+information", r"outras\s+informa", r"otra\s+informaci[oó]n", r"annan information"],
}


_HEADER_NUM_RE = re.compile(
    r"(?mi)^[\s\*\#\-\.\s]*(?:section|secci[oó]n|se[cç][aã]o|abschnitt|avsnitt|punkt|artikkel|rubriek|sezione|sekcja|odd[ií]l|paragraf)\s*[:\-–\.]?\s*(\d{1,2})\b"
)
_NUMBERED_HEADER_RE = re.compile(
    r"(?mi)^[\s\*\#\-\.\s]*(\d{1,2})\s*[\.\:\-–]\s+[A-Za-z\u00C0-\u017F]"
)


def _detect_section(text: str) -> int | None:
    """Return the GHS section number if the text contains a section header."""
    text_lower = text.lower()

    # 1. Direct section header keyword match (e.g. "SECCIÓN 4.", "SECTION 4:", "AVSNITT 4:")
    header_match = _HEADER_NUM_RE.search(text)
    if header_match:
        sec = int(header_match.group(1))
        if 1 <= sec <= 16:
            return sec

    # 2. Number-prefixed section header match (e.g. "4. FIRST AID MEASURES", "7. HANDLING")
    num_match = _NUMBERED_HEADER_RE.search(text)
    if num_match:
        sec = int(num_match.group(1))
        if 1 <= sec <= 16:
            # Verify that the line contains known section keywords or title for this section number
            sec_patterns = _SECTION_PATTERNS.get(sec, [])
            if any(re.search(p, text_lower) for p in sec_patterns):
                return sec

    # 3. Structured section pattern match (header-anchored)
    for section_num, patterns in _SECTION_PATTERNS.items():
        for pattern in patterns:
            header_pat = (
                rf"(?mi)^[\s\d\.\*\#\-]*\b(?:section|se[cç][aã]o|secci[oó]n|abschnitt|avsnitt|punkt)?\s*"
                rf"{section_num}?\s*[:\-–\.]?\s*(?:{pattern})"
            )
            if re.search(header_pat, text_lower):
                return section_num

    return None


_SECTION_TITLES: dict[int, str] = {
    1: "Identification",
    2: "Hazard(s) Identification",
    3: "Composition / Information on Ingredients",
    4: "First-Aid Measures",
    5: "Fire-Fighting Measures",
    6: "Accidental Release Measures",
    7: "Handling and Storage",
    8: "Exposure Controls / Personal Protection",
    9: "Physical and Chemical Properties",
    10: "Stability and Reactivity",
    11: "Toxicological Information",
    12: "Ecological Information",
    13: "Disposal Considerations",
    14: "Transport Information",
    15: "Regulatory Information",
    16: "Other Information",
}


@dataclass
class DocumentChunk:
    """A single chunk of document text ready for embedding."""

    text: str
    document_id: str
    chunk_index: int
    filename: str = ""
    document_hash: str = ""
    section_number: int | None = None
    section_title: str = "General"
    page: int = 1

    @property
    def metadata(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "document_hash": self.document_hash,
            "chunk_index": self.chunk_index,
            "section": str(self.section_number or 0),
            "section_number": self.section_number or 0,
            "section_title": self.section_title,
            "page": self.page,
            "page_numbers": str(self.page),
        }


_SECTION_SPLIT_RE = re.compile(
    r"(?mi)^(?:[\s\*\-\#]*)(?=(?:SECTION|SECCI[OÓ]N|SE[CÇ][AÃ]O|ABSCHNITT|AVSNITT|PUNKT|SEZIONE|SEKCJA)\s*[:\-–\.]?\s*\d{1,2}\b|"
    r"\d{1,2}\s*[\.\:\-–]\s+(?:IDENTIFICATION|HAZARD|COMPOSITION|FIRST|FIRE|ACCIDENTAL|HANDLING|EXPOSURE|PHYSICAL|STABILITY|TOXICOLOGICAL|ECOLOGICAL|DISPOSAL|TRANSPORT|REGULATORY|"
    r"FARLIGA|SAMMANS[AÄ]TTNING|F[OÖ]RSTA|BRAND|HANTERING|BEGR[AÄ]NSNING|FYSIKALISKA|STABILITET|TOXIKOLOGISK|EKOLOGISK|AVFALLS|G[AÄ]LLANDE)|"
    r"EMERGENCY AND FIRST AID|FIRST AID EXPOSURES|EXPOSURE CONTROLS)",
    re.IGNORECASE,
)


def _split_into_section_blocks(text: str) -> list[str]:
    """Pre-split document page text into section blocks based on GHS and legacy section header boundaries."""
    if not text or not text.strip():
        return []
    lines = text.splitlines()
    blocks: list[str] = []
    current_block: list[str] = []

    for line in lines:
        if current_block and _SECTION_SPLIT_RE.match(line):
            blocks.append("\n".join(current_block).strip())
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block).strip())

    return [b for b in blocks if b.strip()]


class ChunkingService:
    """Splits cleaned SDS document text into semantically meaningful chunks."""

    def __init__(self, settings: Settings) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_pages(
        self,
        pages: list,  # ExtractedPage objects
        document_id: str,
        filename: str = "",
        document_hash: str = "",
    ) -> list[DocumentChunk]:
        """Split text per page while maintaining 1-indexed page numbers and GHS section titles."""
        result: list[DocumentChunk] = []
        chunk_counter = 0

        current_section = 1
        current_title = _SECTION_TITLES[1]

        for p in pages:
            page_text = p.text.strip()
            if not page_text:
                continue

            sec_blocks = _split_into_section_blocks(page_text)
            for block in sec_blocks:
                page_chunks = self._splitter.split_text(block)
                for chunk_text in page_chunks:
                    detected = _detect_section(chunk_text)
                    if detected:
                        current_section = detected
                        current_title = f"Section {detected}: {_SECTION_TITLES.get(detected, 'General')}"

                    sec = current_section
                    title = current_title

                    result.append(
                        DocumentChunk(
                            text=chunk_text,
                            document_id=document_id,
                            filename=filename,
                            document_hash=document_hash,
                            chunk_index=chunk_counter,
                            section_number=sec,
                            section_title=title,
                            page=p.page_number,
                        )
                    )
                    chunk_counter += 1

        logger.debug("Chunked %d pages into %d chunks for document_id=%s", len(pages), len(result), document_id)
        return result

    def chunk(self, text: str, document_id: str, filename: str = "", document_hash: str = "") -> list[DocumentChunk]:
        """Split fallback text into chunks if pages list is unavailable."""
        if not text.strip():
            return []

        sec_blocks = _split_into_section_blocks(text)
        result: list[DocumentChunk] = []
        chunk_counter = 0

        current_section = 1
        current_title = _SECTION_TITLES[1]

        for block in sec_blocks:
            raw_chunks = self._splitter.split_text(block)
            for chunk_text in raw_chunks:
                detected = _detect_section(chunk_text)
                if detected:
                    current_section = detected
                    current_title = f"Section {detected}: {_SECTION_TITLES.get(detected, 'General')}"

                sec = current_section
                title = current_title

                result.append(
                    DocumentChunk(
                        text=chunk_text,
                        document_id=document_id,
                        filename=filename,
                        document_hash=document_hash,
                        chunk_index=chunk_counter,
                        section_number=sec,
                        section_title=title,
                        page=1,
                    )
                )
                chunk_counter += 1

        return result

