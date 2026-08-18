"""Deterministic SDS section detector for multilingual section-aware RAG retrieval."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Multilingual keywords & regex patterns for standard GHS Sections 1-16
_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Section 1: Identification
    (
        "1",
        re.compile(
            r"\b(?:product\s+identification|product\s+name|trade\s+name|commercial\s+name|manufacturer|supplier|"
            r"company\s+name|producer|distributor|emergency\s+telephone|intended\s+use|identificación\s+del\s+producto|"
            r"identificação\s+do\s+produto|identificação\s+da\s+empresa)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 2: Hazard Identification
    (
        "2",
        re.compile(
            r"\b(?:hazard|hazards|danger|warning|hazard\s+statements|identificación\s+de\s+los\s+peligros|"
            r"peligros|peligrosidad|risks|risk\s+phrases|pictogram|ghs\s+classification|classificação\s+de\s+perigo)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 3: Composition / Information on Ingredients
    (
        "3",
        re.compile(
            r"\b(?:composition|ingredients?|components?|chemicals?|chemical\s+composition|chemical\s+formula|"
            r"formula|molecular\s+weight|cas\s+number|cas\s+no|cas\s+registry|substance|mixture|composición|"
            r"ingredientes|componentes|sustancias|composição)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 4: First Aid Measures
    (
        "4",
        re.compile(
            r"\b(?:first\s*aid|primeros\s+auxilios|premiers\s+secours|erste\s*hilfe|primeiros\s+socorros|"
            r"emergency\s+and\s+first\s+aid|first\s+aid\s+exposures|emergency\s+exposures|"
            r"exposed|inhalation|skin\s+contact|eye\s+contact|swallowed|ingestion|inhalación|contacto\s+con|ingestión|"
            r"eye\s+injury|breathe|swallow|touch)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 5: Fire Fighting Measures
    (
        "5",
        re.compile(
            r"\b(?:fire\s*fighting|firefighting|extinguishing\s+media|lucha\s+contra\s+incendios|"
            r"incendio|incendios|extinción|flammable|flammability|combustion|fire\s+hazard)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 6: Accidental Release Measures
    (
        "6",
        re.compile(
            r"\b(?:accidental\s+release|spill|spillage|leak|leakage|spill\s+cleanup|vertido\s+accidental|"
            r"derrame|fuga|limpieza|environmental\s+precautions\s+spill|containment)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 7: Handling and Storage
    (
        "7",
        re.compile(
            r"\b(?:handling|storage|how\s+should\s+it\s+be\s+stored|manipulación|almacenamiento|almacenar|guardar|"
            r"safe\s+handling|storage\s+conditions|incompatibilities|conditions\s+for\s+safe\s+storage|manuseio)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 8: Exposure Controls / Personal Protection (PPE)
    (
        "8",
        re.compile(
            r"\b(?:exposure\s+controls?|ppe|protective\s+equipment|personal\s+protection|controles\s+de\s+exposición|"
            r"protección\s+personal|equipo\s+de\s+protección|mascarilla|guantes|protect|protection|protect\s+myself|"
            r"protective|respiratory\s+protection|gloves|safety\s+goggles|safety\s+glasses|impervious\s+clothing|"
            r"ventilation|occupational\s+exposure|exposure\s+limit|threshold\s+limit)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 9: Physical and Chemical Properties
    (
        "9",
        re.compile(
            r"\b(?:physical\s+properties|physical\s+and\s+chemical|flash\s+point|boiling\s+point|melting\s+point|"
            r"solubility|density|ph\s+value|propiedades\s+físicas|punto\s+de\s+inflamación|punto\s+de\s+ebullición|aspecto|olor)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 10: Stability and Reactivity
    (
        "10",
        re.compile(
            r"\b(?:stability|reactivity|estabilidad|reactividad|incompatibilidad|descomposición|hazardous\s+decomposition|"
            r"chemical\s+stability)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 11: Toxicological Information
    (
        "11",
        re.compile(
            r"\b(?:toxicology|toxic\s+effects|health\s+effects|información\s+toxicológica|toxicidad|ld50|lc50|"
            r"carcinogen|carcinogenicity|mutagenicity|teratogenicity|acute\s+toxicity)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 12: Ecological Information
    (
        "12",
        re.compile(
            r"\b(?:ecological|environmental|información\s+ecológica|ecotoxicidad|medio\s+ambiente|aquatic\s+toxicity|"
            r"biodegradability|bioaccumulation|ecotoxicity)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 13: Disposal Considerations
    (
        "13",
        re.compile(
            r"\b(?:disposal|elimination|consideraciones\s+relativas\s+a\s+la\s+eliminación|residuos|eliminación|"
            r"waste\s+treatment|waste\s+disposal)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 14: Transport Information
    (
        "14",
        re.compile(
            r"\b(?:transport|shipping|un\s+number|información\s+relativa\s+al\s+transporte|transporte|adr|rid|imdg|iata|"
            r"dot\s+classification|proper\s+shipping\s+name)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 15: Regulatory Information
    (
        "15",
        re.compile(
            r"\b(?:regulatory|regulation|reach|clp|osha|whmis|tsca|jurisdiction|información\s+reglamentaria|reglamentación|"
            r"safety\s+health\s+and\s+environmental\s+regulations)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 16: Other Information
    (
        "16",
        re.compile(
            r"\b(?:other\s+information|revision\s+date|abbreviations|disclaimer|otra\s+información)\b",
            re.IGNORECASE,
        ),
    ),
]

# Generic intent patterns that map to multiple contiguous SDS sections
_MULTI_SECTION_INTENTS: list[tuple[list[str], re.Pattern]] = [
    # Broad "Safety Measures" -> Sections 4 (First Aid), 5 (Fire), 6 (Spill/Accidental Release), 7 (Handling/Storage), 8 (Exposure/PPE)
    (
        ["4", "5", "6", "7", "8"],
        re.compile(
            r"\b(?:safety\s+measures?|safety\s+precautions?|safety\s+guidelines?|safety\s+instructions?|"
            r"safety\s+information|general\s+safety|medidas\s+de\s+seguridad|medidas\s+de\s+protecci[oó]n|"
            r"precau[cç][oõó]es\s+de\s+seguran[cç]a|how\s+to\s+handle\s+safely|precautionary\s+measures)\b",
            re.IGNORECASE,
        ),
    ),
    # Emergency Response -> Sections 4 (First Aid), 5 (Fire), 6 (Spill/Release)
    (
        ["4", "5", "6"],
        re.compile(
            r"\b(?:emergency\s+response|emergency\s+measures|emergency\s+procedures|emergency\s+action|"
            r"medidas\s+de\s+emergencia|respuestas\s+de\s+emergencia)\b",
            re.IGNORECASE,
        ),
    ),
    # Environmental & Disposal -> Sections 12 (Ecological) and 13 (Disposal)
    (
        ["12", "13"],
        re.compile(
            r"\b(?:environmental\s+and\s+disposal|environment\s+and\s+waste|eco\s+and\s+disposal)\b",
            re.IGNORECASE,
        ),
    ),
]


class SectionDetector:
    """Detects target GHS SDS section numbers from user queries."""

    def detect_section(self, query: str) -> str | None:
        """Return the primary target SDS section number string (e.g. '4') or None."""
        sections = self.detect_sections(query)
        return sections[0] if sections else None

    def detect_sections(self, query: str) -> list[str]:
        """Return ALL matching SDS section number strings for a query in order."""
        q = query.strip()
        matched: list[str] = []
        seen: set[str] = set()

        # 1. Check multi-section compound intents first (e.g. "safety measures" -> 4, 5, 6, 7, 8)
        for sec_list, pattern in _MULTI_SECTION_INTENTS:
            if pattern.search(q):
                for s in sec_list:
                    if s not in seen:
                        seen.add(s)
                        matched.append(s)

        # 2. Check individual section patterns
        for section_num, pattern in _SECTION_PATTERNS:
            if pattern.search(q):
                if section_num not in seen:
                    seen.add(section_num)
                    matched.append(section_num)

        if matched:
            logger.info("SECTION DETECTOR | query='%s' -> target_sections=%s", q[:50], matched)
        return matched
