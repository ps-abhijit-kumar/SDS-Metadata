"""Deterministic SDS section detector for multilingual section-aware RAG retrieval."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Multilingual keywords & regex patterns for standard GHS Sections 1-16
_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Section 1: Identification of the substance/mixture and of the company/undertaking
    (
        "1",
        re.compile(
            r"\b(?:product\s+identifier|chemical\s+identification|substance\s+identification|product\s+identification|"
            r"emergency\s+telephone|emergency\s+contact|intended\s+use|uses\s+advised\s+against|"
            r"identificación\s+del\s+producto|identificação\s+do\s+produto|identificação\s+da\s+empresa|"
            r"section\s+1\b|secci[oó]n\s+1\b|sec[cç][aã]o\s+1\b)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 2: Hazard Identification & Precautionary Statements
    (
        "2",
        re.compile(
            r"\b(?:hazard|hazards|danger|warning|hazard\s+statements|identificación\s+de\s+los\s+peligros|"
            r"peligros|peligrosidad|risks|risk\s+phrases|pictogram|ghs\s+classification|classificação\s+de\s+perigo|"
            r"precaution|precautions|precautionary|precautionary\s+statements|p-phrases|p\s+phrases|"
            r"frases\s+de\s+precauci[oó]n|recomenda[cç][oõó]es\s+de\s+prud[eê]ncia|cuidados\s+de\s+seguran[cç]a|"
            r"hazard\s+precautions|safety\s+precautions|statement\s+of\s+hazard)\b",
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
            r"safe\s+handling|storage\s+conditions|incompatibilities|conditions\s+for\s+safe\s+storage|manuseio|"
            r"armazenamento|storage\s+precautions|handling\s+precautions|precau[cç][oõó]es\s+de\s+manuseio|"
            r"precau[cç][oõó]es\s+de\s+armazenamento|condi[cç][oõó]es\s+de\s+armazenamento)\b",
            re.IGNORECASE,
        ),
    ),
    # Section 8: Exposure Controls / Personal Protection (PPE)
    (
        "8",
        re.compile(
            r"\b(?:exposure\s+controls?|ppe|protective\s+equipment|personal\s+protection|controles\s+de\s+exposición|"
            r"controles\s+de\s+exposi[cç][aã]o|protección\s+personal|equipo\s+de\s+protección|mascarilla|guantes|protect|"
            r"protection|protect\s+myself|protective|respiratory\s+protection|gloves|safety\s+goggles|safety\s+glasses|"
            r"impervious\s+clothing|ventilation|occupational\s+exposure|exposure\s+limit|threshold\s+limit|"
            r"personal\s+precautions|protective\s+precautions)\b",
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
    # Section 16: Other Information (Abbreviations, Full forms, Revision Date)
    (
        "16",
        re.compile(
            r"\b(?:other\s+information|revision\s+date|abbreviations?|disclaimer|otra\s+información|outras\s+informa[cç][oõó]es|"
            r"full\s+form|full\s+name|sigla|siglas|abreviatura|abreviaturas|acronym|acronyms|sds\s+stands\s+for|"
            r"what\s+does\s+sds\s+mean|ficha\s+de\s+dados\s+de\s+seguran[cç]a)\b",
            re.IGNORECASE,
        ),
    ),
]

# Generic intent patterns that map to multiple contiguous SDS sections
_MULTI_SECTION_INTENTS: list[tuple[list[str], re.Pattern]] = [
    # Broad "Safety Measures / Precautions" -> Sections 2 (Hazard Precautions), 4 (First Aid), 7 (Handling/Storage), 8 (Exposure/PPE)
    (
        ["2", "4", "7", "8"],
        re.compile(
            r"\b(?:precautions?|precautionary(?:\s+statements?)?|safety\s+precautions?|safety\s+measures?|"
            r"safety\s+guidelines?|safety\s+instructions?|safety\s+information|general\s+safety|"
            r"medidas\s+de\s+seguridad|medidas\s+de\s+protecci[oó]n|precau[cç][oõó]es(?:\s+de\s+seguran[cç]a)?|"
            r"recomenda[cç][oõó]es\s+de\s+prud[eê]ncia|how\s+to\s+handle\s+safely|precautionary\s+measures)\b",
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
        q_lower = q.lower()

        # Pre-guard 1: Conversational noise / banter words (e.g. "company name abhijeet bhai what is this")
        if re.search(r"\b(?:bhai|bro|dude|abhijeet|abhijit|random|lol|wtf|haha)\b", q_lower):
            logger.info("SECTION DETECTOR | Ignored noisy query='%s'", q[:50])
            return []

        # Pre-guard 2: Non-SDS outside-world / personal identity queries
        if re.search(r"\b(?:meaning\s+of\s+the\s+word|definition\s+of\s+the\s+word|my\s+name|who\s+am\s+i|fifa|world\s+cup|capital\s+of)\b", q_lower):
            logger.info("SECTION DETECTOR | Ignored non-SDS query='%s'", q[:50])
            return []

        # Pre-guard 3: Unstructured noisy phrases with 'what is this' attached to random words
        if "what is this" in q_lower and len(q.split()) > 4 and not any(k in q_lower for k in ["product", "document", "sds", "chemical", "substance"]):
            logger.info("SECTION DETECTOR | Ignored unstructured query='%s'", q[:50])
            return []

        matched: list[str] = []
        seen: set[str] = set()

        # 1. Check multi-section compound intents first (e.g. "precautions", "safety measures" -> 2, 4, 7, 8)
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
