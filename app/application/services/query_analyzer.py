"""Query analyzer service for semantic SDS query understanding, multi-intent decomposition, and document overview detection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysisResult:
    original_query: str
    is_overview: bool = False
    is_multi_intent: bool = False
    sub_queries: list[str] = field(default_factory=list)
    semantic_expansions: list[str] = field(default_factory=list)
    requested_metadata_fields: list[str] = field(default_factory=list)


# Keywords indicating document overview / summary intent
_OVERVIEW_PATTERNS = [
    re.compile(
        r"\b(?:tell\s+me\s+about\s+(?:the|this)\s+(?:document|sds|file|pdf|product)|"
        r"give\s+me\s+(?:an?\s+)?(?:overall\s+)?(?:overview|summary)\s+of\s+(?:this|the|all)?\s*(?:uploaded\s+)?(?:document|sds|file|pdf|product)?|"
        r"summarize\s+(?:the\s+)?(?:safety\s+information|safety\s+measures|document|file|sds|pdf|product|this)|"
        r"summary\s+of\s+(?:the|this)\s+(?:document|sds|file|pdf|product|safety\s+information)|"
        r"what\s+is\s+this\s+(?:document|sds|file|pdf)\s+about|"
        r"what\s+should\s+i\s+know\s+about\s+this\s+(?:product|document|sds|file|pdf)|"
        r"give\s+me\s+(?:overall\s+)?information\s+about\s+(?:the|this)\s+(?:uploaded|document|sds|file|pdf|product)|"
        r"overall\s+information\s+about\s+(?:the|this)\s+(?:document|sds|file|pdf)|"
        r"give\s+me\s+a\s+summary\s+of\s+the\s+uploaded\s+document)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:tell\s+me\s+about\s+this|information\s+about\s+this|overview|summary|give\s+me\s+an\s+overview)\s*(?:file|document|sds|pdf)?\??\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:give\s+me\s+information|important\s+information)\s+about\s+(?:this|the)\s+(?:document|file|sds|pdf)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*tell\s+me\s+about\s+(?:the|this)\s+(?:file|document|sds|pdf|product)\??\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:overview|summary|summarize)\??\s*$",
        re.IGNORECASE,
    ),
]

# Semantic term mappings for natural language expansion
_EXPANSION_MAP = {
    "safety": ["safety measures precautions", "personal protective equipment PPE", "first aid emergency measures", "handling and storage safety"],
    "precautions": ["safety precautions", "personal protective equipment", "exposure controls handling", "hazard safety precautions"],
    "exposure": ["first aid measures exposure", "inhalation skin eye contact ingestion", "exposure controls personal protection"],
    "exposed": ["first aid measures exposure", "inhalation skin eye contact ingestion", "emergency treatment"],
    "handling": ["handling and storage", "precautions for safe handling", "personal protective equipment"],
    "stored": ["handling and storage", "conditions for safe storage", "incompatibilities"],
    "storage": ["handling and storage", "conditions for safe storage", "incompatibilities"],
    "hazards": ["hazard identification", "hazard statements", "risk phrases", "dangers"],
    "dangerous": ["hazard identification", "hazard statements", "classification danger"],
    "emergency": ["emergency measures", "first aid measures", "accidental release measures", "fire fighting"],
    "contact": ["first aid measures", "skin contact", "eye contact", "inhalation", "ingestion"],
    "touch": ["first aid measures", "skin contact", "eye contact"],
    "breathe": ["first aid measures", "inhalation", "respiratory protection"],
    "swallow": ["first aid measures", "ingestion", "swallowed"],
    "composition": ["composition information on ingredients", "chemical components CAS formula molecular weight"],
    "chemical": ["chemical composition ingredients", "CAS number formula molecular weight"],
    "ppe": ["personal protective equipment", "exposure controls respiratory protection gloves safety goggles"],
}


class QueryAnalyzer:
    """Analyzes user queries to detect multi-intent, overview requests, and semantic expansions."""

    def analyze(self, query: str) -> QueryAnalysisResult:
        q_strip = query.strip()
        result = QueryAnalysisResult(original_query=q_strip)

        # 1. Overview Check
        for pat in _OVERVIEW_PATTERNS:
            if pat.search(q_strip):
                result.is_overview = True
                result.sub_queries = [
                    "Section 1 Identification product name company manufacturer",
                    "Section 2 Hazards identification danger risk phrases",
                    "Section 3 Composition ingredients CAS formula",
                    "Section 4 First aid measures emergency treatment",
                    "Section 7 Handling and storage precautions",
                    "Section 8 Exposure controls personal protection PPE",
                    "Section 15 Regulatory information jurisdiction",
                ]
                logger.info("QUERY ANALYZER | Overview intent detected for query='%s'", q_strip[:50])
                return result

        # 2. Multi-Intent Decomposition
        sub_intents = self._split_multi_intent(q_strip)
        if len(sub_intents) > 1:
            result.is_multi_intent = True
            result.sub_queries = sub_intents
            logger.info("QUERY ANALYZER | Multi-intent detected (%d sub-queries: %s) for query='%s'", len(sub_intents), sub_intents, q_strip[:50])
        else:
            result.sub_queries = [q_strip]

        # 3. Semantic Query Expansions
        expansions = self._generate_expansions(q_strip)
        result.semantic_expansions = expansions

        return result

    def _split_multi_intent(self, query: str) -> list[str]:
        """Decompose a multi-part question into distinct sub-queries."""
        q_lower = query.lower()

        # Check if question has multiple distinct topic clauses
        topics = []
        
        # 1. Topic indicators
        if re.search(r"\b(?:safety\s+measures?|safety\s+precautions?|safe\s+handling|safety\s+guidelines?|safety\s+info)\b", q_lower):
            topics.append("safety measures precautions first aid handling PPE")
        if re.search(r"\b(?:chemical\s+composition|composition|ingredients?|components?|formula|molecular\s+weight|cas\s+number)\b", q_lower):
            topics.append("chemical composition information on ingredients")
        if re.search(r"\b(?:first\s*aid|emergency\s+treatment|eye\s+contact|skin\s+contact|inhalation|ingestion|swallowed)\b", q_lower):
            if not any("first aid" in t for t in topics):
                topics.append("first aid measures emergency treatment")
        if re.search(r"\b(?:fire\s*fighting|firefighting|extinguishing|flammable)\b", q_lower):
            topics.append("fire fighting measures extinguishing media")
        if re.search(r"\b(?:accidental\s+release|spill|spill\s+cleanup|leak)\b", q_lower):
            topics.append("accidental release measures spill cleanup")
        if re.search(r"\b(?:storage|stored|handling|keep\s+conditions)\b", q_lower):
            if not any("handling" in t for t in topics):
                topics.append("handling and storage conditions")
        if re.search(r"\b(?:ppe|protective\s+equipment|personal\s+protection|respiratory\s+protection|gloves|goggles)\b", q_lower):
            if not any("PPE" in t for t in topics):
                topics.append("exposure controls personal protection PPE")
        if re.search(r"\b(?:hazards?|danger|risk\s+phrases|pictogram)\b", q_lower):
            topics.append("hazard identification risk phrases")
        if re.search(r"\b(?:toxicology|toxicity|health\s+effects)\b", q_lower):
            topics.append("toxicological information health effects")
        if re.search(r"\b(?:ecological|environmental|ecotoxicity)\b", q_lower):
            topics.append("ecological information environmental impact")
        if re.search(r"\b(?:disposal|waste|waste\s+treatment)\b", q_lower):
            topics.append("disposal considerations waste treatment")
        if re.search(r"\b(?:transport|shipping|un\s+number)\b", q_lower):
            topics.append("transport information shipping")
        if re.search(r"\b(?:regulatory|regulation|reach|clp|osha|whmis)\b", q_lower):
            topics.append("regulatory information compliance")

        # If 2 or more distinct topics were explicitly identified in a multi-clause question:
        if len(topics) >= 2 and (
            "," in query
            or " and " in q_lower
            or " also " in q_lower
            or " as well as " in q_lower
            or " along with " in q_lower
            or "?" in query[:-1]
        ):
            return topics

        # Clause-based splitting
        delimiters = [
            r"\s+and\s+also\s+",
            r"\s+as\s+well\s+as\s+",
            r"\s+along\s+with\s+",
            r"\s+and\s+(?=give|tell|what|who|how|explain|describe)\b",
            r",\s*and\s+",
            r";\s*",
        ]
        combined_delim = "|".join(delimiters)
        parts = re.split(combined_delim, query, flags=re.IGNORECASE)
        valid_parts = [p.strip() for p in parts if len(p.strip()) >= 5]

        if len(valid_parts) >= 2:
            return valid_parts

        # Standard split on ' and ' / commas if multiple question triggers exist
        fallback_parts = re.split(r",\s*|\s+and\s+", query, flags=re.IGNORECASE)
        valid_fb = [p.strip() for p in fallback_parts if len(p.strip()) >= 4]
        if len(valid_fb) >= 2 and any(kw in q_lower for kw in ["what", "give", "tell", "who", "how", "list"]):
            return valid_fb

        return [query]

    def _generate_expansions(self, query: str) -> list[str]:
        """Generate semantic search expansion terms for natural language queries."""
        q_words = set(re.findall(r"\w+", query.lower()))
        expansions = []
        for word, expanded_terms in _EXPANSION_MAP.items():
            if word in q_words or any(word in w for w in q_words):
                expansions.extend(expanded_terms)

        # Deduplicate while maintaining order
        seen = set()
        unique_exp = []
        for term in expansions:
            if term not in seen:
                seen.add(term)
                unique_exp.append(term)
        return unique_exp
