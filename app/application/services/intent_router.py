"""Deterministic metadata intent router for ultra-fast non-LLM chat responses."""

from __future__ import annotations

import logging
import re
from enum import Enum

from app.domain.entities.document import Document

logger = logging.getLogger(__name__)


class MetadataIntent(str, Enum):
    PRODUCT_NAME = "PRODUCT_NAME"
    COMPANY_NAME = "COMPANY_NAME"
    LANGUAGE = "LANGUAGE"
    JURISDICTION = "JURISDICTION"
    NONE = "NONE"


_INTENT_PATTERNS: list[tuple[MetadataIntent, re.Pattern]] = [
    # Language intent
    (
        MetadataIntent.LANGUAGE,
        re.compile(
            r"^\s*(?:what\s+language(?:\s+is\s+this)?|what\s+is\s+the\s+language|identify\s+the\s+language|language)\??\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        MetadataIntent.LANGUAGE,
        re.compile(
            r"\b(?:what\s+language\s+is\s+(?:this|the)\s+(?:document|sds|file|pdf)\s+(?:written\s+in|in)|"
            r"what\s+is\s+the\s+language\s+of\s+(?:this|the)\s+(?:document|sds|file|pdf)|"
            r"in\s+what\s+language\s+is\s+(?:this|the)\s+(?:document|sds|file|pdf)\s+written)\b",
            re.IGNORECASE,
        ),
    ),
    # Product Name intent
    (
        MetadataIntent.PRODUCT_NAME,
        re.compile(
            r"^\s*(?:what\s+(?:is\s+the\s+)?product(?:\s+name)?|what\s+product\s+is\s+this|what\s+is\s+this\s+product(?:\s+called)?|product\s+name|trade\s+name|commercial\s+name)\??\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        MetadataIntent.PRODUCT_NAME,
        re.compile(
            r"\b(?:what\s+is\s+the\s+(?:product\s+name|trade\s+name|commercial\s+name)|"
            r"what\s+is\s+this\s+product\s+called|tell\s+me\s+the\s+product\s+name)\b",
            re.IGNORECASE,
        ),
    ),
    # Company / Manufacturer intent
    (
        MetadataIntent.COMPANY_NAME,
        re.compile(
            r"^\s*(?:what\s+(?:is\s+the\s+)?company(?:\s+name)?|what\s+is\s+company|who\s+(?:is\s+the\s+)?(?:manufacturer|supplier|producer|company)|"
            r"who\s+(?:manufactures?|makes?|produces?)\s+(?:this|the)(?:\s+product)?|who\s+makes\s+this|manufacturer(?:\s+name)?|supplier(?:\s+name)?|company\s+name)\??\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        MetadataIntent.COMPANY_NAME,
        re.compile(
            r"\b(?:who\s+(?:is\s+the\s+)?(?:manufacturer|supplier|producer|company)|"
            r"who\s+(?:manufactures?|makes?|produces?)\s+(?:this|the)\s+(?:product|chemical|sds)|"
            r"what\s+is\s+the\s+(?:manufacturer|supplier|producer|company)\s+name)\b",
            re.IGNORECASE,
        ),
    ),
    # Jurisdiction intent
    (
        MetadataIntent.JURISDICTION,
        re.compile(
            r"^\s*(?:what\s+is\s+the\s+jurisdiction|what\s+jurisdiction|what\s+regulations?\s+does\s+this\s+(?:document|sds|file|pdf)\s+follow|jurisdiction)\??\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        MetadataIntent.JURISDICTION,
        re.compile(
            r"\b(?:what\s+(?:regulatory\s+framework|jurisdiction|regulation)\s+does\s+(?:this|the)\s+(?:sds|document|file|pdf)\s+follow|"
            r"what\s+is\s+the\s+regulatory\s+jurisdiction)\b",
            re.IGNORECASE,
        ),
    ),
]


class IntentRouter:
    """Detects if a user question targets an explicit metadata field."""

    def detect_intent(self, question: str) -> MetadataIntent:
        """Map user question to metadata intent or NONE."""
        q = question.strip()
        q_lower = q.lower()

        # Pre-guard 1: Language definition/meaning questions (e.g. "what is the meaning of the word company")
        if re.search(r"\b(?:meaning\s+of|definition\s+of|define|what\s+does\s+(?:\w+\s+)?mean|explain\s+the\s+word|what\s+is\s+the\s+meaning)\b", q_lower):
            return MetadataIntent.NONE

        # Pre-guard 2: Personal identity / out-of-bounds user questions (e.g. "what is my name")
        if re.search(r"\b(?:my\s+name|who\s+am\s+i|your\s+name|who\s+are\s+you)\b", q_lower):
            return MetadataIntent.NONE

        # Pre-guard 3: Guard against composition/ingredient or multi-intent RAG questions being misclassified as metadata
        rag_topic_count = sum(
            1 for kw in [
                "hazard", "risk", "first aid", "exposure", "storage", "handling",
                "protection", "ppe", "composition", "ingredient", "fire", "spill", "disposal"
            ] if kw in q_lower
        )
        if rag_topic_count >= 1 and any(meta_kw in q_lower for meta_kw in ["product", "manufacturer", "company", "who makes"]):
            return MetadataIntent.NONE

        if re.search(r"\b(?:chemical|composition|ingredient|component|element|substance|formula)\b", q, re.IGNORECASE):
            return MetadataIntent.NONE

        # Pre-guard 4: If question contains multiple distinct clauses / 'and' joining multiple fields, defer to RAG
        if q_lower.count(",") >= 2 or (" and " in q_lower and q_lower.count("what") + q_lower.count("who") > 1):
            return MetadataIntent.NONE

        # Pre-guard 5: Noisy queries with excessive non-question tokens (e.g. "company name abhijeet bhai what is this")
        if len(q.split()) > 7 and not any(q_lower.startswith(p) for p in ["what is the", "who is the", "which company", "what language", "what product", "what jurisdiction", "what are the regulations", "tell me the"]):
            return MetadataIntent.NONE

        for intent, pattern in _INTENT_PATTERNS:
            if pattern.search(q):
                logger.info(
                    "CHAT ROUTER | intent=%s | source=DOCUMENT_METADATA | llm_called=false | retrieval_ms=0",
                    intent.value,
                )
                return intent

        return MetadataIntent.NONE

    def format_metadata_response(
        self,
        intent: MetadataIntent,
        document: Document | None,
        all_documents: list[Document] | None = None,
        scope: str = "single",
    ) -> tuple[str, bool, list[dict]]:
        """Construct instant non-LLM answer from document metadata record.

        Returns:
            (answer_str, is_grounded, sources_list)
        """
        if scope == "all" and all_documents:
            return self._format_multi_doc_response(intent, all_documents)

        if not document or not document.metadata:
            return "Information not available in the uploaded file.", False, []

        meta = document.metadata
        val = None

        if intent == MetadataIntent.LANGUAGE:
            val = meta.language
            field_name = "language"
            prefix = "The document is written in "
        elif intent == MetadataIntent.PRODUCT_NAME:
            val = meta.product_name
            field_name = "product_name"
            prefix = "Product Name: "
        elif intent == MetadataIntent.COMPANY_NAME:
            val = meta.company_name
            field_name = "company_name"
            prefix = "The manufacturer is "
        elif intent == MetadataIntent.JURISDICTION:
            val = meta.jurisdiction
            field_name = "jurisdiction"
            prefix = "Jurisdiction: "
        else:
            return "Information not available in the uploaded file.", False, []

        if not val or val.lower() in ("unknown", "n/a", "none", "not available in document"):
            return "Information not available in the uploaded file.", False, []

        answer = f"{prefix}{val}." if intent in (MetadataIntent.LANGUAGE, MetadataIntent.COMPANY_NAME) else f"{prefix}{val}"
        sources = [
            {
                "document": document.filename,
                "source_type": "document_metadata",
                "page": 1,
                "section": "Metadata",
                "section_title": "Extracted document metadata",
            }
        ]
        return answer, True, sources

    def _format_multi_doc_response(
        self,
        intent: MetadataIntent,
        all_documents: list[Document],
    ) -> tuple[str, bool, list[dict]]:
        """Format multi-document metadata query response on canonical unique documents."""
        matches = []
        sources = []
        seen_keys = set()

        for doc in all_documents:
            if not doc.metadata:
                continue

            # Deduplicate by canonical identity (file_hash or filename)
            key = doc.file_hash if doc.file_hash else doc.filename
            if key in seen_keys:
                continue
            seen_keys.add(key)

            meta = doc.metadata
            val = None
            if intent == MetadataIntent.LANGUAGE:
                val = meta.language
            elif intent == MetadataIntent.PRODUCT_NAME:
                val = meta.product_name
            elif intent == MetadataIntent.COMPANY_NAME:
                val = meta.company_name
            elif intent == MetadataIntent.JURISDICTION:
                val = meta.jurisdiction

            if val:
                matches.append(f"- **{doc.filename}**: {val}")
                sources.append({
                    "document": doc.filename,
                    "source_type": "document_metadata",
                    "page": 1,
                    "section": "Metadata",
                    "section_title": "Extracted document metadata",
                })

        if not matches:
            return "Information not available in the uploaded documents.", False, []

        intent_labels = {
            MetadataIntent.LANGUAGE: "Languages of uploaded documents:",
            MetadataIntent.PRODUCT_NAME: "Product names of uploaded documents:",
            MetadataIntent.COMPANY_NAME: "Manufacturers/Companies of uploaded documents:",
            MetadataIntent.JURISDICTION: "Jurisdictions of uploaded documents:",
        }

        answer = intent_labels.get(intent, "Uploaded document metadata:") + "\n" + "\n".join(matches)
        return answer, True, sources
