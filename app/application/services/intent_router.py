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
            r"\b(?:what|which|in\s+what|in\s+which|tell\s+me\s+the|identify\s+the|is\s+this\s+(?:document|sds|pdf)?\s*in)\b"
            r".*?\b(?:language|idioma|langue|sprache|spanish|english|german|french|portuguese|italian|dutch)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MetadataIntent.LANGUAGE,
        re.compile(
            r"\b(?:what|which|can\s+you\s+identify\s+the)\s+language\b",
            re.IGNORECASE,
        ),
    ),
    (
        MetadataIntent.LANGUAGE,
        re.compile(
            r"\bwhat\s+language\s+is\s+(?:this|the)\s+(?:document|sds|file|pdf)\s+(?:written|in)\b",
            re.IGNORECASE,
        ),
    ),
    # Product Name intent (Excludes questions asking about chemical composition, ingredients, or multi-part questions)
    (
        MetadataIntent.PRODUCT_NAME,
        re.compile(
            r"\b(?:what\s+(?:is|called)|identify|tell\s+me)\b(?![^?]*\b(?:chemical|composition|ingredient|hazard|effect|first\s*aid|protection|storage|handling)\b)"
            r".*?\b(?:product\s+name|trade\s+name|commercial\s+name|product\s+called|this\s+product|product\s+is\s+this)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MetadataIntent.PRODUCT_NAME,
        re.compile(
            r"^\s*what\s+(?:product\s+is\s+this|(?:is\s+the\s+)?product(?:\s+name|\s+is\s+this|\s+called)?)\??\s*$",
            re.IGNORECASE,
        ),
    ),
    # Company / Manufacturer intent
    (
        MetadataIntent.COMPANY_NAME,
        re.compile(
            r"\b(?:who|what|which\s+company|identify)\b.*?\b(?:manufactur\w*|suppli\w*|company|produc(?:er|ers|es|ed|ing)|makes?|made|responsible)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MetadataIntent.COMPANY_NAME,
        re.compile(
            r"\b(?:manufacturer|supplier|producer|company\s+name|manufacturer\s+name|who\s+makes\s+this)\b",
            re.IGNORECASE,
        ),
    ),
    # Jurisdiction intent
    (
        MetadataIntent.JURISDICTION,
        re.compile(
            r"\b(?:what|which|is\s+this\s+under)\b.*?\b(?:jurisdiction|regulatory|framework|reach|clp|osha|whmis|regulation|regulations)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MetadataIntent.JURISDICTION,
        re.compile(
            r"^\s*what\s+(?:regulations\s+does\s+this\s+document\s+follow|(?:is\s+the\s+)?jurisdiction)\??\s*$",
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

        # Guard against composition/ingredient or multi-intent RAG questions being misclassified as a single metadata field
        rag_topic_count = sum(
            1 for kw in [
                "hazard", "risk", "first aid", "exposure", "storage", "handling",
                "protection", "ppe", "composition", "ingredient", "fire", "spill", "disposal"
            ] if kw in q_lower
        )
        if rag_topic_count >= 1 and any(meta_kw in q_lower for meta_kw in ["product", "manufacturer", "company", "who makes"]):
            # Multi-part question combining metadata + RAG content -> defer to full RAG execution
            return MetadataIntent.NONE

        if re.search(r"\b(?:chemical|composition|ingredient|component|element|substance|formula)\b", q, re.IGNORECASE):
            return MetadataIntent.NONE

        # If question contains multiple distinct commas / 'and' joining multiple fields, defer to RAG
        if q_lower.count(",") >= 2 or (" and " in q_lower and q_lower.count("what") + q_lower.count("who") > 1):
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
