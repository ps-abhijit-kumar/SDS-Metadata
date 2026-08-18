"""Grounding service for verifying context sufficiency and anti-hallucination."""

from __future__ import annotations

import logging
from app.application.services.retrieval_service import RetrievedChunk
from app.infrastructure.configuration.settings import Settings

logger = logging.getLogger(__name__)


class GroundingService:
    """Enforces strict document grounding and anti-hallucination checks."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def verify_grounding(
        self,
        chunks: list[RetrievedChunk],
        query: str | None = None,
        scope: str = "single",
        is_overview: bool = False,
    ) -> tuple[bool, str]:
        """Verify if retrieved chunks provide sufficient evidence for the user question.

        Returns:
            (is_grounded: bool, fallback_message: str)
        """
        fallback = (
            self._settings.multi_doc_fallback_response
            if scope == "all"
            else self._settings.fallback_response
        )

        if not chunks:
            logger.info("Grounding check failed: No relevant chunks retrieved above threshold.")
            return False, fallback

        # If explicit overview query and valid chunks exist, evidence is sufficient
        if is_overview:
            return True, ""

        # Check chunk quality and relevance scores
        best_score = min(c.score for c in chunks)

        # Check if any chunk has an explicit section match with an acceptable score
        has_section_match = any(
            str(c.section).strip() not in ("0", "N/A", "None", "")
            and c.score <= 1.35
            for c in chunks
        )

        distance_threshold = getattr(self._settings, "rag_distance_threshold", 1.20)

        # Check lexical keyword evidence if query is provided
        has_lexical_evidence = False
        if query:
            import re
            query_words = set(re.findall(r"\w{3,}", query.lower()))
            stop_words = {
                "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
                "the", "and", "is", "are", "was", "were", "this", "that", "these", "those",
                "for", "with", "about", "from", "into", "during", "including", "until",
                "against", "among", "throughout", "despite", "towards", "upon", "concerning",
                "to", "in", "for", "on", "by", "at", "tell", "give", "can", "you", "please",
                "should", "does", "have", "been", "listed", "mentioned", "contain", "contains",
            }
            content_words = query_words - stop_words
            if content_words:
                for c in chunks:
                    chunk_text_lower = c.text.lower()
                    chunk_title_lower = c.section_title.lower()
                    if any(w in chunk_text_lower or w in chunk_title_lower for w in content_words):
                        has_lexical_evidence = True
                        break
            else:
                has_lexical_evidence = True

        # Evidence verification:
        # 1. High-confidence pure semantic match
        if best_score <= 0.65:
            logger.info("Grounding verified: High-confidence semantic match (score=%.3f <= 0.65)", best_score)
            return True, ""

        # 2. Section match with valid score
        if has_section_match and best_score <= 1.35:
            logger.info("Grounding verified: Section match with valid score (score=%.3f <= 1.35)", best_score)
            return True, ""

        # 3. Lexical evidence present with valid semantic distance
        if has_lexical_evidence and best_score <= 1.25:
            logger.info("Grounding verified: Semantic match with lexical evidence (score=%.3f <= 1.25)", best_score)
            return True, ""

        logger.info(
            "Grounding check failed: Insufficient evidence (best_score=%.3f, threshold=%.3f, lexical=%s, section=%s)",
            best_score,
            distance_threshold,
            has_lexical_evidence,
            has_section_match,
        )
        return False, fallback
