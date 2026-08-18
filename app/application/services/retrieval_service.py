"""Retrieval service implementing hybrid, section-aware SDS chunk retrieval and ranking."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.application.services.query_analyzer import QueryAnalyzer
from app.application.services.section_detector import SectionDetector
from app.infrastructure.configuration.settings import Settings
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    document_id: str
    filename: str
    page: int
    section: str
    section_title: str
    score: float  # Cosine distance / hybrid distance score (lower is more relevant)


class RetrievalService:
    """Retrieves relevant document chunks using semantic query analysis, multi-intent decomposition, section awareness & hybrid ranking."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        settings: Settings,
        section_detector: SectionDetector | None = None,
        query_analyzer: QueryAnalyzer | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._settings = settings
        self._section_detector = section_detector or SectionDetector()
        self._query_analyzer = query_analyzer or QueryAnalyzer()

    def retrieve(
        self,
        query: str,
        document_id: str = "all",
        k: int | None = None,
        max_distance: float | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid semantic chunk retrieval with multi-intent decomposition, per-intent section awareness, and hybrid ranking."""
        top_k = k or self._settings.rag_top_k
        distance_limit = max_distance if max_distance is not None else self._settings.rag_distance_threshold

        analysis = self._query_analyzer.analyze(query)
        target_sections = self._section_detector.detect_sections(query)

        seen_chunk_keys = set()
        raw_results: list[tuple[any, float]] = []

        if analysis.is_overview:
            # ── 1. Overview Path: Multi-Section Representative Retrieval ───────────
            overview_queries = [
                query,
                "Section 1 Product Identification company manufacturer supplier",
                "Section 2 Hazards Identification danger classification risk phrases",
                "Section 3 Composition ingredients CAS formula molecular weight",
                "Section 4 First aid measures emergency treatment",
                "Section 7 Handling and storage precautions safe conditions",
                "Section 8 Exposure controls personal protection PPE equipment",
                "Section 15 Regulatory information jurisdiction framework",
            ]
            for o_query in overview_queries:
                res = self._vector_store.similarity_search_with_score(
                    query=o_query,
                    document_id=document_id,
                    k=15,
                )
                for doc, score in res:
                    meta = doc.metadata or {}
                    chunk_key = (meta.get("document_id"), meta.get("chunk_index"), doc.page_content[:60])
                    if chunk_key not in seen_chunk_keys:
                        seen_chunk_keys.add(chunk_key)
                        raw_results.append((doc, score))

            overview_sections = ["1", "2", "3", "4", "7", "8", "15"]
            for sec_str in overview_sections:
                sec_res = self._vector_store.similarity_search_with_score(
                    query=query,
                    document_id=document_id,
                    k=8,
                    where_filter={"section": sec_str},
                )
                for doc, score in sec_res:
                    meta = doc.metadata or {}
                    chunk_key = (meta.get("document_id"), meta.get("chunk_index"), doc.page_content[:60])
                    if chunk_key not in seen_chunk_keys:
                        seen_chunk_keys.add(chunk_key)
                        raw_results.append((doc, score))

            top_k = max(top_k, 8)

        elif analysis.is_multi_intent and len(analysis.sub_queries) > 1:
            # ── 2. Multi-Intent Path: Independent Per-Subquery Retrieval ───────────
            # Execute retrieval for the original query + each subquery with its own specific target sections
            sub_results: list[list[tuple[any, float]]] = []
            
            for sub_q in analysis.sub_queries:
                sub_seen = set()
                sub_list: list[tuple[any, float]] = []
                sub_target_sections = self._section_detector.detect_sections(sub_q)

                # Search subquery text
                res = self._vector_store.similarity_search_with_score(
                    query=sub_q,
                    document_id=document_id,
                    k=20,
                )
                for doc, score in res:
                    meta = doc.metadata or {}
                    chunk_key = (meta.get("document_id"), meta.get("chunk_index"), doc.page_content[:60])
                    if chunk_key not in sub_seen:
                        sub_seen.add(chunk_key)
                        sub_list.append((doc, score))
                        if chunk_key not in seen_chunk_keys:
                            seen_chunk_keys.add(chunk_key)
                            raw_results.append((doc, score))

                # Also search with explicit section filter for this subquery's specific target sections
                for sec_str in sub_target_sections:
                    sec_res = self._vector_store.similarity_search_with_score(
                        query=sub_q,
                        document_id=document_id,
                        k=15,
                        where_filter={"section": str(sec_str)},
                    )
                    for doc, score in sec_res:
                        meta = doc.metadata or {}
                        chunk_key = (meta.get("document_id"), meta.get("chunk_index"), doc.page_content[:60])
                        if chunk_key not in sub_seen:
                            sub_seen.add(chunk_key)
                            sub_list.append((doc, score))
                            if chunk_key not in seen_chunk_keys:
                                seen_chunk_keys.add(chunk_key)
                                raw_results.append((doc, score))

                sub_results.append(sub_list)

            # Search general expansions
            if analysis.semantic_expansions:
                for exp in analysis.semantic_expansions:
                    res = self._vector_store.similarity_search_with_score(
                        query=exp,
                        document_id=document_id,
                        k=15,
                    )
                    for doc, score in res:
                        meta = doc.metadata or {}
                        chunk_key = (meta.get("document_id"), meta.get("chunk_index"), doc.page_content[:60])
                        if chunk_key not in seen_chunk_keys:
                            seen_chunk_keys.add(chunk_key)
                            raw_results.append((doc, score))

            top_k = max(top_k, len(analysis.sub_queries) * 3)

        else:
            # ── 3. Single Intent Path ──────────────────────────────────────────────
            search_queries = [query]
            if analysis.semantic_expansions:
                search_queries.extend(analysis.semantic_expansions)

            candidate_k = max(top_k * 4, 25)
            for q_var in search_queries:
                if not q_var or not q_var.strip():
                    continue
                res = self._vector_store.similarity_search_with_score(
                    query=q_var,
                    document_id=document_id,
                    k=candidate_k,
                )
                for doc, score in res:
                    meta = doc.metadata or {}
                    chunk_key = (meta.get("document_id"), meta.get("chunk_index"), doc.page_content[:60])
                    if chunk_key not in seen_chunk_keys:
                        seen_chunk_keys.add(chunk_key)
                        raw_results.append((doc, score))

            if target_sections:
                for sec_num in target_sections:
                    sec_res = self._vector_store.similarity_search_with_score(
                        query=query,
                        document_id=document_id,
                        k=candidate_k,
                        where_filter={"section": str(sec_num)},
                    )
                    for doc, score in sec_res:
                        meta = doc.metadata or {}
                        chunk_key = (meta.get("document_id"), meta.get("chunk_index"), doc.page_content[:60])
                        if chunk_key not in seen_chunk_keys:
                            seen_chunk_keys.add(chunk_key)
                            raw_results.append((doc, score))

        if not raw_results:
            logger.info("RETRIEVAL QUERY | query='%s' | scope=%s | raw_results=0 | passed=0", query[:50], document_id)
            return []

        # Extract query keywords for lexical scoring
        query_words = set(re.findall(r"\w{3,}", query.lower()))

        ranked_candidates: list[tuple[float, RetrievedChunk]] = []

        for doc, score in raw_results:
            meta = doc.metadata or {}
            chunk_section = str(meta.get("section", "0")).strip()
            chunk_sec_num = str(meta.get("section_number", "0")).strip()
            chunk_title = str(meta.get("section_title", "")).strip().lower()
            text_lower = doc.page_content.lower()

            # Check Section Match against all detected target sections
            is_section_match = False
            if analysis.is_overview and (chunk_section in ("1", "2", "3", "4", "7", "8", "15") or chunk_sec_num in ("1", "2", "3", "4", "7", "8", "15")):
                is_section_match = True
            elif target_sections:
                for target_str in target_sections:
                    if (
                        chunk_section == target_str
                        or chunk_sec_num == target_str
                        or f"section {target_str}" in chunk_title
                        or f"sección {target_str}" in chunk_title
                        or f"secção {target_str}" in chunk_title
                        or f"abschnitt {target_str}" in chunk_title
                    ):
                        is_section_match = True
                        break

            # Compute Lexical Keyword Overlap
            matched_words = sum(1 for w in query_words if w in text_lower or w in chunk_title)
            lexical_ratio = (matched_words / max(len(query_words), 1)) if query_words else 0.0

            # Hybrid Distance & Ranking Score
            effective_distance = score
            if is_section_match:
                effective_distance = max(0.05, score - 0.35)

            hybrid_score = (0.7 * effective_distance) + (0.3 * (1.0 - lexical_ratio))

            chunk = RetrievedChunk(
                text=doc.page_content,
                document_id=meta.get("document_id", "unknown"),
                filename=meta.get("filename", "document.pdf"),
                page=int(meta.get("page", 1)),
                section=chunk_section if chunk_section != "0" else chunk_sec_num,
                section_title=meta.get("section_title", "General"),
                score=score,
            )

            # Calibrated relevance thresholding (Allows legitimate and cross-lingual SDS queries while strictly rejecting out-of-scope/outside-world questions)
            is_relevant = False
            if is_section_match and score <= 1.35:
                is_relevant = True
            elif score <= distance_limit:
                is_relevant = True
            elif hybrid_score <= 0.80 and score <= 1.30:
                is_relevant = True
            elif analysis.is_overview and score <= 1.30:
                is_relevant = True

            if is_relevant:
                ranked_candidates.append((hybrid_score, chunk))

        # Sort candidates by hybrid_score ascending (most relevant first)
        ranked_candidates.sort(key=lambda item: item[0])

        # If multi-intent query, ensure balanced representation across all detected sub-intents
        if analysis.is_multi_intent and len(analysis.sub_queries) > 1:
            final_chunks: list[RetrievedChunk] = []
            selected_keys = set()

            # Ensure at least 1-3 top chunks from each detected target section group
            for target_sec in target_sections:
                sec_matches = [
                    item[1] for item in ranked_candidates
                    if (item[1].section == target_sec or target_sec in item[1].section_title.lower())
                    and (item[1].document_id, item[1].text[:60]) not in selected_keys
                ]
                for sc in sec_matches[:2]:
                    selected_keys.add((sc.document_id, sc.text[:60]))
                    final_chunks.append(sc)

            # Fill remaining slots up to top_k from ranked candidates
            for item in ranked_candidates:
                ck = (item[1].document_id, item[1].text[:60])
                if ck not in selected_keys:
                    selected_keys.add(ck)
                    final_chunks.append(item[1])
                if len(final_chunks) >= top_k:
                    break
        else:
            final_chunks = [item[1] for item in ranked_candidates[:top_k]]

        logger.info(
            "HYBRID RETRIEVAL | query='%s' | target_sections=%s | scope=%s | raw=%d | passed=%d",
            query[:50],
            target_sections or "NONE",
            document_id,
            len(raw_results),
            len(final_chunks),
        )

        return final_chunks
