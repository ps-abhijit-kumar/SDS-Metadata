"""Use case for executing document-grounded RAG chat with metadata intent routing."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Generator

from app.application.dto.chat_dto import ChatResponseDTO, SourceCitationDTO
from app.application.services.chat_service import ChatService
from app.application.services.grounding_service import GroundingService
from app.application.services.intent_router import IntentRouter, MetadataIntent
from app.application.services.query_analyzer import QueryAnalyzer
from app.application.services.retrieval_service import RetrievalService
from app.domain.entities.chat_message import ChatMessage
from app.domain.repositories.chat_repository import ChatRepository
from app.domain.repositories.document_repository import DocumentRepository
from app.infrastructure.configuration.settings import Settings

logger = logging.getLogger(__name__)


class ChatWithDocumentUseCase:
    """Orchestrates document-grounded RAG conversation and fast metadata routing."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        grounding_service: GroundingService,
        chat_service: ChatService,
        intent_router: IntentRouter,
        document_repository: DocumentRepository,
        chat_repository: ChatRepository,
        settings: Settings,
    ) -> None:
        self._retrieval = retrieval_service
        self._grounding = grounding_service
        self._chat = chat_service
        self._intent_router = intent_router
        self._doc_repo = document_repository
        self._chat_repo = chat_repository
        self._settings = settings

    def execute(
        self,
        user_query: str,
        document_id: str = "all",
        conversation_id: str | None = None,
    ) -> ChatResponseDTO:
        """Synchronous chat execution with metadata intent routing & grounding validation."""
        start_time = time.time()
        conv_id = conversation_id or str(uuid.uuid4())
        is_multi_doc = document_id.lower() == "all"
        scope = "all" if is_multi_doc else "single"

        # ── 1. Metadata Intent Detection ───────────────────────────────────
        intent = self._intent_router.detect_intent(user_query)

        if intent != MetadataIntent.NONE:
            intent_start = time.time()
            target_doc = None
            all_docs = None

            if is_multi_doc:
                all_docs = self._doc_repo.find_all_canonical()
            else:
                target_doc = self._doc_repo.find_by_id(document_id)

            answer, is_grounded, sources_list = self._intent_router.format_metadata_response(
                intent=intent,
                document=target_doc,
                all_documents=all_docs,
                scope=scope,
            )

            intent_ms = (time.time() - intent_start) * 1000
            total_ms = (time.time() - start_time) * 1000

            logger.info(
                "CHAT ROUTER | intent=%s | source=DOCUMENT_METADATA | llm_called=False | retrieval_ms=0.0 | prompt_build_ms=0.0 | llm_first_token_ms=0.0 | llm_total_ms=0.0 | total_chat_ms=%.2f",
                intent.value,
                total_ms,
            )

            sources_dto = [SourceCitationDTO(**s) for s in sources_list] if is_grounded else []

            msg = ChatMessage(
                id=str(uuid.uuid4()),
                conversation_id=conv_id,
                document_id=document_id,
                user_query=user_query,
                assistant_response=answer,
                grounded=is_grounded,
                sources_json=json.dumps(sources_list) if is_grounded else "[]",
            )
            self._chat_repo.save(msg)

            return ChatResponseDTO(
                answer=answer,
                grounded=is_grounded,
                conversation_id=conv_id,
                document_id=document_id,
                sources=sources_dto,
                retrieval_ms=0.0,
                llm_ms=0.0,
                total_ms=total_ms,
            )

        # ── 2. Normal RAG Path: Retrieval & Distance Thresholding ───────────
        ret_start = time.time()
        analysis = self._retrieval._query_analyzer.analyze(user_query) if hasattr(self._retrieval, "_query_analyzer") else QueryAnalyzer().analyze(user_query)
        chunks = self._retrieval.retrieve(query=user_query, document_id=document_id)
        ret_ms = (time.time() - ret_start) * 1000

        # Retrieve document metadata for grounded overview injection
        doc_meta = None
        if analysis.is_overview or not is_multi_doc:
            if is_multi_doc:
                all_canonical = self._doc_repo.find_all_canonical()
                doc_meta = [
                    {
                        "filename": d.filename,
                        "product_name": d.metadata.product_name if d.metadata else None,
                        "company_name": d.metadata.company_name if d.metadata else None,
                        "language": d.metadata.language if d.metadata else None,
                        "jurisdiction": d.metadata.jurisdiction if d.metadata else None,
                    }
                    for d in all_canonical if d.metadata
                ]
            else:
                target_doc = self._doc_repo.find_by_id(document_id)
                if target_doc and target_doc.metadata:
                    doc_meta = {
                        "filename": target_doc.filename,
                        "product_name": target_doc.metadata.product_name,
                        "company_name": target_doc.metadata.company_name,
                        "language": target_doc.metadata.language,
                        "jurisdiction": target_doc.metadata.jurisdiction,
                    }

        # ── 3. Grounding Verification ──────────────────────────────────────
        grounding_start = time.time()
        is_grounded, fallback_msg = self._grounding.verify_grounding(
            chunks=chunks,
            query=user_query,
            scope=scope,
            is_overview=analysis.is_overview and bool(chunks or doc_meta),
        )
        grounding_ms = (time.time() - grounding_start) * 1000

        if not is_grounded:
            total_ms = (time.time() - start_time) * 1000
            logger.info(
                "GROUNDING REJECTED | query='%s' | scope=%s | llm_called=False | retrieval_ms=%.1f | total_chat_ms=%.1f",
                user_query[:40],
                document_id,
                ret_ms,
                total_ms,
            )
            msg = ChatMessage(
                id=str(uuid.uuid4()),
                conversation_id=conv_id,
                document_id=document_id,
                user_query=user_query,
                assistant_response=fallback_msg,
                grounded=False,
                sources_json="[]",
            )
            self._chat_repo.save(msg)

            return ChatResponseDTO(
                answer=fallback_msg,
                grounded=False,
                conversation_id=conv_id,
                document_id=document_id,
                sources=[],
                retrieval_ms=ret_ms,
                llm_ms=0.0,
                total_ms=total_ms,
            )

        # ── 4. LLM Generation for Grounded RAG ─────────────────────────────
        prompt_start = time.time()
        sys_prompt, user_prompt = self._chat.build_prompt(
            question=user_query,
            chunks=chunks,
            is_multi_doc=is_multi_doc,
            doc_metadata=doc_meta if analysis.is_overview else None,
            is_overview=analysis.is_overview,
            document_id=document_id,
        )
        prompt_build_ms = (time.time() - prompt_start) * 1000

        llm_start = time.time()
        answer, first_token_ms = self._chat.generate_chat_response_with_metrics(sys_prompt, user_prompt)
        llm_total_ms = (time.time() - llm_start) * 1000

        sources = self._chat.extract_sources(
            chunks,
            doc_metadata=doc_meta if analysis.is_overview else None,
        )
        sources_json = json.dumps([s.to_dict() for s in sources])

        total_chat_ms = (time.time() - start_time) * 1000

        msg = ChatMessage(
            id=str(uuid.uuid4()),
            conversation_id=conv_id,
            document_id=document_id,
            user_query=user_query,
            assistant_response=answer,
            grounded=True,
            sources_json=sources_json,
        )
        self._chat_repo.save(msg)

        logger.info(
            "CHAT PERFORMANCE METRICS | retrieval_ms=%.1f | prompt_build_ms=%.1f | llm_first_token_ms=%.1f | llm_total_ms=%.1f | total_chat_ms=%.1f",
            ret_ms,
            prompt_build_ms,
            first_token_ms,
            llm_total_ms,
            total_chat_ms,
        )

        return ChatResponseDTO(
            answer=answer,
            grounded=True,
            conversation_id=conv_id,
            document_id=document_id,
            sources=sources,
            retrieval_ms=ret_ms,
            llm_ms=llm_total_ms,
            total_ms=total_chat_ms,
        )

    def execute_stream(
        self,
        user_query: str,
        document_id: str = "all",
        conversation_id: str | None = None,
    ) -> tuple[Generator[str, None, None], list[dict], bool, str]:
        """Streaming chat execution with intent routing & grounding check."""
        conv_id = conversation_id or str(uuid.uuid4())
        is_multi_doc = document_id.lower() == "all"
        scope = "all" if is_multi_doc else "single"

        # ── 1. Metadata Intent Check ───────────────────────────────────────
        intent = self._intent_router.detect_intent(user_query)
        if intent != MetadataIntent.NONE:
            target_doc = None
            all_docs = None

            if is_multi_doc:
                all_docs = self._doc_repo.find_all_canonical()
            else:
                target_doc = self._doc_repo.find_by_id(document_id)

            answer, is_grounded, sources_list = self._intent_router.format_metadata_response(
                intent=intent,
                document=target_doc,
                all_documents=all_docs,
                scope=scope,
            )

            def meta_gen():
                yield answer

            msg = ChatMessage(
                id=str(uuid.uuid4()),
                conversation_id=conv_id,
                document_id=document_id,
                user_query=user_query,
                assistant_response=answer,
                grounded=is_grounded,
                sources_json=json.dumps(sources_list) if is_grounded else "[]",
            )
            self._chat_repo.save(msg)
            return meta_gen(), sources_list if is_grounded else [], is_grounded, conv_id

        # ── 2. Retrieval & Distance Check ───────────────────────────────────
        analysis = self._retrieval._query_analyzer.analyze(user_query) if hasattr(self._retrieval, "_query_analyzer") else QueryAnalyzer().analyze(user_query)
        chunks = self._retrieval.retrieve(query=user_query, document_id=document_id)

        # Retrieve document metadata for grounded overview injection
        doc_meta = None
        if analysis.is_overview or not is_multi_doc:
            if is_multi_doc:
                all_canonical = self._doc_repo.find_all_canonical()
                doc_meta = [
                    {
                        "filename": d.filename,
                        "product_name": d.metadata.product_name if d.metadata else None,
                        "company_name": d.metadata.company_name if d.metadata else None,
                        "language": d.metadata.language if d.metadata else None,
                        "jurisdiction": d.metadata.jurisdiction if d.metadata else None,
                    }
                    for d in all_canonical if d.metadata
                ]
            else:
                target_doc = self._doc_repo.find_by_id(document_id)
                if target_doc and target_doc.metadata:
                    doc_meta = {
                        "filename": target_doc.filename,
                        "product_name": target_doc.metadata.product_name,
                        "company_name": target_doc.metadata.company_name,
                        "language": target_doc.metadata.language,
                        "jurisdiction": target_doc.metadata.jurisdiction,
                    }

        is_grounded, fallback_msg = self._grounding.verify_grounding(
            chunks=chunks,
            query=user_query,
            scope=scope,
            is_overview=analysis.is_overview and bool(chunks or doc_meta),
        )

        if not is_grounded:
            def fallback_gen():
                yield fallback_msg

            msg = ChatMessage(
                id=str(uuid.uuid4()),
                conversation_id=conv_id,
                document_id=document_id,
                user_query=user_query,
                assistant_response=fallback_msg,
                grounded=False,
                sources_json="[]",
            )
            self._chat_repo.save(msg)
            return fallback_gen(), [], False, conv_id

        # ── 3. Streaming Grounded RAG ───────────────────────────────────────
        sys_prompt, user_prompt = self._chat.build_prompt(
            question=user_query,
            chunks=chunks,
            is_multi_doc=is_multi_doc,
            doc_metadata=doc_meta if analysis.is_overview else None,
            is_overview=analysis.is_overview,
            document_id=document_id,
        )
        sources = self._chat.extract_sources(
            chunks,
            doc_metadata=doc_meta if analysis.is_overview else None,
        )
        sources_list = [s.to_dict() for s in sources]

        def stream_with_persistence():
            full_response = []
            for token in self._chat.stream_chat_response(sys_prompt, user_prompt):
                full_response.append(token)
                yield token

            complete_answer = "".join(full_response).strip()
            msg = ChatMessage(
                id=str(uuid.uuid4()),
                conversation_id=conv_id,
                document_id=document_id,
                user_query=user_query,
                assistant_response=complete_answer,
                grounded=True,
                sources_json=json.dumps(sources_list),
            )
            try:
                self._chat_repo.save(msg)
            except Exception as exc:
                logger.error("Failed to save streamed chat message: %s", exc)

        return stream_with_persistence(), sources_list, True, conv_id
