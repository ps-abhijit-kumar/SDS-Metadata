"""ExtractMetadataUseCase — the core orchestration use case.

This class is the entry point for the entire metadata extraction pipeline.
It receives all dependencies through constructor injection and coordinates:

  PDF → Text → Clean → Chunk → Embed → Store → Retrieve → Prompt → LLM
  → Parse → Validate → Persist → Return

Business rules:
  - The document record is created as PENDING immediately on receipt.
  - Status is updated to PROCESSING before the pipeline begins.
  - On success → COMPLETED with metadata.
  - On any failure → FAILED with a human-readable error message.
  - The original uploaded file is never modified.
  - Only retrieved chunks are sent to the LLM — the full document is never sent.

Debug Mode (DEBUG_RAG=true):
  - Records execution time for each of the 10 pipeline stages.
  - Captures retrieved chunks, prompt, raw LLM response, and parsed metadata.
  - Exposed in the API response for development troubleshooting.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from app.application.dto.extraction_result import ExtractionResultDTO
from app.application.services.chunking_service import ChunkingService
from app.application.services.debug_context import DebugContext, StageTimer
from app.application.services.metadata_validator import MetadataValidator
from app.application.services.prompt_builder import build_extraction_prompt
from app.application.services.text_cleaner import TextCleaner
from app.domain.entities.document import Document
from app.domain.exceptions.base import ApplicationException
from app.domain.repositories.document_repository import DocumentRepository
from app.infrastructure.configuration.settings import Settings
from app.infrastructure.llm.ollama_llm_client import OllamaLLMClient
from app.infrastructure.pdf.document_reader import DocumentReader
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# Retrieval queries targeting sections most relevant to each metadata field.
# Minimal set for faster LLM processing.
_RETRIEVAL_QUERIES: list[str] = [
    # Product name, company name, and key identification info
    "product name trade name company chemical identification section 1",
    # Regulatory jurisdiction and language indicators
    "REACH CLP OSHA WHMIS regulation standard jurisdiction section 15 16",
]


class ExtractMetadataUseCase:
    """Orchestrates the full SDS metadata extraction pipeline."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_reader: DocumentReader,
        text_cleaner: TextCleaner,
        chunking_service: ChunkingService,
        vector_store: ChromaVectorStore,
        llm_client: OllamaLLMClient,
        metadata_validator: MetadataValidator,
        settings: Settings,
        retrieval_k: int = 2,
    ) -> None:
        self._repository = document_repository
        self._reader = document_reader
        self._cleaner = text_cleaner
        self._chunker = chunking_service
        self._vector_store = vector_store
        self._llm = llm_client
        self._validator = metadata_validator
        self._settings = settings
        self._retrieval_k = retrieval_k

    def execute(self, file_path: Path, original_filename: str) -> ExtractionResultDTO:
        """Run the full pipeline for a single PDF file.

        Args:
            file_path: Absolute path to the uploaded PDF on disk.
            original_filename: The original filename from the upload.

        Returns:
            ExtractionResultDTO with the extracted metadata or failure info.
            When DEBUG_RAG is enabled, includes timing and debug information.
        """
        document_id = str(uuid.uuid4())
        document = Document(
            id=document_id,
            filename=original_filename,
            file_path=str(file_path),
        )

        # Create debug context if enabled
        debug_ctx = DebugContext() if self._settings.debug_rag else None
        pipeline_start = time.time()

        # Persist the document record immediately so the client can poll status
        self._repository.save(document)
        logger.info("Pipeline start | document_id=%s | file=%s", document_id, original_filename)

        try:
            # ── Stage 1: Mark processing ──────────────────────────────────────
            document.mark_processing()
            self._repository.update(document)
            if debug_ctx:
                debug_ctx.add_stage_timing("Pipeline init", (time.time() - pipeline_start) * 1000)

            # ── Stage 2: PDF text extraction ──────────────────────────────────
            with StageTimer("PDF extraction") as timer:
                logger.debug("[%s] Stage 2: PDF extraction", document_id)
                extracted = self._reader.read(file_path)
            if debug_ctx:
                debug_ctx.add_stage_timing("PDF extraction", timer.duration_ms)
            logger.info(
                "✓ PDF extraction completed | time=%.1f ms | pages=%d | chars=%d",
                timer.duration_ms,
                extracted.num_pages if hasattr(extracted, 'num_pages') else 0,
                len(extracted.full_text) if extracted else 0,
            )

            # ── Stage 3: Text cleaning ────────────────────────────────────────
            with StageTimer("Text cleaning") as timer:
                logger.debug("[%s] Stage 3: Text cleaning", document_id)
                clean_text = self._cleaner.clean(extracted.full_text)
            if debug_ctx:
                debug_ctx.add_stage_timing("Text cleaning", timer.duration_ms)
            logger.info(
                "✓ Text cleaning completed | time=%.1f ms | chars_before=%d | chars_after=%d",
                timer.duration_ms,
                len(extracted.full_text) if extracted else 0,
                len(clean_text),
            )

            # ── Stage 4: Semantic chunking ────────────────────────────────────
            with StageTimer("Semantic chunking") as timer:
                logger.debug("[%s] Stage 4: Semantic chunking", document_id)
                chunks = self._chunker.chunk(clean_text, document_id)
            if debug_ctx:
                debug_ctx.add_stage_timing("Semantic chunking", timer.duration_ms)
            logger.info(
                "✓ Semantic chunking completed | time=%.1f ms | chunks=%d",
                timer.duration_ms,
                len(chunks),
            )

            if not chunks:
                raise ApplicationException(
                    "Document produced no text chunks after processing. "
                    "The PDF may be empty or contain only images."
                )

            # ── Stage 5: Embedding & storage ──────────────────────────────────
            with StageTimer("Embedding & storage") as timer:
                logger.debug("[%s] Stage 5: Embedding %d chunks", document_id, len(chunks))
                self._vector_store.add_documents(
                    document_id=document_id,
                    texts=[c.text for c in chunks],
                    metadatas=[c.metadata for c in chunks],
                )
            if debug_ctx:
                debug_ctx.add_stage_timing("Embedding & storage", timer.duration_ms)
            logger.info(
                "✓ Embedding & storage completed | time=%.1f ms | chunks_stored=%d",
                timer.duration_ms,
                len(chunks),
            )

            # ── Stage 6: Semantic retrieval ───────────────────────────────────
            with StageTimer("Semantic retrieval") as timer:
                logger.debug("[%s] Stage 6: Retrieval | queries=%d", document_id, len(_RETRIEVAL_QUERIES))
                relevant_chunks = self._vector_store.similarity_search(
                    queries=_RETRIEVAL_QUERIES,
                    document_id=document_id,
                    k=self._retrieval_k,
                )
            if debug_ctx:
                debug_ctx.add_stage_timing("Semantic retrieval", timer.duration_ms)
                debug_ctx.retrieved_chunks = relevant_chunks
                debug_ctx.retrieval_query = " | ".join(_RETRIEVAL_QUERIES)
            logger.info(
                "✓ Semantic retrieval completed | time=%.1f ms | chunks_retrieved=%d",
                timer.duration_ms,
                len(relevant_chunks),
            )

            if not relevant_chunks:
                raise ApplicationException(
                    "No relevant chunks were retrieved from the vector store. "
                    "Please try reprocessing the document."
                )

            # ── Stage 7: Prompt construction ──────────────────────────────────
            with StageTimer("Prompt building") as timer:
                logger.debug("[%s] Stage 7: Building prompt | chunks=%d", document_id, len(relevant_chunks))
                prompt = build_extraction_prompt(relevant_chunks)
            if debug_ctx:
                debug_ctx.add_stage_timing("Prompt building", timer.duration_ms)
                debug_ctx.llm_prompt = prompt
            logger.info(
                "✓ Prompt building completed | time=%.1f ms | prompt_len=%d",
                timer.duration_ms,
                len(prompt),
            )

            # ── Stage 8: LLM inference ────────────────────────────────────────
            with StageTimer("LLM inference") as timer:
                logger.debug("[%s] Stage 8: LLM inference", document_id)
                llm_response = self._llm.generate(prompt)
            if debug_ctx:
                debug_ctx.add_stage_timing("LLM inference", timer.duration_ms)
                debug_ctx.llm_raw_response = llm_response
            logger.info(
                "✓ LLM inference completed | time=%.1f s | response_len=%d",
                timer.duration_ms / 1000.0,
                len(llm_response),
            )

            # ── Stage 9: Parse & validate ─────────────────────────────────────
            with StageTimer("Metadata parsing") as timer:
                logger.debug("[%s] Stage 9: Parsing LLM response", document_id)
                metadata = self._validator.parse_and_validate(document_id, llm_response)
            if debug_ctx:
                debug_ctx.add_stage_timing("Metadata parsing", timer.duration_ms)
                debug_ctx.parsed_metadata = metadata.to_dict()
            logger.info(
                "✓ Metadata parsing completed | time=%.1f ms | product=%s | lang=%s | jurisdiction=%s",
                timer.duration_ms,
                metadata.product_name or "?",
                metadata.language or "?",
                metadata.jurisdiction or "?",
            )

            # ── Stage 10: Persist result ──────────────────────────────────────
            with StageTimer("Database persistence") as timer:
                document.mark_completed(metadata)
                self._repository.update(document)
            if debug_ctx:
                debug_ctx.add_stage_timing("Database persistence", timer.duration_ms)
            logger.info(
                "✓ Database persistence completed | time=%.1f ms | document_id=%s",
                timer.duration_ms,
                document_id,
            )

            total_ms = (time.time() - pipeline_start) * 1000
            logger.info(
                "=== ✓ PIPELINE COMPLETE ===\n"
                "document_id=%s | file=%s\n"
                "total_time=%.1f s (%d ms)\n"
                "product_name=%s\n"
                "language=%s\n"
                "jurisdiction=%s\n"
                "company_name=%s",
                document_id,
                original_filename,
                total_ms / 1000.0,
                int(total_ms),
                metadata.product_name or "?",
                metadata.language or "?",
                metadata.jurisdiction or "?",
                metadata.company_name or "?",
            )

            dto = self._to_dto(document)
            if debug_ctx:
                dto.debug_metadata = debug_ctx.to_dict()
            return dto

        except Exception as exc:
            error_msg = str(exc)
            logger.error(
                "=== ✗ PIPELINE FAILED ===\n"
                "document_id=%s | file=%s\n"
                "error=%s",
                document_id,
                original_filename,
                error_msg,
            )
            document.mark_failed(error_msg)
            try:
                self._repository.update(document)
            except Exception as repo_exc:
                logger.error("Failed to persist failure state: %s", repo_exc)
            dto = self._to_dto(document)
            if debug_ctx:
                dto.debug_metadata = debug_ctx.to_dict()
            return dto

    def _to_dto(self, document: Document) -> ExtractionResultDTO:
        meta = document.metadata
        return ExtractionResultDTO(
            document_id=document.id,
            filename=document.filename,
            status=document.status,
            product_name=meta.product_name if meta else None,
            language=meta.language if meta else None,
            jurisdiction=meta.jurisdiction if meta else None,
            company_name=meta.company_name if meta else None,
            error_message=document.error_message,
            created_at=document.created_at,
        )
