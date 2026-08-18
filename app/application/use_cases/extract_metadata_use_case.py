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
from app.application.services.language_detector import LanguageDetector
from app.application.services.language_normalizer import normalize_language
from app.application.services.metadata_validator import MetadataValidator, _verify_and_disambiguate_company, _verify_and_disambiguate_product
from app.application.services.prompt_builder import build_extraction_prompt
from app.application.services.text_cleaner import TextCleaner
from app.domain.entities.document import Document
from app.domain.exceptions.base import ApplicationException
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.value_objects.sds_metadata import SDSMetadata
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


CURRENT_PROCESSING_VERSION = "v3"


class ExtractMetadataUseCase:
    """Orchestrates PDF ingestion, duplicate checking, text extraction, semantic chunking, and LLM metadata extraction."""

    def __init__(
        self,
        document_repository: IDocumentRepository,
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
        self._retrieval_k = getattr(settings, 'metadata_retrieval_k', retrieval_k)

    def execute(
        self,
        file_path: Path,
        original_filename: str,
    ) -> ExtractionResultDTO:
        import hashlib
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
        except Exception:
            file_hash = ""

        latest_by_name = self._repository.find_latest_by_filename(original_filename)
        existing_by_hash = self._repository.find_by_hash(file_hash)

        if existing_by_hash:
            if existing_by_hash.processing_version == CURRENT_PROCESSING_VERSION:
                logger.info(
                    "⚡ Instant Duplicate Path: Content SHA-256 (%s) matches existing document %s (version %d). Reusing metadata.",
                    file_hash[:12],
                    existing_by_hash.id,
                    existing_by_hash.version_number,
                )
                self._repository.deactivate_previous_versions(original_filename, existing_by_hash.id)
                existing_by_hash.is_active = True
                self._repository.update(existing_by_hash)

                dup_doc = Document(
                    id=str(uuid.uuid4()),
                    filename=original_filename,
                    file_path=str(file_path),
                    file_hash=file_hash,
                    processing_version=CURRENT_PROCESSING_VERSION,
                    version_number=existing_by_hash.version_number,
                    is_active=False,
                )
                dup_doc.mark_duplicate(existing_by_hash.metadata)
                self._repository.save(dup_doc)
                return self._to_dto(dup_doc)

            # ── CASE B: Same Content + Outdated Index Version -> Vector Rebuild Only ──
            logger.info(
                "🔄 Index Rebuild Path: Content SHA-256 (%s) matches but index version is '%s' (current='%s'). Rebuilding vectors WITHOUT LLM call for %s",
                file_hash[:12],
                existing_by_hash.processing_version,
                CURRENT_PROCESSING_VERSION,
                original_filename,
            )
            extracted = self._reader.read(file_path)
            clean_text = self._cleaner.clean(extracted.full_text)
            if hasattr(extracted, 'pages') and extracted.pages:
                chunks = self._chunker.chunk_pages(
                    pages=extracted.pages,
                    document_id=existing_by_hash.id,
                    filename=original_filename,
                    document_hash=file_hash,
                )
            else:
                chunks = self._chunker.chunk(clean_text, existing_by_hash.id, filename=original_filename, document_hash=file_hash)

            if chunks:
                self._vector_store.add_documents(
                    document_id=existing_by_hash.id,
                    texts=[c.text for c in chunks],
                    metadatas=[c.metadata for c in chunks],
                )

            existing_by_hash.processing_version = CURRENT_PROCESSING_VERSION
            existing_by_hash.is_active = True
            self._repository.deactivate_previous_versions(original_filename, existing_by_hash.id)
            self._repository.update(existing_by_hash)

            return self._to_dto(existing_by_hash)

        # ── CASE C / D: New Content Version or New Document ───────────────
        next_version_num = (latest_by_name.version_number + 1) if latest_by_name else 1

        document_id = str(uuid.uuid4())
        document = Document(
            id=document_id,
            filename=original_filename,
            file_path=str(file_path),
            file_hash=file_hash,
            processing_version=CURRENT_PROCESSING_VERSION,
            version_number=next_version_num,
            is_active=True,
        )

        debug_ctx = DebugContext() if self._settings.debug_rag else None
        pipeline_start = time.time()

        self._repository.deactivate_previous_versions(original_filename, document_id)
        self._repository.save(document)
        logger.info(
            "Pipeline start | document_id=%s | file=%s | hash=%s | ver=%d",
            document_id,
            original_filename,
            file_hash[:12] if file_hash else "",
            next_version_num,
        )

        try:
            # ── Stage 1: Mark processing ──────────────────────────────────────
            document.mark_processing()
            self._repository.update(document)
            if debug_ctx:
                debug_ctx.add_stage_timing("Pipeline init", (time.time() - pipeline_start) * 1000)

            # ── Stage 2: PDF text extraction & Deterministic Language Detection ──
            with StageTimer("PDF extraction") as timer:
                logger.debug("[%s] Stage 2: PDF extraction", document_id)
                extracted = self._reader.read(file_path)
                doc_language = LanguageDetector.detect_language(extracted.full_text) if extracted and extracted.full_text else None
            if debug_ctx:
                debug_ctx.add_stage_timing("PDF extraction", timer.duration_ms)
            logger.info(
                "✓ PDF extraction completed | time=%.1f ms | pages=%d | chars=%d | detected_lang=%s",
                timer.duration_ms,
                extracted.page_count if hasattr(extracted, 'page_count') else 0,
                len(extracted.full_text) if extracted else 0,
                doc_language,
            )

            # ── Stage 3: Text cleaning ────────────────────────────────────────
            with StageTimer("Text cleaning") as timer:
                logger.debug("[%s] Stage 3: Text cleaning", document_id)
                clean_text = self._cleaner.clean(extracted.full_text)
            if debug_ctx:
                debug_ctx.add_stage_timing("Text cleaning", timer.duration_ms)

            # ── Stage 4: Semantic chunking with page & section metadata ──────
            with StageTimer("Semantic chunking") as timer:
                logger.debug("[%s] Stage 4: Semantic chunking", document_id)
                if hasattr(extracted, 'pages') and extracted.pages:
                    chunks = self._chunker.chunk_pages(
                        pages=extracted.pages,
                        document_id=document_id,
                        filename=original_filename,
                        document_hash=file_hash,
                    )
                else:
                    chunks = self._chunker.chunk(clean_text, document_id, filename=original_filename, document_hash=file_hash)
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

            # Include Page 1 header snippet in prompt context if available
            p1_snippet = clean_text[:1200] if clean_text else ""
            prompt_context_chunks = [p1_snippet] + relevant_chunks if p1_snippet else relevant_chunks

            # ── Stage 7: Prompt construction ──────────────────────────────────
            with StageTimer("Prompt building") as timer:
                logger.debug("[%s] Stage 7: Building prompt | chunks=%d", document_id, len(prompt_context_chunks))
                prompt = build_extraction_prompt(prompt_context_chunks)
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
                logger.debug("[%s] Stage 8: LLM inference using model=%s", document_id, self._settings.metadata_model)
                llm_response = self._llm.generate(prompt, model=self._settings.metadata_model)
            llm_ms = timer.duration_ms
            if debug_ctx:
                debug_ctx.add_stage_timing("LLM inference", llm_ms)
                debug_ctx.llm_raw_response = llm_response
            logger.info(
                "✓ LLM inference completed | time=%.1f s | response_len=%d",
                llm_ms / 1000.0,
                len(llm_response),
            )

            # ── Stage 9: Parse, Disambiguate & validate ───────────────────────
            with StageTimer("Metadata parsing") as timer:
                logger.debug("[%s] Stage 9: Parsing LLM response", document_id)
                full_context_str = "\n\n".join(prompt_context_chunks)
                try:
                    metadata = self._validator.parse_and_validate(document_id, llm_response, context_text=full_context_str)
                except TypeError:
                    metadata = self._validator.parse_and_validate(document_id, llm_response)

                # Deterministic language override from local language detector
                if doc_language and metadata.language != doc_language:
                    metadata = SDSMetadata(
                        file_id=metadata.file_id,
                        product_name=metadata.product_name,
                        language=doc_language,
                        jurisdiction=metadata.jurisdiction,
                        company_name=metadata.company_name,
                    )
            val_ms = timer.duration_ms
            if debug_ctx:
                debug_ctx.add_stage_timing("Metadata parsing", val_ms)
                debug_ctx.parsed_metadata = metadata.to_dict()

            total_ms = (time.time() - pipeline_start) * 1000

            # ── Stage 10: Persist result ──────────────────────────────────────
            with StageTimer("Database persistence") as timer:
                document.mark_completed(metadata, processing_time_ms=total_ms)
                self._repository.update(document)
            db_ms = timer.duration_ms
            if debug_ctx:
                debug_ctx.add_stage_timing("Database persistence", db_ms)

            logger.info(
                "METADATA EXTRACTION | filename=%s | model=%s | total_ms=%.1f | llm_ms=%.1f | product=%s | lang=%s | jurisdiction=%s | company=%s",
                original_filename,
                self._settings.metadata_model,
                total_ms,
                llm_ms,
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
