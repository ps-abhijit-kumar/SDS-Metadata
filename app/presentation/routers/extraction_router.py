"""Extraction API router - simplified without background tasks.

The extraction runs synchronously but quickly due to:
- Reduced retrieval queries
- Reduced chunk retrieval count
- Optimized pipeline

Endpoints:
  POST /api/v1/extract           — Upload one or more SDS PDFs
  GET  /api/v1/documents         — List all processed documents
  GET  /api/v1/documents/{id}    — Get a single document result
  DELETE /api/v1/documents/{id}  — Delete a document record
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi import status as http_status

from app.application.use_cases.extract_metadata_use_case import ExtractMetadataUseCase
from app.domain.exceptions.base import DocumentNotFoundException, InvalidDocumentException
from app.domain.repositories.document_repository import DocumentRepository
from app.infrastructure.configuration.settings import get_settings
from app.presentation.dependencies.container import (
    get_document_repository,
    get_extract_use_case,
)
from app.presentation.schemas.responses import (
    BatchExtractionResponse,
    DocumentListResponse,
    MetadataResponse,
)

router = APIRouter(prefix="/api/v1", tags=["SDS Extraction"])
logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".pdf"}


def _validate_upload(file: UploadFile, max_bytes: int) -> None:
    """Validate file type. Size is validated after saving."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise InvalidDocumentException(
            f"File type '{suffix}' is not supported. Only PDF files are accepted."
        )


def _save_upload(file: UploadFile, upload_dir: Path) -> Path:
    """Save an uploaded file to the upload directory with a UUID-prefixed name."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4()}_{Path(file.filename or 'upload').name}"
    dest = upload_dir / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


@router.post(
    "/extract",
    response_model=BatchExtractionResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Upload SDS PDF documents and extract metadata",
    description=(
        "Upload one or more SDS PDF files. The system runs the RAG pipeline "
        "for each document and returns the extracted metadata. "
        "This endpoint may take 20-60 seconds per file."
    ),
)
def extract_metadata(
    files: list[UploadFile] = File(..., description="One or more SDS PDF files."),
    use_case: ExtractMetadataUseCase = Depends(get_extract_use_case),
) -> BatchExtractionResponse:
    """Upload and extract metadata from SDS documents."""
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    results: list[MetadataResponse] = []

    for upload_file in files:
        logger.info("Received upload: %s", upload_file.filename)
        try:
            _validate_upload(upload_file, settings.upload_max_size_bytes)
            saved_path = _save_upload(upload_file, upload_dir)

            # Size check after saving
            if saved_path.stat().st_size > settings.upload_max_size_bytes:
                saved_path.unlink(missing_ok=True)
                raise InvalidDocumentException(
                    f"File '{upload_file.filename}' exceeds the maximum allowed size "
                    f"of {settings.upload_max_size_mb} MB."
                )

            # Execute extraction
            dto = use_case.execute(
                file_path=saved_path,
                original_filename=upload_file.filename or saved_path.name,
            )
            results.append(MetadataResponse(**dto.to_dict()))

        except InvalidDocumentException as exc:
            logger.warning("Invalid upload: %s — %s", upload_file.filename, exc.message)
            results.append(
                MetadataResponse(
                    document_id="",
                    filename=upload_file.filename or "unknown",
                    status="failed",
                    error_message=exc.message,
                )
            )
        except Exception as exc:
            logger.exception("Unexpected error processing %s", upload_file.filename)
            results.append(
                MetadataResponse(
                    document_id="",
                    filename=upload_file.filename or "unknown",
                    status="failed",
                    error_message=str(exc),
                )
            )

    return BatchExtractionResponse(total=len(results), results=results)


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all processed documents",
)
def list_documents(
    repository: DocumentRepository = Depends(get_document_repository),
) -> DocumentListResponse:
    documents = repository.find_all()
    responses = []
    for doc in documents:
        meta = doc.metadata
        responses.append(
            MetadataResponse(
                document_id=doc.id,
                filename=doc.filename,
                status=doc.status.value,
                product_name=meta.product_name if meta else None,
                language=meta.language if meta else None,
                jurisdiction=meta.jurisdiction if meta else None,
                company_name=meta.company_name if meta else None,
                error_message=doc.error_message,
                created_at=doc.created_at,
            )
        )
    return DocumentListResponse(total=len(responses), documents=responses)


@router.get(
    "/documents/{document_id}",
    response_model=MetadataResponse,
    summary="Get extraction result for a single document",
)
def get_document(
    document_id: str,
    repository: DocumentRepository = Depends(get_document_repository),
) -> MetadataResponse:
    doc = repository.find_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    meta = doc.metadata
    return MetadataResponse(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status.value,
        product_name=meta.product_name if meta else None,
        language=meta.language if meta else None,
        jurisdiction=meta.jurisdiction if meta else None,
        company_name=meta.company_name if meta else None,
        error_message=doc.error_message,
        created_at=doc.created_at,
    )


@router.delete(
    "/documents/{document_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    summary="Delete a document record",
    response_model=None,
)
def delete_document(
    document_id: str,
    repository: DocumentRepository = Depends(get_document_repository),
):
    try:
        repository.delete(document_id)
    except DocumentNotFoundException:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
