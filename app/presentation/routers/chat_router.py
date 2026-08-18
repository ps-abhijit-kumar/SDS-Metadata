"""Chat router for RAG QA and response streaming."""

from __future__ import annotations

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.application.use_cases.chat_with_document_use_case import ChatWithDocumentUseCase
from app.domain.repositories.chat_repository import ChatRepository
from app.presentation.dependencies.container import (
    get_chat_repository,
    get_chat_use_case,
)
from app.presentation.schemas.chat_schemas import ChatRequest, ChatResponseSchema

router = APIRouter(prefix="/api/v1/chat", tags=["SDS Conversational RAG"])
logger = logging.getLogger(__name__)


@router.post(
    "",
    response_model=ChatResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Ask a document-grounded question",
    description="Query a specific SDS document or search across all indexed SDS documents with strict anti-hallucination grounding.",
)
def chat_with_document(
    request: ChatRequest,
    use_case: ChatWithDocumentUseCase = Depends(get_chat_use_case),
) -> ChatResponseSchema:
    """Synchronous chat endpoint."""
    try:
        dto = use_case.execute(
            user_query=request.question,
            document_id=request.document_id,
            conversation_id=request.conversation_id,
        )
        return ChatResponseSchema(**dto.to_dict())
    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {exc}",
        )


@router.post(
    "/stream",
    summary="Stream document-grounded chat response",
    description="Returns a Server-Sent Events (SSE) stream of text tokens. Grounding is validated BEFORE streaming begins.",
)
def chat_stream(
    request: ChatRequest,
    use_case: ChatWithDocumentUseCase = Depends(get_chat_use_case),
):
    """Streaming SSE chat endpoint."""
    try:
        stream_gen, sources, is_grounded, conv_id = use_case.execute_stream(
            user_query=request.question,
            document_id=request.document_id,
            conversation_id=request.conversation_id,
        )

        def sse_event_generator():
            # Send initial metadata header event
            header = {
                "event": "metadata",
                "conversation_id": conv_id,
                "grounded": is_grounded,
                "sources": sources,
            }
            yield f"data: {json.dumps(header)}\n\n"

            # Stream response tokens
            for token in stream_gen:
                event_data = {"event": "token", "token": token}
                yield f"data: {json.dumps(event_data)}\n\n"

            # Send done event
            yield f"data: {json.dumps({'event': 'done'})}\n\n"

        return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

    except Exception as exc:
        logger.error("Chat streaming endpoint error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat streaming failed: {exc}",
        )


@router.get(
    "/history/{document_id}",
    summary="Get chat history for a document or 'all'",
)
def get_chat_history(
    document_id: str,
    repository: ChatRepository = Depends(get_chat_repository),
):
    """Retrieve persistent conversation history."""
    try:
        messages = repository.find_by_document_id(document_id)
        return {
            "total": len(messages),
            "document_id": document_id,
            "messages": [
                {
                    "id": m.id,
                    "conversation_id": m.conversation_id,
                    "user_query": m.user_query,
                    "assistant_response": m.assistant_response,
                    "grounded": m.grounded,
                    "sources": json.loads(m.sources_json) if m.sources_json else [],
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch chat history: {exc}",
        )
