"""Health check router for verifying system dependencies."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends

from app.infrastructure.configuration.settings import Settings, get_settings
from app.infrastructure.database.sqlite_database import SQLiteDatabase
from app.presentation.dependencies.container import get_database

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", summary="Comprehensive system health check")
def health_check(
    settings: Settings = Depends(get_settings),
    db: SQLiteDatabase = Depends(get_database),
) -> dict:
    """Verify backend, SQLite, ChromaDB, Ollama connection and model availability."""
    status_report = {
        "status": "ok",
        "fastapi": "online",
        "sqlite": "unknown",
        "ollama": "unknown",
        "chat_model": "unknown",
        "embedding_model": "unknown",
    }

    # 1. SQLite check
    try:
        with db.connection() as conn:
            conn.execute("SELECT 1")
        status_report["sqlite"] = "connected"
    except Exception as exc:
        status_report["sqlite"] = f"error: {exc}"
        status_report["status"] = "degraded"

    # 2. Ollama connection & model check
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        if r.status_code == 200:
            status_report["ollama"] = "reachable"
            models_data = r.json().get("models", [])
            installed_models = {m.get("name") for m in models_data}

            # Check chat model
            chat_m = settings.chat_model
            status_report["chat_model"] = "available" if any(chat_m in m for m in installed_models) else f"not installed ({chat_m})"

            # Check embedding model
            emb_m = settings.ollama_embedding_model.split(":")[0]
            status_report["embedding_model"] = "available" if any(emb_m in m for m in installed_models) else f"not installed ({emb_m})"
        else:
            status_report["ollama"] = f"unreachable (HTTP {r.status_code})"
            status_report["status"] = "degraded"
    except Exception as exc:
        status_report["ollama"] = f"offline ({exc})"
        status_report["status"] = "degraded"

    return status_report
