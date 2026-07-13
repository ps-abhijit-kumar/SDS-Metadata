"""FastAPI application entry point.

The lifespan context manager handles startup and shutdown:
  - Startup: configure logging, initialise the DI container (database, vector store, LLM clients).
  - Shutdown: log graceful shutdown message.

All application state is owned by the lifespan. Nothing is module-level global
outside the DI container.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.domain.exceptions.base import ApplicationException
from app.infrastructure.configuration.settings import get_settings
from app.infrastructure.logging.log_config import configure_logging
from app.presentation.dependencies.container import initialise
from app.presentation.middleware.exception_handlers import (
    application_exception_handler,
    unhandled_exception_handler,
)
from app.presentation.routers.extraction_router import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    settings = get_settings()
    configure_logging(settings)
    logger.info("Starting AI Document Intelligence Platform | env=%s", settings.app_env)
    initialise()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown")


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title="AI Document Intelligence Platform",
        description=(
            "Production-grade local RAG pipeline for SDS document metadata extraction. "
            "Powered by Ollama + Qwen3:8B + ChromaDB + LangChain. "
            "Runs 100% locally — no cloud APIs required."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow the Streamlit frontend running locally
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            f"http://{settings.app_host}:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(ApplicationException, application_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Routers
    app.include_router(router)

    @app.get("/", tags=["Root"])
    def root():
        return {
            "service": "AI Document Intelligence Platform",
            "status": "online",
            "frontend": "http://127.0.0.1:8501",
            "docs": "http://127.0.0.1:8000/docs",
        }

    @app.get("/health", tags=["Health"])
    def health_check():
        return {"status": "ok", "service": "AI Document Intelligence Platform"}

    return app


app = create_app()
