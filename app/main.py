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
from app.presentation.routers.chat_router import router as chat_router
from app.presentation.routers.extraction_router import router as extraction_router
from app.presentation.routers.health_router import router as health_router

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
        title="SDS Document Intelligence & Conversational RAG Platform",
        description=(
            "Production-grade SDS metadata extraction and document-grounded RAG platform. "
            "Powered by Ollama + qwen3:4b-instruct + ChromaDB + PyMuPDF. "
            "Runs 100% locally — no cloud APIs required."
        ),
        version="2.0.0",
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
    app.include_router(extraction_router)
    app.include_router(chat_router)
    app.include_router(health_router)

    @app.get("/", tags=["Root"])
    def root():
        return {
            "service": "SDS Document Intelligence & Conversational RAG Platform",
            "status": "online",
            "version": "2.0.0",
            "frontend": "http://127.0.0.1:8501",
            "docs": "http://127.0.0.1:8000/docs",
        }

    return app


app = create_app()
