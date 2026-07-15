"""Application settings loaded from environment variables.

Uses pydantic-settings so all values are validated and typed at startup.
A single Settings instance is created at application launch and injected
into every component that needs configuration — no component reads os.environ
directly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for the AI Document Intelligence Platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ───────────────────────────────
    # Application
    # ───────────────────────────────
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    debug: bool = False

    # ───────────────────────────────
    # File Upload
    # ───────────────────────────────
    upload_dir: str = "./data/uploads"
    upload_max_size_mb: int = 50

    # ───────────────────────────────
    # SQLite Database
    # ───────────────────────────────
    database_url: str = "sqlite:///./data/platform.db"

    # ───────────────────────────────
    # ChromaDB
    # ───────────────────────────────
    chroma_db_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "documents"

    # ───────────────────────────────
    # Ollama
    # ───────────────────────────────
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_llm_model: str = "qwen3:8b"
    ollama_embedding_model: str = "nomic-embed-text:latest"
    ollama_timeout_seconds: int = 600

    # -------- LLM Generation Options --------

    # Maximum tokens to generate.
    # Metadata extraction needs very few tokens.
    ollama_num_predict: int = 96

    # Context window supplied to the model.
    ollama_num_ctx: int = 4096

    # Deterministic generation.
    ollama_temperature: float = 0.0

    # Sampling controls.
    ollama_top_k: int = 20
    ollama_top_p: float = 0.90

    # Reduce repetitive output.
    ollama_repeat_penalty: float = 1.10

    # Hardware tuning.
    # -1 lets Ollama decide automatically.
    ollama_num_gpu: int = -1

    # CPU threads.
    # 0 lets Ollama choose.
    ollama_num_thread: int = 0

    # ───────────────────────────────
    # RAG Pipeline
    # ───────────────────────────────
    chunk_size: int = 600
    chunk_overlap: int = 100
    retrieval_k: int = 8
    embedding_batch_size: int = 32

    # ───────────────────────────────
    # Debug
    # ───────────────────────────────
    debug_rag: bool = False
    log_stages: bool = False

    # ───────────────────────────────
    # Logging
    # ───────────────────────────────
    log_level: str = "INFO"
    log_dir: str = "./logs"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

        upper = v.upper()

        if upper not in allowed:
            raise ValueError(
                f"log_level must be one of {allowed}, got '{v}'"
            )

        return upper

    @property
    def db_path(self) -> Path:
        """Resolve SQLite path."""
        raw = self.database_url.replace("sqlite:///", "")
        return Path(raw)

    @property
    def upload_max_size_bytes(self) -> int:
        return self.upload_max_size_mb * 1024 * 1024

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton Settings instance."""
    return Settings()