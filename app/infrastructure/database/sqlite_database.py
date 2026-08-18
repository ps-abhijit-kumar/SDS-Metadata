"""SQLite database initialiser and connection manager.

Responsible for:
  1. Ensuring the database file and parent directories exist.
  2. Creating the schema (CREATE TABLE IF NOT EXISTS).
  3. Providing a context-managed connection factory used by repositories.

This class knows about SQLite only — no domain types are imported here.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from app.infrastructure.configuration.settings import Settings

logger = logging.getLogger(__name__)

_BASE_TABLES_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id                  TEXT PRIMARY KEY,
    filename            TEXT NOT NULL,
    file_path           TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',
    product_name        TEXT,
    company_name        TEXT,
    language            TEXT,
    jurisdiction        TEXT,
    error_message       TEXT,
    file_hash           TEXT,
    processing_version  TEXT DEFAULT 'v1',
    version_number      INTEGER DEFAULT 1,
    is_active           INTEGER DEFAULT 1,
    processing_time_ms  REAL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_history (
    id                  TEXT PRIMARY KEY,
    conversation_id     TEXT NOT NULL,
    document_id         TEXT NOT NULL,
    user_query          TEXT NOT NULL,
    assistant_response  TEXT NOT NULL,
    grounded            INTEGER NOT NULL DEFAULT 1,
    sources_json        TEXT,
    created_at          TEXT NOT NULL
);
"""

_INDEXES_DDL = """
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_chat_doc_id ON chat_history(document_id);
CREATE INDEX IF NOT EXISTS idx_chat_conv_id ON chat_history(conversation_id);
"""


class SQLiteDatabase:
    """Manages the SQLite connection lifecycle and schema initialisation."""

    def __init__(self, settings: Settings) -> None:
        self._db_path: Path = settings.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialise(self) -> None:
        """Create tables if they do not already exist, migrate missing columns & data, then create indexes."""
        logger.info("Initialising SQLite database at %s", self._db_path)
        with self.connection() as conn:
            # 1. Create base tables
            conn.executescript(_BASE_TABLES_DDL)
            # 2. Migrate existing columns
            self._migrate_schema(conn)
            # 3. Migrate historical data (canonical language names)
            self._migrate_data(conn)
            # 4. Create indexes AFTER columns exist
            conn.executescript(_INDEXES_DDL)
        logger.info("SQLite schema ready")

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Safe non-destructive migration for existing SQLite databases."""
        cursor = conn.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cursor.fetchall()}
        
        if "file_hash" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT")
        if "company_name" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN company_name TEXT")
        if "processing_time_ms" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN processing_time_ms REAL")
        if "processing_version" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN processing_version TEXT DEFAULT 'v1'")
        if "version_number" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN version_number INTEGER DEFAULT 1")
        if "is_active" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN is_active INTEGER DEFAULT 1")

    def _migrate_data(self, conn: sqlite3.Connection) -> None:
        """Backfill and normalize historical language values in existing database."""
        mappings = [
            ("Spanish", ["es", "spa", "espanol", "español", "castellano"]),
            ("English", ["en", "eng", "english", "inglés", "ingles"]),
            ("Portuguese", ["pt", "por", "portuguese", "português", "portugues"]),
            ("German", ["de", "ger", "deu", "german", "deutsch", "alemán"]),
            ("French", ["fr", "fra", "fre", "french", "français", "francais"]),
            ("Italian", ["it", "ita", "italian", "italiano"]),
            ("Dutch", ["nl", "nld", "dut", "dutch", "nederlands"]),
        ]
        for canonical, variants in mappings:
            placeholders = ", ".join("?" for _ in variants)
            conn.execute(
                f"UPDATE documents SET language = ? WHERE LOWER(TRIM(language)) IN ({placeholders})",
                [canonical, *variants],
            )

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection that auto-commits on clean exit and rolls back on error."""
        conn = sqlite3.connect(
            str(self._db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
