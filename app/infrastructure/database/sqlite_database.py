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

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    product_name    TEXT,
    language        TEXT,
    jurisdiction    TEXT,
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""


class SQLiteDatabase:
    """Manages the SQLite connection lifecycle and schema initialisation."""

    def __init__(self, settings: Settings) -> None:
        self._db_path: Path = settings.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialise(self) -> None:
        """Create tables if they do not already exist. Called at application startup."""
        logger.info("Initialising SQLite database at %s", self._db_path)
        with self.connection() as conn:
            conn.executescript(_DDL)
        logger.info("SQLite schema ready")

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
