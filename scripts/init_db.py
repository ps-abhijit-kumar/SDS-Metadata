#!/usr/bin/env python
"""Initialize the SQLite database.

This script is called during Docker image build to ensure the schema exists.
Can also be run manually to reset the database.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.infrastructure.configuration.settings import Settings
from app.infrastructure.database.sqlite_database import SQLiteDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    settings = Settings()
    db = SQLiteDatabase(settings)
    db.initialise()
    logger.info("Database initialized at %s", settings.db_path)


if __name__ == "__main__":
    main()
