"""Async extraction service for parallel document processing.

Handles multiple PDFs concurrently without blocking on individual LLM calls.
Uses asyncio to manage concurrent extraction tasks efficiently.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.application.use_cases.extract_metadata_use_case import ExtractMetadataUseCase
from app.infrastructure.configuration.settings import Settings

logger = logging.getLogger(__name__)


class AsyncExtractionService:
    """Manages parallel extraction of multiple documents."""

    def __init__(
        self,
        use_case: ExtractMetadataUseCase,
        settings: Settings,
        max_concurrent: int = 2,
    ) -> None:
        """Initialize async extraction service.

        Args:
            use_case: The synchronous extraction use case
            settings: Application settings
            max_concurrent: Maximum concurrent extractions (default: 2)
        """
        self._use_case = use_case
        self._settings = settings
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(
            "AsyncExtractionService ready | max_concurrent=%d",
            max_concurrent,
        )

    async def extract_parallel(
        self,
        files: list[tuple[Path, str]],
    ) -> list:
        """Extract metadata from multiple files in parallel.

        Args:
            files: List of (file_path, original_filename) tuples

        Returns:
            List of extraction DTOs
        """
        logger.info("Starting parallel extraction | files=%d", len(files))

        # Create tasks for each file
        tasks = [
            self._extract_one(file_path, filename)
            for file_path, filename in files
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        logger.info("Parallel extraction complete | results=%d", len(results))
        return results

    async def _extract_one(self, file_path: Path, filename: str):
        """Extract a single file with concurrency control.

        Uses a semaphore to limit concurrent operations.
        """
        async with self._semaphore:
            logger.debug("Extracting: %s", filename)

            # Run sync extraction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._use_case.execute,
                file_path,
                filename,
            )

            logger.info("Extraction complete: %s | status=%s", filename, result.status)
            return result
