#!/usr/bin/env python
"""End-to-end test of the SDS extraction pipeline."""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from app.infrastructure.configuration.settings import get_settings
from app.presentation.dependencies.container import initialise

def main():
    """Test the full pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("=== Starting E2E Test ===")
    
    try:
        # Initialize DI container
        logger.info("Initializing dependency container...")
        initialise()
        logger.info("✓ Container initialized")
        
        # Get the use case
        from app.presentation.dependencies.container import _extract_use_case
        use_case = _extract_use_case
        
        # Find a test PDF
        test_pdf = Path("data/uploads") / "71a8c0d2-b733-420c-bc30-3dbad3c099d5_l0288.pdf"
        if not test_pdf.exists():
            # Try to find any PDF
            test_pdf = list(Path("data/uploads").glob("*.pdf"))
            if not test_pdf:
                logger.error("No test PDF found in data/uploads")
                return
            test_pdf = test_pdf[0]
        
        logger.info(f"Running extraction on: {test_pdf.name}")
        
        # Execute
        result = use_case.execute(
            file_path=test_pdf,
            original_filename=test_pdf.name,
        )
        
        logger.info("=== Result ===")
        logger.info(f"Status: {result.status}")
        logger.info(f"Product Name: {result.product_name}")
        logger.info(f"Language: {result.language}")
        logger.info(f"Jurisdiction: {result.jurisdiction}")
        logger.info(f"Company Name: {result.company_name}")
        
        if result.error_message:
            logger.error(f"Error: {result.error_message}")
        
        logger.info("=== E2E Test Complete ===")
        
    except Exception as e:
        logger.exception(f"E2E Test Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
