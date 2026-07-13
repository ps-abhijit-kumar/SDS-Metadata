#!/usr/bin/env python3
"""Debug script to test LLM response parsing directly."""

import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.infrastructure.configuration.settings import get_settings
from app.infrastructure.llm.ollama_llm_client import OllamaLLMClient
from app.application.services.prompt_builder import build_extraction_prompt

# Sample test chunks
test_chunks = [
    "SAFETY DATA SHEET (SDS) Product Name: Acetic Acid Optima LC/MS Section 1: Identification Product Name: Acetic Acid, Optima LC/MS Grade Classification: Acute toxicity (oral), Acute toxicity (inhalation fumes)",
    "Section 15: Regulatory Information This SDS is prepared in accordance with Regulation (EU) No 1907/2006 (REACH) and Regulation (EU) No 1272/2008 (CLP) GHS Classification: Acute Tox. 3 (Oral), Acute Tox. 3 (Vapour) H values: H301, H311, H331",
    "The document language is English. Jurisdiction: European Union (REACH / CLP). Language: English SDS prepared according to REACH and CLP regulations.",
]

def main():
    print("=" * 80)
    print("SDS METADATA EXTRACTION - DEBUG SCRIPT")
    print("=" * 80)
    
    settings = get_settings()
    llm_client = OllamaLLMClient(settings)
    
    print("\n1. Building extraction prompt from test chunks...")
    prompt = build_extraction_prompt(test_chunks)
    print(f"Prompt length: {len(prompt)} characters")
    print("\nPrompt preview (first 500 chars):")
    print(prompt[:500])
    
    print("\n" + "=" * 80)
    print("2. Calling Ollama LLM (qwen3:8b)...")
    print("This may take several minutes...")
    print("=" * 80)
    
    try:
        response = llm_client.generate(prompt)
        print("\n" + "=" * 80)
        print("RAW LLM RESPONSE:")
        print("=" * 80)
        print(response)
        print("=" * 80)
        print(f"Response length: {len(response)} characters")
        print("=" * 80)
        
        # Now test the parser
        from app.application.services.metadata_validator import MetadataValidator
        validator = MetadataValidator()
        
        print("\n3. Attempting to parse the response...")
        try:
            metadata = validator.parse_and_validate("test-doc-id", response)
            print("\n✅ PARSING SUCCESSFUL!")
            print(f"  Product Name: {metadata.product_name}")
            print(f"  Language: {metadata.language}")
            print(f"  Jurisdiction: {metadata.jurisdiction}")
        except Exception as parse_exc:
            print(f"\n❌ PARSING FAILED: {parse_exc}")
            print("\nDebug: Testing individual regex patterns...")
            
            import re
            patterns = {
                "product_name": re.compile(r"^Product\s+Name\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
                "language": re.compile(r"^Language\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
                "jurisdiction": re.compile(r"^Jurisdiction\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
            }
            
            for field_name, pattern in patterns.items():
                match = pattern.search(response)
                if match:
                    print(f"  ✅ {field_name}: '{match.group(1)}'")
                else:
                    print(f"  ❌ {field_name}: NO MATCH")
            
    except Exception as exc:
        print(f"\n❌ LLM call failed: {exc}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
