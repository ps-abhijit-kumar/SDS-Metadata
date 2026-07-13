#!/usr/bin/env python
"""Quick test of embedding functionality."""

from langchain_ollama import OllamaEmbeddings

try:
    print("Testing Ollama embedding client...")
    emb = OllamaEmbeddings(
        model='nomic-embed-text:latest',
        base_url='http://127.0.0.1:11434'
    )
    result = emb.embed_documents(['test embedding'])
    print(f"✓ SUCCESS: Embedding works! Dimension: {len(result[0])}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")
