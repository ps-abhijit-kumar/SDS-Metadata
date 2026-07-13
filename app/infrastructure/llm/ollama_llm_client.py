"""Ollama LLM client using direct HTTP calls with proper timeout handling.

The application layer calls generate() with a prompt string and
receives the raw text response. No third-party wrapper overhead.
"""

from __future__ import annotations

import json
import logging

import httpx

from app.domain.exceptions.base import LLMException
from app.infrastructure.configuration.settings import Settings

logger = logging.getLogger(__name__)


class OllamaLLMClient:
    """Sends prompts to the locally running Ollama LLM and returns text responses."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.ollama_llm_model
        self._base_url = settings.ollama_base_url
        self._timeout = settings.ollama_timeout_seconds
        logger.info(
            "OllamaLLMClient ready | model=%s | base_url=%s | timeout=%ds",
            settings.ollama_llm_model,
            settings.ollama_base_url,
            settings.ollama_timeout_seconds,
        )

    def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response text.

        Raises:
            LLMException: if the Ollama server is unreachable or returns an error.
        """
        logger.debug("LLM generate | model=%s | prompt_len=%d", self._model, len(prompt))
        
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0,  # deterministic — metadata extraction is not creative
        }
        
        url = f"{self._base_url}/api/chat"
        
        try:
            # Create a client with explicit timeout (connect, read, write, pool)
            # Read timeout must be long for LLM inference
            timeout = httpx.Timeout(timeout=10.0, read=float(self._timeout))
            
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
            
            response.raise_for_status()
            data = response.json()
            
            # Extract the response text
            if "message" in data and "content" in data["message"]:
                text = data["message"]["content"]
            else:
                raise LLMException(f"Unexpected Ollama response format: {data}")
            
            logger.info("LLM response received | len=%d | first_100_chars=%s", len(text), text[:100])
            logger.debug("Full LLM response:\n%s", text)
            return text
            
        except httpx.TimeoutException as exc:
            raise LLMException(
                f"Ollama request timed out after {self._timeout}s (model={self._model}): {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMException(
                f"Ollama returned error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise LLMException(
                f"Ollama LLM call failed (model={self._model}): {exc}"
            ) from exc
