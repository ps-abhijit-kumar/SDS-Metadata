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
from app.infrastructure.llm.ollama_health_check import OllamaHealthCheck

logger = logging.getLogger(__name__)


import json
import logging
import re
import httpx

from app.domain.exceptions.base import LLMException
from app.infrastructure.configuration.settings import Settings

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaLLMClient:
    """Sends prompts to the locally running Ollama LLM and returns text responses."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.metadata_model
        self._base_url = settings.ollama_base_url
        self._timeout = settings.ollama_timeout_seconds
        self._keep_alive = settings.ollama_keep_alive
        logger.info(
            "OllamaLLMClient ready | default_model=%s | base_url=%s | timeout=%ds | keep_alive=%s",
            self._model,
            self._base_url,
            self._timeout,
            self._keep_alive,
        )

    def generate(self, prompt: str, model: str | None = None) -> str:
        """Send a prompt to the LLM and return the response text."""
        target_model = model or self._model
        logger.info("METADATA EXTRACTION | model=%s | prompt_len=%d", target_model, len(prompt))

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": 0.0,
                "num_predict": 128,
                "think": False,  # Disable reasoning tokens if supported by model/Ollama
            },
        }

        url = f"{self._base_url}/api/chat"

        try:
            timeout = httpx.Timeout(timeout=5.0, read=float(self._timeout))

            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)

            response.raise_for_status()
            data = response.json()

            if "message" in data and "content" in data["message"]:
                raw_text = data["message"]["content"]
            else:
                raise LLMException(f"Unexpected Ollama response format: {data}")

            # Strip any residual <think>...</think> reasoning blocks if present
            cleaned_text = _THINK_RE.sub("", raw_text).strip()

            logger.info("Metadata LLM response received | model=%s | len=%d", target_model, len(cleaned_text))
            return cleaned_text

        except httpx.TimeoutException as exc:
            raise LLMException(
                f"Ollama request timed out after {self._timeout}s (model={target_model}): {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMException(
                f"Cannot connect to Ollama at {self._base_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMException(
                f"Ollama returned error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise LLMException(
                f"Ollama LLM call failed (model={target_model}): {exc}"
            ) from exc
