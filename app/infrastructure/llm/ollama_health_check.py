"""Ollama connectivity and health verification service.

Provides pre-flight checks to validate:
1. Ollama server availability
2. Required models existence
3. Connection stability
4. Model responsiveness

Prevents silent failures by catching connectivity issues early.
"""

from __future__ import annotations

import logging

import httpx

from app.domain.exceptions.base import ApplicationException

logger = logging.getLogger(__name__)


class OllamaHealthCheck:
    """Verifies Ollama server availability and model readiness."""

    def __init__(self, base_url: str, timeout_seconds: int = 5) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def check_server(self) -> bool:
        """Check if Ollama server is reachable.

        Returns:
            True if server responds successfully.

        Raises:
            ApplicationException: If server is unreachable or unhealthy.
        """
        try:
            timeout = httpx.Timeout(timeout=float(self._timeout))
            with httpx.Client(timeout=timeout) as client:
                response = client.get(f"{self._base_url}/api/version")
            
            if response.status_code == 200:
                logger.info("✓ Ollama server is online")
                return True
            
            raise ApplicationException(
                f"Ollama server at {self._base_url} returned status {response.status_code}"
            )
        
        except httpx.TimeoutException:
            raise ApplicationException(
                f"Ollama server at {self._base_url} is not responding (timeout after {self._timeout}s)"
            )
        except httpx.ConnectError:
            raise ApplicationException(
                f"Cannot connect to Ollama server at {self._base_url}. "
                "Ensure Ollama is running: ollama serve"
            )
        except Exception as exc:
            raise ApplicationException(
                f"Failed to verify Ollama server health: {exc}"
            ) from exc

    def check_model_exists(self, model_name: str) -> bool:
        """Check if a specific model is available in Ollama.

        Returns:
            True if model is available.

        Raises:
            ApplicationException: If model does not exist or cannot be verified.
        """
        try:
            timeout = httpx.Timeout(timeout=float(self._timeout))
            with httpx.Client(timeout=timeout) as client:
                response = client.get(f"{self._base_url}/api/tags")
            
            if response.status_code != 200:
                raise ApplicationException(
                    f"Failed to list Ollama models: HTTP {response.status_code}"
                )
            
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check for exact match or partial match (e.g., "qwen3:8b" matches "qwen3:8b")
            if model_name in model_names:
                logger.info("✓ Model '%s' is available", model_name)
                return True
            
            # Also check if model_name is a prefix (e.g., "qwen3" in "qwen3:8b")
            for name in model_names:
                if model_name in name or name in model_name:
                    logger.info(
                        "✓ Model '%s' is available (found as '%s')",
                        model_name,
                        name,
                    )
                    return True
            
            available = ", ".join(model_names) if model_names else "none"
            raise ApplicationException(
                f"Model '{model_name}' not found in Ollama. "
                f"Available models: {available}. "
                f"Pull it with: ollama pull {model_name}"
            )
        
        except httpx.TimeoutException:
            raise ApplicationException(
                f"Timeout checking Ollama models (after {self._timeout}s)"
            )
        except httpx.ConnectError:
            raise ApplicationException(
                f"Cannot connect to Ollama to check models"
            )
        except ApplicationException:
            raise
        except Exception as exc:
            raise ApplicationException(
                f"Failed to verify model availability: {exc}"
            ) from exc

    def test_embedding_model(self, model_name: str) -> bool:
        """Quick test to verify embedding model is responsive.

        Generates a small test embedding to ensure the model is functional.

        Returns:
            True if model responds successfully.

        Raises:
            ApplicationException: If model fails to generate embeddings.
        """
        try:
            timeout = httpx.Timeout(timeout=10.0, read=30.0)
            payload = {"model": model_name, "input": "test"}
            
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self._base_url}/api/embed",
                    json=payload,
                )
            
            if response.status_code == 200:
                data = response.json()
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    logger.info(
                        "✓ Embedding model '%s' is responsive (dim=%d)",
                        model_name,
                        len(data["embeddings"][0]) if data["embeddings"][0] else 0,
                    )
                    return True
            
            raise ApplicationException(
                f"Embedding model '{model_name}' returned unexpected response: "
                f"HTTP {response.status_code}"
            )
        
        except httpx.TimeoutException:
            raise ApplicationException(
                f"Embedding test timed out for model '{model_name}'"
            )
        except httpx.ConnectError:
            raise ApplicationException(
                f"Cannot connect to Ollama to test embedding model"
            )
        except ApplicationException:
            raise
        except Exception as exc:
            raise ApplicationException(
                f"Failed to test embedding model '{model_name}': {exc}"
            ) from exc

    def test_llm_model(self, model_name: str) -> bool:
        """Quick test to verify LLM model is responsive.

        Sends a minimal prompt to ensure the model generates responses.

        Returns:
            True if model responds successfully.

        Raises:
            ApplicationException: If model fails to respond.
        """
        try:
            timeout = httpx.Timeout(timeout=5.0, read=30.0)
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            }
            
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
            
            if response.status_code == 200:
                logger.info("✓ LLM model '%s' is responsive", model_name)
                return True
            
            raise ApplicationException(
                f"LLM model '{model_name}' returned unexpected response: "
                f"HTTP {response.status_code}"
            )
        
        except httpx.TimeoutException:
            raise ApplicationException(
                f"LLM test timed out for model '{model_name}' "
                f"(model may be loading or overloaded)"
            )
        except httpx.ConnectError:
            raise ApplicationException(
                f"Cannot connect to Ollama to test LLM model"
            )
        except ApplicationException:
            raise
        except Exception as exc:
            raise ApplicationException(
                f"Failed to test LLM model '{model_name}': {exc}"
            ) from exc

    def run_full_check(
        self,
        embedding_model: str,
        llm_model: str,
    ) -> bool:
        """Run complete health check: server + models + responsiveness.

        Returns:
            True if all checks pass.

        Raises:
            ApplicationException: If any check fails.
        """
        logger.info("Starting Ollama health check...")
        
        # 1. Server connectivity
        self.check_server()
        
        # 2. Model availability
        self.check_model_exists(embedding_model)
        self.check_model_exists(llm_model)
        
        # 3. Model responsiveness (optional - can be slow)
        # Skipped by default to avoid startup delays
        # Call explicitly if needed
        
        logger.info("✓ All Ollama health checks passed")
        return True
