"""Optional local Ollama generation and embedding client."""

import logging
from typing import Any, Dict, List, Optional

import httpx
import ollama

from graph_rag.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper for local Ollama LLM and embedding interactions."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.llm_model = llm_model or settings.OLLAMA_LLM_MODEL
        self.embed_model = embed_model or settings.OLLAMA_EMBED_MODEL
        self.timeout_retries = settings.OLLAMA_TIMEOUT_RETRIES
        self.client = ollama.Client(
            host=self.base_url,
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )

    def is_available(self) -> bool:
        """Return whether the configured Ollama service is reachable."""
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List locally available Ollama models."""
        try:
            models_info = self.client.list()
            return [model.model for model in models_info.models]
        except Exception as exc:
            logger.warning("Failed to query Ollama models: %s", exc)
            return []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate text, falling back to another installed local model."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        target_model = self.llm_model
        try:
            response = self._chat_with_timeout_retries(
                model=target_model,
                messages=messages,
                temperature=temperature,
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            if "not found" in str(exc).lower():
                available = self.list_models()
                fallback = next(
                    (
                        model
                        for model in available
                        if "qwen" in model.lower() or "deepseek" in model.lower()
                    ),
                    None,
                )
                if fallback and fallback != target_model:
                    logger.info(
                        "Model %r not found, falling back to installed %r",
                        target_model,
                        fallback,
                    )
                    self.llm_model = fallback
                    response = self._chat_with_timeout_retries(
                        model=fallback,
                        messages=messages,
                        temperature=temperature,
                    )
                    return response["message"]["content"].strip()
            raise

    def _chat_with_timeout_retries(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> Any:
        """Retry timed-out local generations."""
        for attempt in range(self.timeout_retries + 1):
            try:
                return self.client.chat(
                    model=model,
                    messages=messages,
                    options={"temperature": temperature},
                )
            except (httpx.TimeoutException, TimeoutError) as exc:
                if attempt >= self.timeout_retries:
                    raise
                logger.warning(
                    "Ollama request timed out; retrying (%d/%d): %s",
                    attempt + 1,
                    self.timeout_retries,
                    exc,
                )

        raise RuntimeError("Ollama request retry loop exited unexpectedly")

    def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for one text."""
        response = self.client.embeddings(model=self.embed_model, prompt=text)
        return response["embedding"]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        response = self.client.embed(model=self.embed_model, input=texts)
        return [list(embedding) for embedding in response.embeddings]


ollama_client = OllamaClient()
