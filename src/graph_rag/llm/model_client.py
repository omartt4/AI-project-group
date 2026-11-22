"""Cloud model client used for generation and embeddings.

The implementation uses OpenRouter's OpenAI-compatible HTTP endpoints without
binding the application to a particular chat or embedding model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from graph_rag.config import settings
from graph_rag.llm.cost_tracker import cost_tracker

logger = logging.getLogger(__name__)


class ModelConfigurationError(RuntimeError):
    """Raised when required cloud model settings are missing."""


class CloudModelClient:
    """Small OpenRouter client bound to one chat role and/or embeddings."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        role: str = "chat",
    ) -> None:
        self.base_url = (base_url or settings.MODEL_API_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.MODEL_API_KEY
        self.llm_model = settings.CHAT_MODEL if llm_model is None else llm_model
        self.embed_model = (
            settings.EMBEDDING_MODEL if embed_model is None else embed_model
        )
        self.timeout_seconds = settings.MODEL_TIMEOUT_SECONDS
        self.max_retries = settings.MODEL_MAX_RETRIES
        self.role = role

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ModelConfigurationError(
                "OPENROUTER_API_KEY is missing. Add it to the local .env file."
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": settings.OPENROUTER_APP_NAME,
        }
        if settings.OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
        return headers

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        retryable_statuses = {408, 409, 429, 500, 502, 503, 504, 524, 529}

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(
                    base_url=self.base_url,
                    headers=self._headers(),
                    timeout=self.timeout_seconds,
                ) as client:
                    response = client.post(path, json=payload)
                if response.status_code in retryable_statuses and attempt < self.max_retries:
                    logger.warning(
                        "Cloud model request returned %s; retrying (%s/%s)",
                        response.status_code,
                        attempt + 1,
                        self.max_retries,
                    )
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("Cloud model returned a non-object response")
                return data
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in retryable_statuses or attempt >= self.max_retries:
                    break
                logger.warning(
                    "Cloud model request failed; retrying (%s/%s): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                logger.warning(
                    "Cloud model request failed; retrying (%s/%s): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )

        detail = ""
        if isinstance(last_error, httpx.HTTPStatusError):
            try:
                detail = f" Response: {last_error.response.text[:500]}"
            except Exception:
                pass
        raise RuntimeError(f"Cloud model request failed: {last_error}.{detail}")

    def is_available(self) -> bool:
        """Return whether the configured API is reachable with valid credentials."""
        if not self.api_key:
            return False
        try:
            with httpx.Client(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=5.0,
            ) as client:
                response = client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate one text response using the configured cloud chat model."""
        if not self.llm_model:
            raise ModelConfigurationError(
                "This client's OpenRouter chat model is missing. Configure the "
                "matching OPENROUTER_*_MODEL value in .env."
            )

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": temperature,
            "usage": {"include": True},
        }
        if response_format:
            payload["response_format"] = response_format

        data = self._post("/chat/completions", payload)
        cost_tracker.record(
            role=self.role,
            operation="generation",
            model=self.llm_model,
            endpoint="/chat/completions",
            usage=data.get("usage") or {},
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Cloud chat response did not contain message content") from exc
        if not isinstance(content, str):
            raise RuntimeError("Cloud chat response content was not text")
        return content.strip()

    def get_embedding(self, text: str) -> List[float]:
        """Create one embedding vector."""
        vectors = self.get_embeddings_batch([text])
        return vectors[0]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings in one ordered batch request."""
        if not texts:
            return []
        if not self.embed_model:
            raise ModelConfigurationError(
                "OPENROUTER_EMBED_MODEL is missing. Add an embedding model ID to .env."
            )

        payload: Dict[str, Any] = {
            "model": self.embed_model,
            "input": texts,
            "usage": {"include": True},
        }
        if settings.EMBEDDING_DIMENSIONS:
            payload["dimensions"] = settings.EMBEDDING_DIMENSIONS

        response = self._post("/embeddings", payload)
        cost_tracker.record(
            role=self.role,
            operation="embedding",
            model=self.embed_model,
            endpoint="/embeddings",
            usage=response.get("usage") or {},
        )
        rows = response.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise RuntimeError("Embedding response count did not match the input count")
        ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
        return [list(row["embedding"]) for row in ordered]


# Bind each pipeline responsibility to its own configured model. Keeping these
# as separate instances makes routing explicit at every call site and prevents
# extraction or evaluation from accidentally using the answer-generation model.
extraction_model_client = CloudModelClient(
    llm_model=settings.EXTRACTION_MODEL,
    embed_model="",
    role="extraction",
)
chat_model_client = CloudModelClient(
    llm_model=settings.CHAT_MODEL,
    embed_model="",
    role="chat",
)
evaluation_model_client = CloudModelClient(
    llm_model=settings.EVALUATION_MODEL,
    embed_model="",
    role="evaluation",
)
embedding_model_client = CloudModelClient(
    llm_model="",
    embed_model=settings.EMBEDDING_MODEL,
    role="embedding",
)

# Backward-compatible alias for external imports. New application code should
# import the role-specific client it actually needs.
model_client = chat_model_client

