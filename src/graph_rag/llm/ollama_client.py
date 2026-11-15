"""
Ollama Client module.

Wraps the Ollama API for:
- Text Generation (qwen2.5)
- Text Embeddings (nomic-embed-text)
- Health check and model availability
"""

import logging
from typing import Any, Dict, List, Optional
import httpx
import ollama

from graph_rag.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper for local Ollama LLM and Embedding interactions."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.llm_model = llm_model or settings.OLLAMA_MODEL
        self.embed_model = embed_model or settings.OLLAMA_EMBED_MODEL
        self.client = ollama.Client(
            host=self.base_url,
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
        self.keep_alive = settings.OLLAMA_KEEP_ALIVE

    def is_available(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List all models currently available in local Ollama."""
        try:
            models_info = self.client.list()
            # ollama-python returns ListResponse object with 'models'
            return [m.model for m in models_info.models]
        except Exception as e:
            logger.warning(f"Failed to query Ollama models: {e}")
            return []

    def missing_models(self, required_models: List[str]) -> List[str]:
        """Return configured models that are not installed locally."""
        def normalized_name(model: str) -> str:
            name, separator, tag = model.partition(":")
            return f"{name}:{tag if separator else 'latest'}"

        installed = {normalized_name(model) for model in self.list_models()}
        return [
            model
            for model in required_models
            if normalized_name(model) not in installed
        ]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: Optional[int] = None,
    ) -> str:
        """Generate a text response using Ollama LLM with automatic model fallback and retry logic."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try configured model, fallback to any available qwen or local model
        target_model = self.llm_model
        max_retries = max_retries if max_retries is not None else settings.OLLAMA_MAX_RETRIES
        
        # Initial options
        num_ctx = settings.OLLAMA_NUM_CTX
        num_predict = settings.OLLAMA_NUM_PREDICT
        
        for attempt in range(max_retries + 1):
            options = {
                "temperature": temperature,
                "num_gpu": settings.OLLAMA_NUM_GPU,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            }
            
            try:
                response = self.client.chat(
                    model=target_model,
                    messages=messages,
                    options=options,
                    keep_alive=self.keep_alive,
                )
                return response["message"]["content"].strip()
            except Exception as e:
                err_str = str(e).lower()
                # Check if 404 not found and attempt fallback
                if "not found" in err_str and attempt == 0:
                    available = self.list_models()
                    fallback = next((m for m in available if "qwen" in m.lower() or "deepseek" in m.lower()), None)
                    if fallback and fallback != target_model:
                        logger.info(f"Model '{target_model}' not found, falling back to installed '{fallback}'")
                        self.llm_model = fallback
                        target_model = fallback
                        continue
                    raise RuntimeError(
                        f"Ollama model not found: '{target_model}'. "
                        f"Install it with `ollama pull {target_model}`."
                    ) from e
                
                # Retry on timeout - halve context and output limits
                if "timeout" in err_str or "timed out" in err_str:
                    if attempt < max_retries:
                        logger.warning(f"Ollama request timed out (attempt {attempt + 1}/{max_retries + 1}), retrying with reduced context...")
                        num_ctx = num_ctx // 2
                        num_predict = num_predict // 2
                        logger.info(f"Reduced num_ctx to {num_ctx}, num_predict to {num_predict}")
                        continue
                    else:
                        logger.error(f"Ollama request failed after {max_retries + 1} attempts: {e}")
                        raise RuntimeError(f"Ollama request failed: timed out") from e
                
                # Other errors
                raise

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a single text chunk."""
        # Use nomic-embed-text
        try:
            res = self.client.embeddings(
                model=self.embed_model,
                prompt=text,
                options={"num_gpu": settings.OLLAMA_NUM_GPU},
            )
        except Exception as e:
            if "not found" in str(e).lower():
                raise RuntimeError(
                    f"Ollama model not found: '{self.embed_model}'. "
                    f"Install it with `ollama pull {self.embed_model}`."
                ) from e
            raise
        return res["embedding"]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of text chunks."""
        if not texts:
            return []

        # Ollama's embed endpoint accepts the complete batch in one request.
        # This avoids paying model-loading and HTTP overhead once per chunk.
        try:
            response = self.client.embed(
                model=self.embed_model,
                input=texts,
                options={"num_gpu": settings.OLLAMA_NUM_GPU},
            )
        except Exception as e:
            if "not found" in str(e).lower():
                raise RuntimeError(
                    f"Ollama model not found: '{self.embed_model}'. "
                    f"Install it with `ollama pull {self.embed_model}`."
                ) from e
            raise
        return [list(embedding) for embedding in response.embeddings]


ollama_client = OllamaClient()
