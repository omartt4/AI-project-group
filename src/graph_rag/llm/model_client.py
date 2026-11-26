"""LangChain-backed model clients for configurable OpenRouter models."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from openrouter.sdk import OpenRouter
from openrouter.utils.retries import BackoffStrategy, RetryConfig

from graph_rag.config import settings
from graph_rag.llm.cost_tracker import cost_tracker
from graph_rag.llm.pricing import PRICING_VERSION, calculate_cost_usd

logger = logging.getLogger(__name__)


class ModelConfigurationError(RuntimeError):
    """Raised when required cloud model settings are missing."""


class ModelRequestContext:
    """Context shared by model calls participating in one user or ingest run."""

    def __init__(
        self,
        run_id: str | None = None,
        user_id: int | None = None,
        document_id: str | None = None,
        query_id: str | None = None,
        language: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.user_id = user_id
        self.document_id = document_id
        self.query_id = query_id
        self.language = language


_request_context: ContextVar[ModelRequestContext] = ContextVar(
    "model_request_context",
    default=ModelRequestContext(),
)


@contextmanager
def model_request_context(
    run_id: str | None = None,
    user_id: int | None = None,
    document_id: str | None = None,
    query_id: str | None = None,
    language: str | None = None,
):
    """Attach correlation identifiers to nested LangChain and embedding calls."""
    token = _request_context.set(
        ModelRequestContext(
            run_id=run_id,
            user_id=user_id,
            document_id=document_id,
            query_id=query_id,
            language=language,
        )
    )
    try:
        yield
    finally:
        _request_context.reset(token)


class ModelUsageCallbackHandler(BaseCallbackHandler):
    """Track usage and failures for LangChain model calls outside generate()."""

    run_inline = True

    def __init__(self, role: str, model: str) -> None:
        self.role = role
        self.model = model
        self.request_id = uuid.uuid4().hex
        self.context = _request_context.get()
        self.started: Dict[str, float] = {}
        self.response_records: List[Dict[str, Any]] = []

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.started[str(run_id)] = time.perf_counter()

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.started[str(run_id)] = time.perf_counter()

    def on_llm_end(self, response: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        usage = CloudModelClient._usage_from_response(response)
        generation = None
        generations = getattr(response, "generations", None)
        if generations and generations[0]:
            generation = generations[0][0]
        message = getattr(generation, "message", response)
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        if not content:
            additional_kwargs = getattr(message, "additional_kwargs", None) or {}
            tool_calls = getattr(message, "tool_calls", None) or additional_kwargs.get(
                "tool_calls",
                [],
            )
            if tool_calls:
                content = json.dumps(tool_calls, ensure_ascii=False, default=str)

        response_metadata = getattr(message, "response_metadata", None) or {}
        generation_info = getattr(generation, "generation_info", None) or {}
        finish_reason = (
            response_metadata.get("finish_reason")
            or generation_info.get("finish_reason")
            or response_metadata.get("stop_reason")
            or generation_info.get("stop_reason")
        )
        normalized_finish_reason = str(finish_reason or "").casefold()
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        try:
            input_tokens = int(input_tokens) if input_tokens is not None else None
        except (TypeError, ValueError, OverflowError):
            input_tokens = None
        try:
            output_tokens = int(output_tokens) if output_tokens is not None else None
        except (TypeError, ValueError, OverflowError):
            output_tokens = None
        calculation = calculate_cost_usd(
            self.model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        latency_ms = int(
            (time.perf_counter() - self.started.get(str(run_id), time.perf_counter()))
            * 1000
        )
        serializable_usage = json.loads(
            json.dumps(usage, ensure_ascii=False, default=str)
        )
        provider_cost = usage.get("cost")
        try:
            provider_cost = float(provider_cost) if provider_cost is not None else None
        except (TypeError, ValueError, OverflowError):
            provider_cost = None
        self.response_records.append(
            {
                "request_id": self.request_id,
                "role": self.role,
                "operation": "graph_extraction",
                "provider": settings.MODEL_API_BASE_URL,
                "endpoint": "/chat/completions",
                "model": self.model,
                "status": "success",
                "retry_count": None,
                "cache_status": "miss",
                "raw_model_output": content,
                "finish_reason": str(finish_reason) if finish_reason is not None else None,
                "truncated": (
                    normalized_finish_reason
                    in {"length", "max_tokens", "max_output_tokens"}
                    if finish_reason is not None
                    else None
                ),
                "usage": serializable_usage,
                "provider_cost_usd": provider_cost,
                "estimated_cost_usd": (
                    float(calculation.cost_usd)
                    if calculation.cost_usd is not None
                    else None
                ),
                "pricing_version": PRICING_VERSION,
                "latency_ms": max(0, latency_ms),
                "response_id": response_metadata.get("id"),
                "response_metadata": json.loads(
                    json.dumps(response_metadata, ensure_ascii=False, default=str)
                ),
                "generation_info": json.loads(
                    json.dumps(generation_info, ensure_ascii=False, default=str)
                ),
            }
        )
        if normalized_finish_reason in {
            "length",
            "max_tokens",
            "max_output_tokens",
        }:
            logger.warning(
                "Model output may be truncated: role=%s model=%s request_id=%s finish_reason=%s",
                self.role,
                self.model,
                self.request_id,
                finish_reason,
            )
        self._record(
            status="success",
            usage=usage,
            run_id=str(run_id),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self._record(
            status="failed",
            usage={},
            run_id=str(run_id),
            error_message=str(error),
        )

    def _record(
        self,
        *,
        status: str,
        usage: Dict[str, Any],
        run_id: str,
        error_message: str | None = None,
    ) -> None:
        started = self.started.pop(run_id, time.perf_counter())
        cost_tracker.record(
            role=self.role,
            operation="generation",
            model=self.model,
            endpoint="/chat/completions",
            usage=usage,
            request_id=self.request_id,
            run_id=self.context.run_id,
            user_id=self.context.user_id,
            document_id=self.context.document_id,
            query_id=self.context.query_id,
            language=self.context.language,
            latency_ms=int((time.perf_counter() - started) * 1000),
            status=status,
            error_message=error_message,
        )


class OpenRouterEmbeddings(Embeddings):
    """Expose the configured OpenRouter embedding endpoint to LangChain stores."""

    def __init__(self, client: "CloudModelClient") -> None:
        self.client = client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.get_embeddings_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.client.get_embedding(text)


class CloudModelClient:
    """Role-specific LangChain model adapter with backward-compatible methods."""

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
        self._chat_model_instance: Optional[ChatOpenRouter] = None
        self._embedding_model_instance: Optional[OpenAIEmbeddings] = None

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise ModelConfigurationError(
                "OPENROUTER_API_KEY is missing. Add it to the local .env file."
            )
        return self.api_key

    def _get_chat_model(self) -> ChatOpenRouter:
        if not self.llm_model:
            raise ModelConfigurationError(
                "This client's OpenRouter chat model is missing. Configure the "
                "matching OPENROUTER_*_MODEL value in .env."
            )
        if self._chat_model_instance is None:
            # Configure bounded exponential backoff and Retry-After awareness
            retry_config = RetryConfig(
                strategy="backoff",
                backoff=BackoffStrategy(
                    initial_interval=500,
                    max_interval=10000,
                    exponent=2.0,
                    max_elapsed_time=int(self.timeout_seconds * 1000),
                ),
                retry_connection_errors=True,
            )
            sdk_client = OpenRouter(
                api_key=self._require_api_key(),
                server_url=self.base_url,
                http_referer=settings.OPENROUTER_SITE_URL or None,
                x_open_router_title=settings.OPENROUTER_APP_NAME,
                retry_config=retry_config,
                timeout_ms=int(self.timeout_seconds * 1000),
            )
            self._chat_model_instance = ChatOpenRouter(
                model=self.llm_model,
                api_key=self._require_api_key(),
                client=sdk_client,
                base_url=self.base_url,
                temperature=0.2,
                timeout=int(self.timeout_seconds * 1000),
                max_retries=self.max_retries,
                app_url=settings.OPENROUTER_SITE_URL or None,
                app_title=settings.OPENROUTER_APP_NAME,
            )
        return self._chat_model_instance

    def _get_embedding_model(self) -> OpenAIEmbeddings:
        if not self.embed_model:
            raise ModelConfigurationError(
                "OPENROUTER_EMBED_MODEL is missing. Add an embedding model ID to .env."
            )
        if self._embedding_model_instance is None:
            headers = {"X-Title": settings.OPENROUTER_APP_NAME}
            if settings.OPENROUTER_SITE_URL:
                headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
            self._embedding_model_instance = OpenAIEmbeddings(
                model=self.embed_model,
                api_key=self._require_api_key(),
                base_url=self.base_url,
                dimensions=settings.EMBEDDING_DIMENSIONS or None,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
                default_headers=headers,
                tiktoken_enabled=False,
                check_embedding_ctx_length=False,
            )
        return self._embedding_model_instance

    def as_embeddings(self) -> Embeddings:
        """Return a lazy LangChain Embeddings adapter for vector-store integrations."""
        return OpenRouterEmbeddings(self)

    def as_chat_model(self) -> ChatOpenRouter:
        """Return the LangChain chat model for Runnable composition."""
        return self._get_chat_model()

    def is_available(self) -> bool:
        """Check OpenRouter's model-list endpoint without issuing a model request."""
        if not self.api_key or not (self.llm_model or self.embed_model):
            return False
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": settings.OPENROUTER_APP_NAME,
        }
        if settings.OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
        try:
            with httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=5.0,
            ) as client:
                return client.get("/models").status_code == 200
        except httpx.RequestError:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None,
        run_id: str | None = None,
        user_id: int | None = None,
    ) -> str:
        """Generate text through a LangChain prompt/model Runnable sequence."""
        cost_tracker.check_cost_limits()
        context = _request_context.get()
        request_id = uuid.uuid4().hex
        request_run_id = run_id or context.run_id
        request_user_id = user_id if user_id is not None else context.user_id
        request_doc_id = context.document_id
        request_query_id = context.query_id
        request_language = context.language
        started = time.perf_counter()
        try:
            model = self._get_chat_model().bind(temperature=temperature)
            if response_format:
                model = model.bind(response_format=response_format)

            messages = []
            if system_prompt:
                messages.append(("system", system_prompt))
            messages.append(("human", "{prompt}"))
            prompt_template = ChatPromptTemplate.from_messages(messages)
            chain = prompt_template | model
            response = chain.invoke(
                {"prompt": prompt},
                config={
                    "run_name": f"{self.role}-generation",
                    "metadata": {
                        "role": self.role,
                        "model": self.llm_model or "",
                        "run_id": request_run_id or "",
                        "user_id": request_user_id or "",
                        "document_id": request_doc_id or "",
                        "query_id": request_query_id or "",
                        "language": request_language or "",
                    },
                },
            )
            content = response.content
            if isinstance(content, list):
                content = "\n".join(
                    block["text"]
                    for block in content
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                )
            if not isinstance(content, str):
                raise RuntimeError("Cloud chat response content was not text")
        except Exception as exc:
            cost_tracker.record(
                role=self.role,
                operation="generation",
                model=self.llm_model or "",
                endpoint="/chat/completions",
                usage={},
                request_id=request_id,
                run_id=request_run_id,
                user_id=request_user_id,
                document_id=request_doc_id,
                query_id=request_query_id,
                language=request_language,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="failed",
                error_message=str(exc),
            )
            raise

        cost_tracker.record(
            role=self.role,
            operation="generation",
            model=self.llm_model or "",
            endpoint="/chat/completions",
            usage=self._usage_from_response(response),
            request_id=request_id,
            run_id=request_run_id,
            user_id=request_user_id,
            document_id=request_doc_id,
            query_id=request_query_id,
            language=request_language,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return content.strip()

    @staticmethod
    def _usage_from_response(response: Any) -> Dict[str, Any]:
        message = response
        generations = getattr(response, "generations", None)
        if generations and generations[0]:
            message = getattr(generations[0][0], "message", None)

        usage = dict(getattr(message, "usage_metadata", None) or {})
        metadata = getattr(message, "response_metadata", None) or {}
        output = getattr(response, "llm_output", None) or {}
        provider_usage = (
            metadata.get("token_usage")
            or metadata.get("usage")
            or output.get("token_usage")
            or output.get("usage")
            or {}
        )
        if isinstance(provider_usage, dict):
            for key, value in provider_usage.items():
                usage.setdefault(key, value)
        if "input_tokens" in usage:
            usage.setdefault("prompt_tokens", usage["input_tokens"])
        if "output_tokens" in usage:
            usage.setdefault("completion_tokens", usage["output_tokens"])
        if "cost" in metadata:
            usage.setdefault("cost", metadata["cost"])
        elif "cost" in output:
            usage.setdefault("cost", output["cost"])
        return usage

    def get_embedding(self, text: str) -> List[float]:
        """Create one embedding with the configured LangChain integration."""
        return self.get_embeddings_batch([text])[0]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create an ordered embedding batch with the configured integration."""
        if not texts:
            return []
        context = _request_context.get()
        try:
            model = self._get_embedding_model()
            model._ensure_sync_client_available()
        except Exception as exc:
            cost_tracker.record(
                role=self.role,
                operation="embedding",
                model=self.embed_model or "",
                endpoint="/embeddings",
                usage={},
                request_id=uuid.uuid4().hex,
                run_id=context.run_id,
                user_id=context.user_id,
                status="failed",
                error_message=str(exc),
            )
            raise

        cost_tracker.check_cost_limits()
        embeddings: List[List[float]] = []
        chunk_size = max(1, int(model.chunk_size))
        for start_index in range(0, len(texts), chunk_size):
            batch = texts[start_index : start_index + chunk_size]
            request_id = uuid.uuid4().hex
            started = time.perf_counter()
            try:
                response = model.client.create(
                    input=batch,
                    **model._invocation_params,
                )
                payload = (
                    response
                    if isinstance(response, dict)
                    else response.model_dump()
                )
                data = payload.get("data")
                if not isinstance(data, list) or len(data) != len(batch):
                    raise RuntimeError(
                        "Embedding response did not contain one vector per input"
                    )
                data.sort(key=lambda item: item.get("index", 0))
                embeddings.extend(item["embedding"] for item in data)
                response_usage = payload.get("usage") or {}
                input_tokens = response_usage.get(
                    "prompt_tokens",
                    response_usage.get(
                        "input_tokens",
                        response_usage.get("total_tokens"),
                    ),
                )
                usage = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": 0,
                    "total_tokens": response_usage.get("total_tokens", input_tokens),
                }
                if "cost" in response_usage:
                    usage["cost"] = response_usage["cost"]
            except Exception as exc:
                cost_tracker.record(
                    role=self.role,
                    operation="embedding",
                    model=self.embed_model or "",
                    endpoint="/embeddings",
                    usage={},
                    request_id=request_id,
                    run_id=context.run_id,
                    user_id=context.user_id,
                    document_id=context.document_id,
                    query_id=context.query_id,
                    language=context.language,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    status="failed",
                    error_message=str(exc),
                )
                raise
            cost_tracker.record(
                role=self.role,
                operation="embedding",
                model=self.embed_model or "",
                endpoint="/embeddings",
                usage=usage,
                request_id=request_id,
                run_id=context.run_id,
                user_id=context.user_id,
                document_id=context.document_id,
                query_id=context.query_id,
                language=context.language,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        return embeddings


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

# Backward-compatible alias for external imports.
model_client = chat_model_client
