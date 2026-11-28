"""
Configuration module for Graph RAG system.

Loads settings from environment variables and provides a global settings singleton.
"""

import os
import logging
from pathlib import Path
from typing import Optional

# config.py lives at <project>/src/graph_rag/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep old local configurations working while the project migrates from one
# shared chat model to explicit task-specific model routing.
_LEGACY_LLM_MODEL = os.getenv("OPENROUTER_LLM_MODEL", "")


class Settings:
    """Application configuration loaded from environment variables."""

    def __init__(self, **overrides: object) -> None:
        """Optionally override declared settings for tests or scoped runtimes."""
        unknown = [name for name in overrides if not hasattr(type(self), name)]
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown setting override(s): {names}")
        for name, value in overrides.items():
            setattr(self, name, value)

    # ── Cloud Model Settings ──
    MODEL_API_BASE_URL: str = os.getenv(
        "MODEL_API_BASE_URL", "https://openrouter.ai/api/v1"
    )
    MODEL_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    EXTRACTION_MODEL: str = os.getenv(
        "OPENROUTER_EXTRACTION_MODEL",
        "google/gemma-4-26b-a4b-it:free"
        if not _LEGACY_LLM_MODEL
        else _LEGACY_LLM_MODEL,
    )
    CHAT_MODEL: str = os.getenv(
        "OPENROUTER_CHAT_MODEL",
        "nvidia/nemotron-3.5-lightning:free"
        if not _LEGACY_LLM_MODEL
        else _LEGACY_LLM_MODEL,
    )
    EVALUATION_MODEL: str = os.getenv(
        "OPENROUTER_EVAL_MODEL",
        "nvidia/nemotron-3.5-super-vl:free"
        if not _LEGACY_LLM_MODEL
        else _LEGACY_LLM_MODEL,
    )
    EMBEDDING_MODEL: str = os.getenv(
        "OPENROUTER_EMBED_MODEL", "nvidia/nemotron-3.5-embed-8b:free"
    )
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "4096"))
    MODEL_TIMEOUT_SECONDS: float = float(os.getenv("MODEL_TIMEOUT_SECONDS", "120.0"))
    MODEL_MAX_RETRIES: int = int(os.getenv("MODEL_MAX_RETRIES", "2"))
    LANGCHAIN_MAX_CONCURRENCY: int = int(
        os.getenv("LANGCHAIN_MAX_CONCURRENCY", "4")
    )
    OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "")
    OPENROUTER_APP_NAME: str = os.getenv("OPENROUTER_APP_NAME", "Graph RAG Studio")

    # ── LangSmith & Observability Settings ──
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
    LANGCHAIN_API_KEY: str = os.getenv("LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "graph-rag")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    # ── Cost Limits & Budget Thresholds ──
    HARD_COST_LIMIT_USD: float = float(os.getenv("HARD_COST_LIMIT_USD", "10.0"))
    COST_WARNING_THRESHOLD_USD: float = float(os.getenv("COST_WARNING_THRESHOLD_USD", "5.0"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()

    # ── Optional local Ollama Settings ──
    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    # Backward-compatible alias for older environment files.
    OLLAMA_HOST: str = OLLAMA_BASE_URL
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
    OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180.0"))
    OLLAMA_TIMEOUT_RETRIES: int = int(os.getenv("OLLAMA_TIMEOUT_RETRIES", "2"))

    # ── Neo4j Settings ──
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv(
        "NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j")
    )
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password123")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # ── Paths & Storage ──
    # Support both relative and absolute paths
    pdf_dir_env = os.getenv("PDF_DIR", "pdfs")
    PDF_DIR: Path = Path(pdf_dir_env) if Path(pdf_dir_env).is_absolute() else PROJECT_ROOT / pdf_dir_env
    
    metadata_file_env = os.getenv("METADATA_FILE", "pdfs/metadata.csv")
    METADATA_FILE: Path = Path(metadata_file_env) if Path(metadata_file_env).is_absolute() else PROJECT_ROOT / metadata_file_env
    
    data_dir_env = os.getenv("DATA_DIR", "data")
    DATA_DIR: Path = Path(data_dir_env) if Path(data_dir_env).is_absolute() else PROJECT_ROOT / data_dir_env
    
    chroma_dir_env = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    CHROMA_PERSIST_DIR: Path = Path(chroma_dir_env) if Path(chroma_dir_env).is_absolute() else PROJECT_ROOT / chroma_dir_env
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "research_papers_graph_v2"
    )
    
    sqlite_path_env = os.getenv("SQLITE_DB_PATH", "data/graph_rag.db")
    SQLITE_DB_PATH: Path = Path(sqlite_path_env) if Path(sqlite_path_env).is_absolute() else PROJECT_ROOT / sqlite_path_env
    
    upload_temp_env = os.getenv("UPLOAD_TEMP_DIR", "data/temp_uploads")
    UPLOAD_TEMP_DIR: Path = Path(upload_temp_env) if Path(upload_temp_env).is_absolute() else PROJECT_ROOT / upload_temp_env

    extraction_cache_env = os.getenv("EXTRACTION_CACHE_DIR", "data/extraction_cache")
    EXTRACTION_CACHE_DIR: Path = Path(extraction_cache_env) if Path(extraction_cache_env).is_absolute() else PROJECT_ROOT / extraction_cache_env

    # ── Ingestion & Document Processing ──
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    GRAPH_EXTRACTION_MAX_CHARS: int = int(
        os.getenv("GRAPH_EXTRACTION_MAX_CHARS", "14000")
    )
    EXTRACTION_CONTEXT_TOKENS: int = int(
        os.getenv("EXTRACTION_CONTEXT_TOKENS", "131072")
    )
    EXTRACTION_OUTPUT_TOKENS: int = int(
        os.getenv("EXTRACTION_OUTPUT_TOKENS", "8192")
    )
    EXTRACTION_REASONING_ENABLED: bool = os.getenv(
        "EXTRACTION_REASONING_ENABLED", "false"
    ).strip().lower() in ("true", "1", "yes", "on")
    EXTRACTION_SEGMENT_MAX_CHARS: int = int(
        os.getenv("EXTRACTION_SEGMENT_MAX_CHARS", "20000")
    )
    EXTRACTION_SEGMENT_RETRY_DEPTH: int = int(
        os.getenv("EXTRACTION_SEGMENT_RETRY_DEPTH", "2")
    )
    EXTRACTION_BATCH_MAX_FAILURE_RATIO: float = float(
        os.getenv("EXTRACTION_BATCH_MAX_FAILURE_RATIO", "0.50")
    )
    EXTRACTION_MIN_CONFIDENCE: float = float(
        os.getenv("EXTRACTION_MIN_CONFIDENCE", "0.65")
    )
    TOPIC_MIN_CONFIDENCE: float = float(
        os.getenv("TOPIC_MIN_CONFIDENCE", "0.80")
    )
    OCR_DPI: int = int(os.getenv("OCR_DPI", "200"))
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "eng+ara")
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "paddle").strip().lower()
    OCR_FALLBACK_ENGINE: str = os.getenv(
        "OCR_FALLBACK_ENGINE", "tesseract"
    ).strip().lower()
    PADDLE_OCR_LANGUAGE: str = os.getenv("PADDLE_OCR_LANGUAGE", "ar").strip()
    PADDLE_OCR_URL: str = os.getenv("PADDLE_OCR_URL", "").strip().rstrip("/")
    PADDLE_OCR_TIMEOUT_SECONDS: float = float(
        os.getenv("PADDLE_OCR_TIMEOUT_SECONDS", "300")
    )
    SURYA_OCR_COMMAND: str = os.getenv("SURYA_OCR_COMMAND", "surya_ocr").strip()
    SURYA_OCR_TIMEOUT_SECONDS: float = float(
        os.getenv("SURYA_OCR_TIMEOUT_SECONDS", "300")
    )
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
    GRAPH_SCHEMA_VERSION: str = os.getenv("GRAPH_SCHEMA_VERSION", "4")
    EXTRACTION_VERSION: str = os.getenv(
        "EXTRACTION_VERSION", "document-open-graph-v6"
    )
    EXTRACTION_PROMPT_VERSION: str = os.getenv(
        "EXTRACTION_PROMPT_VERSION", "open-schema-document-v1"
    )
    NORMALIZATION_VERSION: str = os.getenv(
        "NORMALIZATION_VERSION", "open-normalization-v2"
    )
    DATASET_ID: str = os.getenv("DATASET_ID", "research-v2")

    # ── JWT Authentication ──
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-key-graph-rag-2026-super-secure")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

    # ── RAG Evaluation ──
    RAG_EVAL_ENABLED: bool = os.getenv("RAG_EVAL_ENABLED", "false").lower() in ("true", "1", "yes")
    # ── App & Logging ──
    APP_HOST: str = os.getenv("APP_HOST", "localhost")
    APP_PORT: int = int(os.getenv("APP_PORT", "8501"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    APP_LOG_FILE: Path = PROJECT_ROOT / os.getenv("APP_LOG_FILE", "data/app.log")

    def ensure_directories(self) -> None:
        """Create required runtime directories if they don't exist and validate env."""
        if self.LANGCHAIN_MAX_CONCURRENCY < 1:
            raise ValueError("LANGCHAIN_MAX_CONCURRENCY must be at least 1")
        
        # Enforce secure configuration in production
        if self.ENVIRONMENT in ("production", "prod"):
            if "dev-secret-key" in self.JWT_SECRET_KEY or len(self.JWT_SECRET_KEY) < 32:
                raise ValueError("In production, JWT_SECRET_KEY must be a secure random key with at least 32 characters.")
            if self.NEO4J_PASSWORD in ("password123", ""):
                raise ValueError("In production, default NEO4J_PASSWORD is not allowed.")
            if not self.MODEL_API_KEY:
                raise ValueError("In production, MODEL_API_KEY / OPENROUTER_API_KEY must be set.")

        # LangSmith setup
        if self.LANGCHAIN_TRACING_V2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if self.LANGCHAIN_API_KEY:
                os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            if self.LANGCHAIN_PROJECT:
                os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT
            if self.LANGCHAIN_ENDPOINT:
                os.environ["LANGCHAIN_ENDPOINT"] = self.LANGCHAIN_ENDPOINT

        logger = logging.getLogger(__name__)

        # Keep startup usable when an optional drive/path is temporarily absent.
        directories = [
            ("DATA_DIR", self.DATA_DIR),
            ("CHROMA_PERSIST_DIR", self.CHROMA_PERSIST_DIR),
            ("SQLITE_DB_PATH.parent", self.SQLITE_DB_PATH.parent),
            ("PDF_DIR", self.PDF_DIR),
            ("UPLOAD_TEMP_DIR", self.UPLOAD_TEMP_DIR),
            ("EXTRACTION_CACHE_DIR", self.EXTRACTION_CACHE_DIR),
        ]

        for name, path in directories:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(f"Could not create {name} at {path}: {e}")
                logger.warning("Some functionality may be limited due to disk space issues.")


# Global settings singleton
settings = Settings()
settings.ensure_directories()
