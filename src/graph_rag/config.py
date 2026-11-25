"""
Configuration module for Graph RAG Chatbot.

Loads all settings from .env file using python-dotenv.
Ensures privacy, security, and strict validation of application variables.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Resolve project root directory (two levels up from src/graph_rag)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file at project root
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

# Keep old local configurations working while the project migrates from one
# shared chat model to explicit task-specific model routing.
_LEGACY_LLM_MODEL = os.getenv("OPENROUTER_LLM_MODEL", "")


@dataclass(frozen=True)
class Settings:
    """Application configuration settings."""

    # ── Cloud Model Settings ──
    MODEL_API_BASE_URL: str = os.getenv(
        "MODEL_API_BASE_URL", "https://openrouter.ai/api/v1"
    )
    MODEL_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    EXTRACTION_MODEL: str = os.getenv(
        "OPENROUTER_EXTRACTION_MODEL", _LEGACY_LLM_MODEL
    )
    CHAT_MODEL: str = os.getenv("OPENROUTER_CHAT_MODEL", _LEGACY_LLM_MODEL)
    EVALUATION_MODEL: str = os.getenv(
        "OPENROUTER_EVAL_MODEL", _LEGACY_LLM_MODEL
    )
    EMBEDDING_MODEL: str = os.getenv("OPENROUTER_EMBED_MODEL", "")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "0"))
    MODEL_TIMEOUT_SECONDS: float = float(os.getenv("MODEL_TIMEOUT_SECONDS", "120.0"))
    MODEL_MAX_RETRIES: int = int(os.getenv("MODEL_MAX_RETRIES", "2"))
    LANGCHAIN_MAX_CONCURRENCY: int = int(
        os.getenv("LANGCHAIN_MAX_CONCURRENCY", "4")
    )
    OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "")
    OPENROUTER_APP_NAME: str = os.getenv("OPENROUTER_APP_NAME", "Graph RAG Studio")

    # ── Neo4j Settings ──
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv(
        "NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j")
    )
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password123")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # ── Paths & Storage ──
    PDF_DIR: Path = PROJECT_ROOT / os.getenv("PDF_DIR", "pdfs")
    METADATA_FILE: Path = PROJECT_ROOT / os.getenv("METADATA_FILE", "pdfs/metadata.csv")
    DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "research_papers_graph_v2"
    )
    SQLITE_DB_PATH: Path = PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "data/graph_rag.db")
    UPLOAD_TEMP_DIR: Path = PROJECT_ROOT / os.getenv("UPLOAD_TEMP_DIR", "data/temp_uploads")
    EXTRACTION_CACHE_DIR: Path = PROJECT_ROOT / os.getenv(
        "EXTRACTION_CACHE_DIR", "data/extraction_cache"
    )

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
    EXTRACTION_SEGMENT_MAX_CHARS: int = int(
        os.getenv("EXTRACTION_SEGMENT_MAX_CHARS", "40000")
    )
    EXTRACTION_SEGMENT_RETRY_DEPTH: int = int(
        os.getenv("EXTRACTION_SEGMENT_RETRY_DEPTH", "2")
    )
    EXTRACTION_MIN_CONFIDENCE: float = float(
        os.getenv("EXTRACTION_MIN_CONFIDENCE", "0.65")
    )
    TOPIC_MIN_CONFIDENCE: float = float(
        os.getenv("TOPIC_MIN_CONFIDENCE", "0.80")
    )
    OCR_DPI: int = int(os.getenv("OCR_DPI", "200"))
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "eng+ara")
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
        """Create required runtime directories if they don't exist."""
        if self.LANGCHAIN_MAX_CONCURRENCY < 1:
            raise ValueError("LANGCHAIN_MAX_CONCURRENCY must be at least 1")
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.EXTRACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.PDF_DIR.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories()
