"""
Configuration module for Graph RAG system.

Loads settings from environment variables and provides a global settings singleton.
"""

import os
import logging
from pathlib import Path
from typing import Optional

# Determine project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class Settings:
    """Application configuration loaded from environment variables."""

    # ── Ollama Settings ──
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
    OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180.0"))
    OLLAMA_TIMEOUT_RETRIES: int = int(os.getenv("OLLAMA_TIMEOUT_RETRIES", "2"))

    # ── Neo4j Settings ──
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password123")

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
    
    sqlite_path_env = os.getenv("SQLITE_DB_PATH", "data/graph_rag.db")
    SQLITE_DB_PATH: Path = Path(sqlite_path_env) if Path(sqlite_path_env).is_absolute() else PROJECT_ROOT / sqlite_path_env
    
    upload_temp_env = os.getenv("UPLOAD_TEMP_DIR", "data/temp_uploads")
    UPLOAD_TEMP_DIR: Path = Path(upload_temp_env) if Path(upload_temp_env).is_absolute() else PROJECT_ROOT / upload_temp_env

    # ── Ingestion & Document Processing ──
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))

    # ── JWT Authentication ──
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-key-graph-rag-2026-super-secure")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

    # ── App & Logging ──
    APP_HOST: str = os.getenv("APP_HOST", "localhost")
    APP_PORT: int = int(os.getenv("APP_PORT", "8501"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    APP_LOG_FILE: Path = PROJECT_ROOT / os.getenv("APP_LOG_FILE", "data/app.log")

    def ensure_directories(self) -> None:
        """Create required runtime directories if they don't exist."""
        logger = logging.getLogger(__name__)
        
        # Make all directory creations optional to handle disk space issues
        directories = [
            ("DATA_DIR", self.DATA_DIR),
            ("CHROMA_PERSIST_DIR", self.CHROMA_PERSIST_DIR),
            ("SQLITE_DB_PATH.parent", self.SQLITE_DB_PATH.parent),
            ("PDF_DIR", self.PDF_DIR),
            ("UPLOAD_TEMP_DIR", self.UPLOAD_TEMP_DIR),
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
