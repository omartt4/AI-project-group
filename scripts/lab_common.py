"""
Shared configuration and utilities for lab retrieval pipeline.
All tunable parameters live in the LabConfig dataclass, loaded from .env with defaults.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


@dataclass
class LabConfig:
    """Central configuration for lab retrieval pipeline."""
    
    # Paths
    pdf_dir: str = field(default_factory=lambda: os.getenv("PDF_DIR", "pdfs"))
    data_dir: str = "data/lab"
    eval_dir: str = "eval_lab"
    chroma_dir: str = "data/chroma_lab"
    
    # PDF parsing
    pdf_library: str = "pymupdf"  # "pymupdf" or "pypdf"
    
    # Chunking
    chunk_unit: str = "tokens"  # "tokens" or "chars"
    chunk_size: int = 512
    chunk_overlap: int = 51  # 10% of 512
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    
    # Embedding models (OpenRouter)
    embed_model: str = field(default_factory=lambda: os.getenv("LAB_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b:free"))
    embed_model_alt: str = field(default_factory=lambda: os.getenv("LAB_EMBED_MODEL_ALT", "liquid/lfm-2.5-embedding-350m:free"))
    chat_model: str = field(default_factory=lambda: os.getenv("LAB_CHAT_MODEL", "nvidia/nemotron-3.5-lightning:free"))
    qgen_model: str = field(default_factory=lambda: os.getenv("LAB_QGEN_MODEL", "google/gemma-4-31b-it:free"))
    rerank_model: str = field(default_factory=lambda: os.getenv("LAB_RERANK_MODEL", ""))
    
    # API settings
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    requests_per_min: int = field(default_factory=lambda: int(os.getenv("LAB_REQUESTS_PER_MIN", "1")))  # 1 per 60s
    batch_size: int = 1  # Single chunk per request to minimize rate limit issues
    
    # Embedding parameters
    embed_dim: Optional[int] = field(default_factory=lambda: int(os.getenv("LAB_EMBED_DIM", "")) if os.getenv("LAB_EMBED_DIM") else None)
    query_prefix: str = field(default_factory=lambda: os.getenv("LAB_QUERY_PREFIX", ""))
    
    # Retrieval
    top_k: int = 5
    
    # Entity extraction
    extract_entities: bool = field(default_factory=lambda: os.getenv("LAB_EXTRACT_ENTITIES", "true").lower() == "true")
    entity_extractor: str = field(default_factory=lambda: os.getenv("LAB_ENTITY_EXTRACTOR", "yake"))  # "yake", "spacy_noun_chunks", "keybert", "llm_open"
    max_entities_per_chunk: int = field(default_factory=lambda: int(os.getenv("LAB_MAX_ENTITIES_PER_CHUNK", "8")))
    max_entities_per_paper: int = 50
    min_paper_freq: int = 2  # Terms appearing in fewer papers are dropped
    ngram_max: int = 3  # Max n-gram size for YAKE
    entity_boost: float = field(default_factory=lambda: float(os.getenv("LAB_ENTITY_BOOST", "0.15")))
    
    def __post_init__(self):
        # Convert paths to absolute
        self.pdf_dir = str(Path(self.pdf_dir).absolute())
        self.data_dir = str(Path(self.data_dir).absolute())
        self.eval_dir = str(Path(self.eval_dir).absolute())
        self.chroma_dir = str(Path(self.chroma_dir).absolute())
        
        # Create directories
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.eval_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)


def get_config() -> LabConfig:
    """Get the singleton config instance."""
    return LabConfig()


def parse_field_from_filename(filename: str) -> Optional[str]:
    r"""
    Parse field from PDF filename using regex: ^\d+_(NLP|Computer_Vision|Machine_Learning)_
    Returns: 'nlp', 'cv', or 'ml', or None if no match.
    """
    pattern = r'^\d+_(NLP|Computer_Vision|Machine_Learning)_'
    match = re.match(pattern, filename)
    if not match:
        return None
    
    field_map = {
        "NLP": "nlp",
        "Computer_Vision": "cv",
        "Machine_Learning": "ml"
    }
    return field_map.get(match.group(1))



def get_collection_name(model_tag: str, chunk_size: int, chunk_overlap: int, mode: str) -> str:
    """Generate Chroma collection name."""
    safe_tag = model_tag.replace("/", "_").replace(":", "_")
    return f"retrieval_lab_{safe_tag}_{chunk_size}_{chunk_overlap}_{mode}"

def print_warning(msg: str):
    """Print a warning message."""
    print(f"\n⚠️  WARNING: {msg}\n")


def print_error(msg: str):
    """Print an error message."""
    print(f"\n❌ ERROR: {msg}\n")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")
