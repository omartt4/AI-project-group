"""
Setup script for multilingual embedding model (BGE-M3).

This script helps pull the BGE-M3 model from Ollama for Arabic + English support.
BGE-M3 supports 100+ languages and is optimized for cross-lingual retrieval.

Usage:
    uv run python scripts/setup_multilingual_embeddings.py
"""

import sys
import logging
from graph_rag.llm.ollama_client import ollama_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_model_availability(model_name: str) -> bool:
    """Check if a model is available in Ollama."""
    available_models = ollama_client.list_models()
    return model_name in available_models or any(model_name in m for m in available_models)


def pull_model(model_name: str) -> bool:
    """Pull a model from Ollama registry."""
    try:
        import ollama
        logger.info(f"Pulling model '{model_name}' from Ollama...")
        ollama.pull(model_name)
        logger.info(f"Successfully pulled '{model_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to pull model '{model_name}': {e}")
        return False


def main():
    """Main setup routine."""
    target_model = "bge-m3"
    
    logger.info("=" * 60)
    logger.info("Multilingual Embedding Model Setup")
    logger.info("=" * 60)
    
    # Check Ollama availability
    if not ollama_client.is_available():
        logger.error("Ollama server is not running. Please start Ollama first.")
        logger.error("Run: ollama serve")
        sys.exit(1)
    
    logger.info(f"Ollama server is running at {ollama_client.base_url}")
    
    # Check if model already exists
    if check_model_availability(target_model):
        logger.info(f"✓ Model '{target_model}' is already available in Ollama")
        logger.info(f"Available models: {ollama_client.list_models()}")
        return
    
    # Pull the model
    logger.info(f"Model '{target_model}' not found. Pulling from Ollama registry...")
    logger.info("This may take a few minutes as BGE-M3 is ~2.2GB...")
    
    if pull_model(target_model):
        logger.info("=" * 60)
        logger.info("✓ Setup complete!")
        logger.info(f"✓ Model '{target_model}' is now available")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Update your .env file to set OLLAMA_EMBED_MODEL=bge-m3")
        logger.info("2. Re-index your documents with the new model:")
        logger.info("   uv run python scripts/ingest.py")
        logger.info("")
    else:
        logger.error("Failed to setup multilingual embedding model")
        sys.exit(1)


if __name__ == "__main__":
    main()
