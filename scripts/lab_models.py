"""
Verify OpenRouter model IDs are available and free.
Embedding models are checked against /embeddings/models, chat models against /models.
Every lab script should call this check first.
"""

import argparse
import sys
from typing import Dict, Any, List
import requests

from lab_common import get_config, print_section, print_error


def get_openrouter_chat_models(base_url: str, api_key: str) -> Dict[str, Any]:
    """Fetch available chat models from OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/omartt4/AI-project-group",
        "X-Title": "Graph RAG Lab"
    }
    
    response = requests.get(f"{base_url}/models", headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # Build dict by id for easy lookup
    models = {}
    for model in data.get("data", []):
        models[model["id"]] = model
    
    return models


def get_openrouter_embedding_models(base_url: str, api_key: str) -> Dict[str, Any]:
    """Fetch available embedding models from OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/omartt4/AI-project-group",
        "X-Title": "Graph RAG Lab"
    }
    
    response = requests.get(f"{base_url}/embeddings/models", headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # Build dict by id for easy lookup
    models = {}
    for model in data.get("data", []):
        models[model["id"]] = model
    
    return models


def verify_embedding_model(model_id: str, embed_models: Dict[str, Any]) -> List[str]:
    """
    Verify an embedding model exists and is free.
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    if model_id not in embed_models:
        errors.append(f"Model '{model_id}' not found in embedding catalog")
        return errors
    
    model = embed_models[model_id]
    pricing = model.get("pricing", {})
    
    # Check if pricing is 0
    prompt_price = pricing.get("prompt")
    if prompt_price is None or float(prompt_price) > 0:
        errors.append(f"Model '{model_id}' is not free: prompt price = ${prompt_price}")
    
    # Print model info
    print(f"✓ {model_id}")
    
    # Context length
    context = model.get("context_length")
    if context:
        print(f"  Context: {context:,} tokens")
    
    # Pricing info
    if pricing:
        print(f"  Pricing: {pricing}")
    
    return errors


def verify_chat_model(model_id: str, chat_models: Dict[str, Any]) -> List[str]:
    """
    Verify a chat model exists and is free.
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    if model_id not in chat_models:
        errors.append(f"Model '{model_id}' not found in chat catalog")
        return errors
    
    model = chat_models[model_id]
    pricing = model.get("pricing", {})
    
    # Check if pricing is 0 for all tiers
    for tier in ["prompt", "completion"]:
        price = pricing.get(tier)
        if price is not None and float(price) > 0:
            errors.append(f"Model '{model_id}' is not free: {tier} = ${price}")
    
    # Print model info
    print(f"✓ {model_id}")
    
    # Context length
    context = model.get("context_length")
    if context:
        print(f"  Context: {context:,} tokens")
    
    # Pricing info
    if pricing:
        print(f"  Pricing: {pricing}")
    
    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify OpenRouter model IDs")
    parser.add_argument("--three-models", action="store_true", help="Also check nvidia/nemotron-3-embed-1b:free")
    args = parser.parse_args()
    
    config = get_config()
    
    if not config.openrouter_api_key:
        print_error("OPENROUTER_API_KEY not set in .env")
        sys.exit(1)
    
    print_section("Fetching OpenRouter Models")
    
    try:
        chat_models = get_openrouter_chat_models(config.openrouter_base_url, config.openrouter_api_key)
        embed_models = get_openrouter_embedding_models(config.openrouter_base_url, config.openrouter_api_key)
        print(f"Found {len(chat_models)} chat models and {len(embed_models)} embedding models\n")
    except Exception as e:
        print_error(f"Failed to fetch models: {e}")
        sys.exit(1)
    
    print_section("Verifying Configured Models")
    
    all_errors = []
    
    # Verify embedding models
    print("\nEmbedding models:")
    for model_id in [config.embed_model, config.embed_model_alt]:
        if not model_id:
            continue
        errors = verify_embedding_model(model_id, embed_models)
        all_errors.extend(errors)
    
    if args.three_models:
        third_model = "nvidia/nemotron-3-embed-1b:free"
        errors = verify_embedding_model(third_model, embed_models)
        all_errors.extend(errors)
    
    # Verify chat model
    print("\nChat models:")
    if config.chat_model:
        errors = verify_chat_model(config.chat_model, chat_models)
        all_errors.extend(errors)
    
    # Verify rerank model if specified
    if config.rerank_model:
        errors = verify_chat_model(config.rerank_model, chat_models)
        all_errors.extend(errors)
    
    # Report all errors
    print()
    if all_errors:
        print_error("Model verification failed:")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("✓ All configured models are available and free")
        sys.exit(0)


if __name__ == "__main__":
    main()
