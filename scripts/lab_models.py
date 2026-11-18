"""
Verify OpenRouter model IDs are available and free.
Embedding models are checked against /embeddings/models, chat models against /models.
Every lab script should call this check first.
"""

import argparse
import sys
from typing import Dict, Any, List
import requests
from datetime import datetime, timedelta
from pathlib import Path
import json

from lab_common import get_config, print_section, print_error


def check_chat_quota(base_url: str, api_key: str, data_dir: str) -> Dict[str, Any]:
    """
    Check chat quota by reading key info and chat_requests.jsonl log.
    Returns dict with quota info including full key_info.
    """
    result = {
        "key_info": None,
        "requests_24h": 0,
        "requests_available": None,
        "reset_time_utc": None,
        "reset_time_cairo": None,
    }
    
    # Fetch key info
    try:
        key_info = get_openrouter_key_info(base_url, api_key)
        result["key_info"] = key_info
        print_section("OpenRouter Key Info")
        print(f"  Data: {key_info}")
    except Exception as e:
        print(f"Warning: Could not fetch key info: {e}")
        key_info = None
    
    # Read chat_requests.jsonl
    log_path = Path(data_dir) / "chat_requests.jsonl"
    if log_path.exists():
        now = datetime.utcnow()
        requests_24h = []
        
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    timestamp_str = entry.get("timestamp")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        if timestamp >= now - timedelta(hours=24):
                            requests_24h.append(timestamp)
                except Exception as e:
                    continue
        
        result["requests_24h"] = len(requests_24h)
        
        # Calculate when requests will be available (rolling 24h window)
        if requests_24h:
            # Sort by timestamp
            requests_24h.sort()
            # The oldest request in the window determines when a slot opens
            oldest_in_window = requests_24h[0]
            reset_time = oldest_in_window + timedelta(hours=24)
            result["reset_time_utc"] = reset_time.isoformat()
            # Cairo time is UTC+2
            result["reset_time_cairo"] = (reset_time + timedelta(hours=2)).isoformat()
        
        # Estimate available requests (free tier typically 50/day)
        if key_info and isinstance(key_info, dict):
            limit = key_info.get("limit") or key_info.get("rate_limit") or 50
            # If log is empty, use API's remaining value from free_model_daily_requests
            if result["requests_24h"] == 0:
                # Try to get remaining from nested data structure
                # key_info is the full response with 'data' key
                if "data" in key_info and isinstance(key_info["data"], dict):
                    free_daily = key_info["data"].get("free_model_daily_requests", {})
                    if isinstance(free_daily, dict):
                        api_remaining = free_daily.get("remaining")
                        if api_remaining is not None:
                            result["requests_available"] = api_remaining
                            print(f"  Using API remaining: {api_remaining}")
                        else:
                            result["requests_available"] = limit
                            print(f"  API remaining not found, using limit: {limit}")
                    else:
                        result["requests_available"] = limit
                        print(f"  free_model_daily_requests not a dict, using limit: {limit}")
                else:
                    result["requests_available"] = limit
                    print(f"  No 'data' key in key_info, using limit: {limit}")
            else:
                result["requests_available"] = max(0, limit - len(requests_24h))
        elif result["requests_24h"] == 0:
            # No key info and no log, assume full quota available
            result["requests_available"] = 50
    
    return result


def get_openrouter_key_info(base_url: str, api_key: str) -> Dict[str, Any]:
    """Fetch API key info from OpenRouter (rate limits, usage)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/omartt4/AI-project-group",
        "X-Title": "Graph RAG Lab"
    }
    
    response = requests.get(f"{base_url}/key", headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


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
    parser.add_argument("--check-quota", action="store_true", help="Check chat quota only (no model verification)")
    args = parser.parse_args()
    
    config = get_config()
    
    if not config.openrouter_api_key:
        print_error("OPENROUTER_API_KEY not set in .env")
        sys.exit(1)
    
    # Check quota if requested
    if args.check_quota:
        quota_info = check_chat_quota(config.openrouter_base_url, config.openrouter_api_key, config.data_dir)
        print_section("Chat Quota Status")
        print(f"  Requests in last 24h: {quota_info['requests_24h']}")
        if quota_info['requests_available'] is not None:
            print(f"  Requests available: {quota_info['requests_available']}")
        if quota_info['reset_time_utc']:
            print(f"  Reset time (UTC): {quota_info['reset_time_utc']}")
            print(f"  Reset time (Cairo, UTC+2): {quota_info['reset_time_cairo']}")
        return
    
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
    
    # Verify chat models
    print("\nChat models:")
    if config.chat_model:
        errors = verify_chat_model(config.chat_model, chat_models)
        all_errors.extend(errors)
    if config.qgen_model:
        errors = verify_chat_model(config.qgen_model, chat_models)
        all_errors.extend(errors)
    # Verify fallback models
    if config.qgen_fallbacks:
        fallback_models = [m.strip() for m in config.qgen_fallbacks.split(",")]
        print("\nFallback models:")
        for model_id in fallback_models:
            if model_id:
                errors = verify_chat_model(model_id, chat_models)
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
