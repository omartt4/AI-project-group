"""
Phase 2: Embed chunks and store in ChromaDB.
Supports caching, rate limiting, and two modes: vector (raw text) and enriched (keywords + text).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import numpy as np
import requests
import tiktoken

from lab_common import (
    get_config,
    print_section,
    print_warning,
    print_error
)


def verify_all_models():
    """Import and run model verification."""
    from lab_models import main as verify_models_main
    old_argv = sys.argv
    sys.argv = ["lab_models.py"]
    try:
        try:
            verify_models_main()
        except SystemExit as e:
            if e.code != 0:
                sys.exit(e.code)
    finally:
        sys.argv = old_argv


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, requests_per_min: int):
        self.requests_per_min = requests_per_min
        self.min_interval = 60.0 / requests_per_min
        self.last_call = 0
    
    def wait(self):
        """Wait if needed to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_call = time.time()


class EmbeddingCache:
    """Disk cache for embeddings keyed by sha256(model + dim + prefix + text)."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embedding_cache.json"
        self.cache: Dict[str, List[float]] = {}
        self._load()
    
    def _load(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
    
    def _save(self):
        """Save cache to disk."""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)
    
    def _key(self, model: str, dim: Optional[int], prefix: str, text: str) -> str:
        """Generate cache key."""
        key_str = f"{model}|{dim}|{prefix}|{text}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, model: str, dim: Optional[int], prefix: str, text: str) -> Optional[List[float]]:
        """Get cached embedding if available."""
        key = self._key(model, dim, prefix, text)
        return self.cache.get(key)
    
    def set(self, model: str, dim: Optional[int], prefix: str, text: str, embedding: List[float]):
        """Cache an embedding."""
        key = self._key(model, dim, prefix, text)
        self.cache[key] = embedding
        self._save()


class OpenRouterEmbedder:
    """Embedding client for OpenRouter with rate limiting and caching."""
    
    def __init__(self, model_id: str, api_key: str, base_url: str, 
                 embed_dim: Optional[int], query_prefix: str,
                 requests_per_min: int, cache_dir: str):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.embed_dim = embed_dim
        self.query_prefix = query_prefix
        self.rate_limiter = RateLimiter(requests_per_min)
        self.cache = EmbeddingCache(cache_dir)
        self.failed_requests = 0
        self.retried_requests = 0
        self.total_latency = 0.0
        self.request_count = 0
        self.usage_returned = False  # Track if API returned usage data
        self.http_status_codes = []  # Track HTTP status codes
        
        # Get model context window from OpenRouter embedding catalog
        self.context_length = self._get_context_length()
    
    def _get_context_length(self) -> int:
        """Get context length for the model from /embeddings/models."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/omartt4/AI-project-group",
            "X-Title": "Graph RAG Lab"
        }
        
        try:
            response = requests.get(f"{self.base_url}/embeddings/models", headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for model in data.get("data", []):
                if model["id"] == self.model_id:
                    return model.get("context_length", 32768)
        except:
            pass
        
        return 32768  # Default fallback
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            return len(text.split())
    
    def embed_batch(self, texts: List[str], is_query: bool = False) -> tuple[List[List[float]], int]:
        """
        Embed a batch of texts with caching and rate limiting.
        Returns (embeddings, tokens_used).
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        tokens_used = 0
        
        # Check cache first
        for i, text in enumerate(texts):
            prefix = self.query_prefix if is_query else ""
            cached = self.cache.get(self.model_id, self.embed_dim, prefix, text)
            if cached is not None:
                embeddings.append(cached)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if not uncached_texts:
            return embeddings, tokens_used
        
        # Embed uncached texts
        new_embeddings, batch_tokens = self._embed_batch_api(uncached_texts, is_query)
        tokens_used += batch_tokens
        
        # Cache and insert in correct positions
        for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
            prefix = self.query_prefix if is_query else ""
            self.cache.set(self.model_id, self.embed_dim, prefix, text, emb)
        
        # Merge results
        result = embeddings.copy()
        for idx, emb in zip(uncached_indices, new_embeddings):
            result.insert(idx, emb)
        
        return result, tokens_used
    
    def _embed_batch_api(self, texts: List[str], is_query: bool = False) -> tuple[List[List[float]], int]:
        """
        Call OpenRouter embeddings API with retry logic.
        Returns (embeddings, tokens_used). Tokens are from API usage or estimated with tiktoken.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/omartt4/AI-project-group",
            "X-Title": "Graph RAG Lab"
        }
        
        # Add prefix if query
        if is_query and self.query_prefix:
            texts = [f"{self.query_prefix}{t}" for t in texts]
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait()
                
                start_time = time.time()
                
                data = {
                    "model": self.model_id,
                    "input": texts
                }
                
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                latency = time.time() - start_time
                self.total_latency += latency
                self.request_count += 1
                self.http_status_codes.append(response.status_code)
                
                if response.status_code == 429:
                    # Rate limited - exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + np.random.uniform(0, 1)
                    print_warning(f"Rate limited, waiting {delay:.1f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                    self.retried_requests += 1
                    continue
                
                response.raise_for_status()
                
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                
                # Get token usage from response if available
                tokens_used = 0
                usage = result.get("usage")
                if usage and "total_tokens" in usage:
                    tokens_used = usage["total_tokens"]
                    self.usage_returned = True
                else:
                    # Estimate with tiktoken
                    tokens_used = sum(self._estimate_tokens(t) for t in texts)
                
                # Truncate if embed_dim specified
                if self.embed_dim:
                    embeddings = [emb[:self.embed_dim] for emb in embeddings]
                
                return embeddings, tokens_used
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print_warning(f"Timeout, retrying in {delay:.1f}s")
                    time.sleep(delay)
                    self.retried_requests += 1
                else:
                    print_error(f"Failed to embed after {max_retries} attempts")
                    self.failed_requests += 1
                    # Return zero vectors as fallback
                    dim = self.embed_dim or 2048
                    tokens_used = sum(self._estimate_tokens(t) for t in texts)
                    return [[0.0] * dim for _ in texts], tokens_used
            
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print_warning(f"Error: {e}, retrying in {delay:.1f}s")
                    time.sleep(delay)
                    self.retried_requests += 1
                else:
                    print_error(f"Failed to embed: {e}")
                    self.failed_requests += 1
                    dim = self.embed_dim or 2048
                    tokens_used = sum(self._estimate_tokens(t) for t in texts)
                    return [[0.0] * dim for _ in texts], tokens_used
        
        # Should not reach here
        dim = self.embed_dim or 2048
        tokens_used = sum(self._estimate_tokens(t) for t in texts)
        return [[0.0] * dim for _ in texts], tokens_used


def get_collection_name(model_tag: str, chunk_size: int, chunk_overlap: int, mode: str) -> str:
    """Generate Chroma collection name."""
    # Sanitize model tag for collection name
    safe_tag = model_tag.replace("/", "_").replace(":", "_")
    return f"retrieval_lab_{safe_tag}_{chunk_size}_{chunk_overlap}_{mode}"


def estimate_requests(chunks: List[Dict], batch_size: int, model_context: int) -> Dict:
    """Estimate API usage for confirmation."""
    # Count tokens
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(encoding.encode(c["text"])) for c in chunks)
    
    # Check if chunks fit in context
    max_chunk_tokens = max(len(encoding.encode(c["text"])) for c in chunks)
    
    if max_chunk_tokens > model_context:
        print_error(f"Chunk size exceeds model context: {max_chunk_tokens} > {model_context}")
        return None
    
    num_requests = (len(chunks) + batch_size - 1) // batch_size
    return {
        "chunks": len(chunks),
        "total_tokens": total_tokens,
        "num_requests": num_requests,
        "max_chunk_tokens": max_chunk_tokens,
        "model_context": model_context,
        "batch_size": batch_size
    }


def embed_and_store(chunks_file: str, model_id: str, mode: str, config, confirm: bool = True):
    """
    Embed chunks and store in ChromaDB.
    mode: "vector" (raw text) or "enriched" (keywords + text)
    """
    print_section("Phase 2: Embedding Chunks")
    
    chunks_path = Path(chunks_file)
    if not chunks_path.exists():
        print_error(f"Chunks file not found: {chunks_file}")
        sys.exit(1)
    
    # Load chunks
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")
    print(f"Model: {model_id}")
    print(f"Mode: {mode}")
    
    # Verify models
    verify_all_models()
    
    # Create embedder
    embedder = OpenRouterEmbedder(
        model_id=model_id,
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
        embed_dim=config.embed_dim,
        query_prefix=config.query_prefix,
        requests_per_min=config.requests_per_min,
        cache_dir=config.data_dir
    )
    
    # Estimate usage
    model_tag = model_id.replace("/", "_").replace(":", "_")
    estimate = estimate_requests(chunks, config.batch_size, embedder.context_length)
    
    if estimate is None:
        sys.exit(1)
    
    # Load manifest for paper info
    manifest_path = Path(config.data_dir) / "manifest.csv"
    manifest = {}
    if manifest_path.exists():
        import csv
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest[row["paper_id"]] = row
        print(f"Loaded manifest with {len(manifest)} papers")
    else:
        print_warning(f"Manifest not found at {manifest_path}")
    
    print(f"\nEstimate:")
    print(f"  Chunks: {estimate['chunks']}")
    print(f"  Total tokens: {estimate['total_tokens']:,}")
    print(f"  Max chunk tokens: {estimate['max_chunk_tokens']}")
    print(f"  Model context: {estimate['model_context']:,}")
    print(f"  Requests (batch size {config.batch_size}): {estimate['num_requests']}")
    print(f"  Estimated time at {config.requests_per_min} req/min: {estimate['num_requests'] / config.requests_per_min:.1f} min")
    
    if confirm:
        response = input("\nProceed? (y/n): ")
        if response.lower() != "y":
            print("Aborted")
            return
    
    # Prepare texts based on mode
    texts = []
    for chunk in chunks:
        if mode == "vector":
            texts.append(chunk["text"])
        elif mode == "enriched":
            entities = chunk.get("entities", [])
            if entities:
                prefix = "Keywords: " + ", ".join(entities[:10]) + "\n"
                texts.append(prefix + chunk["text"])
            else:
                texts.append(chunk["text"])
        else:
            print_error(f"Unknown mode: {mode}")
            sys.exit(1)
    
    # Check enriched mode doesn't exceed context
    if mode == "enriched":
        max_enriched = max(len(t.split()) for t in texts)  # Rough estimate
        if max_enriched > embedder.context_length:
            print_warning(f"Enriched mode may exceed context: {max_enriched} tokens")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != "y":
                return
    
    # Embed in batches
    print_section("Embedding")
    
    start_time = time.time()
    all_embeddings = []
    cache_hits = 0
    total_tokens = 0
    
    for i in range(0, len(texts), config.batch_size):
        batch = texts[i:i + config.batch_size]
        print(f"  Batch {i//config.batch_size + 1}/{(len(texts) + config.batch_size - 1)//config.batch_size}")
        
        # Check cache hits before calling
        batch_cache_hits = sum(
            1 for t in batch
            if embedder.cache.get(model_id, config.embed_dim, config.query_prefix if mode == "vector" else "", t) is not None
        )
        cache_hits += batch_cache_hits
        
        embeddings, batch_tokens = embedder.embed_batch(batch, is_query=False)
        all_embeddings.extend(embeddings)
        total_tokens += batch_tokens
    
    wall_time = time.time() - start_time
    avg_latency = embedder.total_latency / embedder.request_count if embedder.request_count > 0 else 0
    
    # Store in ChromaDB
    print_section("Storing in ChromaDB")
    
    chroma_client = chromadb.PersistentClient(path=config.chroma_dir)
    collection_name = get_collection_name(model_tag, config.chunk_size, config.chunk_overlap, mode)
    
    # Delete existing collection if present
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Prepare documents and metadata
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    
    # Metadata (Chroma can't handle lists, so join entities with |)
    metadatas = []
    for chunk in chunks:
        meta = {
            "paper_id": chunk["paper_id"],
            "field": chunk["field"],
            "page_start": str(chunk["page_start"]),
            "page_end": str(chunk["page_end"]),
            "chunk_index": str(chunk["chunk_index"]),
            "n_tokens": str(chunk["n_tokens"]),
            "entities": "|".join(chunk.get("entities", []))
        }
        metadatas.append(meta)
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Stored {len(all_embeddings)} embeddings in collection: {collection_name}")
    
    # Save report
    report = {
        "model_id": model_id,
        "mode": mode,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "collection_name": collection_name,
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "num_requests": embedder.request_count,
        "failed_requests": embedder.failed_requests,
        "retried_requests": embedder.retried_requests,
        "cache_hits": cache_hits,
        "wall_time_seconds": wall_time,
        "tokens_per_second": total_tokens / wall_time if wall_time > 0 else 0,
        "avg_request_latency_seconds": avg_latency,
        "embed_dim": config.embed_dim or len(all_embeddings[0]) if all_embeddings else 0,
        "query_prefix": config.query_prefix,
        "tokens_estimated": not embedder.usage_returned,  # True if estimated (no usage field)
        "batch_size": config.batch_size,
        "model_context": embedder.context_length,
        "http_status_codes": embedder.http_status_codes,
        "manifest_papers": len(manifest) if manifest else 0
    }
    
    report_path = Path(config.data_dir) / f"embedding_report_{model_tag}_{mode}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {report_path}")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Tokens/sec: {report['tokens_per_second']:.1f}")
    print(f"  Avg request latency: {avg_latency:.2f}s")
    print(f"  Tokens estimated: {report['tokens_estimated']}")
    print(f"  HTTP status codes: {report['http_status_codes']}")
    print(f"  Manifest papers: {report['manifest_papers']}")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Failed requests: {embedder.failed_requests}")
    print(f"  Retried requests: {embedder.retried_requests}")
    print(f"  Total tokens: {total_tokens:,}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Embed chunks")
    parser.add_argument("--model", required=True, help="Embedding model ID")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSONL file")
    parser.add_argument("--mode", choices=["vector", "enriched"], default="vector", help="Embedding mode")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    
    config = get_config()
    
    # Temporarily override embed_model for this run
    original_model = config.embed_model
    config.embed_model = args.model
    
    embed_and_store(args.chunks, args.model, args.mode, config, confirm=not args.no_confirm)
    
    # Restore
    config.embed_model = original_model


if __name__ == "__main__":
    main()
