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


class LocalEmbedder:
    """Local embedding client using sentence-transformers."""
    
    def __init__(self, model_name: str, query_prefix: str, cache_dir: str):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.cache = EmbeddingCache(cache_dir)
        self.model = None
        self.embed_dim = None
        self.max_seq_length = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        from sentence_transformers import SentenceTransformer
        print(f"Loading local model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embed_dim = self.model.get_sentence_embedding_dimension()
        # Get max sequence length from model config
        try:
            self.max_seq_length = self.model.max_seq_length
        except AttributeError:
            # Fallback to common defaults if not available
            self.max_seq_length = 512  # BGE-small default
        print(f"Model loaded: {self.model_name} (dim={self.embed_dim}, max_seq_length={self.max_seq_length})")
    
    def check_chunk_lengths(self, chunks: List[Dict]) -> tuple[int, int, int]:
        """
        Check chunk lengths using the model's tokenizer.
        Returns (max_tokens, mean_tokens, num_exceeding).
        """
        if not hasattr(self.model, 'tokenizer'):
            print_warning("Model has no tokenizer, using word count as fallback")
            token_counts = [len(c.get("text", "").split()) for c in chunks]
            return max(token_counts), int(sum(token_counts) / len(token_counts)), 0
        
        token_counts = []
        for chunk in chunks:
            text = chunk.get("text", "")
            # Use preprocess method (non-deprecated) to get token count
            try:
                # preprocess returns a dict with 'input_ids' or similar
                processed = self.model.preprocess([text])
                if isinstance(processed, dict) and 'input_ids' in processed:
                    count = len(processed['input_ids'][0]) if isinstance(processed['input_ids'], list) else len(processed['input_ids'])
                elif isinstance(processed, list) and len(processed) > 0:
                    count = len(processed[0]) if isinstance(processed[0], list) else len(processed)
                else:
                    # Fallback to word count
                    count = len(text.split())
            except Exception as e:
                print_warning(f"Tokenizer error: {e}, using word count fallback")
                count = len(text.split())
            token_counts.append(count)
        
        max_tokens = max(token_counts)
        mean_tokens = int(sum(token_counts) / len(token_counts))
        num_exceeding = sum(1 for c in token_counts if c > self.max_seq_length)
        
        return max_tokens, mean_tokens, num_exceeding
    
    def embed_batch(self, texts: List[str], is_query: bool = False) -> tuple[List[List[float]], int]:
        """
        Embed a batch of texts with caching.
        Returns (embeddings, tokens_used).
        """
        result = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            prefix = self.query_prefix if is_query else ""
            cached = self.cache.get(self.model_name, self.embed_dim, prefix, text)
            if cached is not None:
                result[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if not uncached_texts:
            return result, 0
        
        # Prepare texts with prefix for queries
        texts_to_embed = []
        for text in uncached_texts:
            if is_query:
                texts_to_embed.append(self.query_prefix + text)
            else:
                texts_to_embed.append(text)
        
        # Embed uncached texts
        new_embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)
        new_embeddings = new_embeddings.tolist()
        
        # Estimate tokens (rough approximation)
        tokens_used = sum(len(t.split()) for t in uncached_texts)
        
        # Cache new embeddings and place in result
        for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
            prefix = self.query_prefix if is_query else ""
            self.cache.set(self.model_name, self.embed_dim, prefix, text, emb)
            result[idx] = emb
        
        return result, tokens_used


def get_collection_name(model_tag: str, embed_dim: int, chunk_size: int, chunk_overlap: int, mode: str) -> str:
    """Generate Chroma collection name with model and dimension."""
    # Sanitize model tag for collection name
    safe_tag = model_tag.replace("/", "_").replace(":", "_")
    return f"retrieval_lab_{safe_tag}_{embed_dim}_{chunk_size}_{chunk_overlap}_{mode}"


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


def embed_and_store(chunks_file: str, model_id: str, mode: str, config, confirm: bool = True, allow_truncation: bool = False):
    """
    Embed chunks and store in ChromaDB.
    mode: "vector" (raw text) or "enriched" (keywords + text)
    backend: "local" or "openrouter" (from config)
    """
    print_section("Phase 2: Embedding Chunks")
    
    chunks_path = Path(chunks_file)
    if not chunks_path.exists():
        print_error(f"Chunks file not found: {chunks_file}")
        sys.exit(1)
    
    # Extract chunk_size and chunk_overlap from filename (e.g., chunks_512_51.jsonl)
    stem = chunks_path.stem  # e.g., "chunks_512_51"
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "chunks":
        try:
            chunk_size = int(parts[1])
            chunk_overlap = int(parts[2])
            print(f"Detected chunk size/overlap from filename: {chunk_size}/{chunk_overlap}")
        except ValueError:
            chunk_size = config.chunk_size
            chunk_overlap = config.chunk_overlap
            print_warning("Could not parse chunk size/overlap from filename, using config defaults")
    else:
        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap
        print_warning("Filename format not recognized, using config defaults")
    
    # Load chunks
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")
    print(f"Backend: {config.embed_backend}")
    print(f"Model: {model_id}")
    print(f"Mode: {mode}")
    
    # Determine actual model to use
    if config.embed_backend == "local":
        actual_model = config.embed_model_local
    else:
        actual_model = model_id
    
    # Create embedder based on backend
    if config.embed_backend == "local":
        embedder = LocalEmbedder(
            model_name=actual_model,
            query_prefix=config.query_prefix,
            cache_dir=config.data_dir
        )
        embed_dim = embedder.embed_dim
        model_context = embedder.max_seq_length  # Use actual model limit
        
        # Check chunk lengths against model's max sequence length
        print_section("Checking Chunk Lengths")
        max_tokens, mean_tokens, num_exceeding = embedder.check_chunk_lengths(chunks)
        print(f"  Max chunk tokens (model tokenizer): {max_tokens}")
        print(f"  Mean chunk tokens (model tokenizer): {mean_tokens}")
        print(f"  Model max sequence length: {embedder.max_seq_length}")
        print(f"  Chunks exceeding limit: {num_exceeding}/{len(chunks)}")
        
        if num_exceeding > 0 and not allow_truncation:
            print_error(f"{num_exceeding} chunks exceed model max sequence length ({embedder.max_seq_length})")
            print_error("This would cause silent truncation and invalid embeddings.")
            print_error("Re-run with --allow-truncation to proceed anyway (not recommended),")
            print_error("or re-chunk with a smaller chunk size (recommended: <= 256 for BGE-small).")
            sys.exit(1)
        elif num_exceeding > 0:
            print_warning(f"{num_exceeding} chunks exceed model max sequence length - proceeding with potential truncation")
    else:
        # Verify models for OpenRouter
        verify_all_models()
        embedder = OpenRouterEmbedder(
            model_id=model_id,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            embed_dim=config.embed_dim,
            query_prefix=config.query_prefix,
            requests_per_min=config.requests_per_min,
            cache_dir=config.data_dir
        )
        embed_dim = config.embed_dim or embedder.context_length
        model_context = embedder.context_length
    
    # Estimate usage
    model_tag = actual_model.replace("/", "_").replace(":", "_")
    estimate = estimate_requests(chunks, config.batch_size, model_context)
    
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
    print(f"  Embedding dimension: {embed_dim}")
    if config.embed_backend == "local":
        print(f"  Local embedding (no rate limit)")
    else:
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
        if max_enriched > model_context:
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
            if embedder.cache.get(actual_model, embed_dim, config.query_prefix if mode == "vector" else "", t) is not None
        )
        cache_hits += batch_cache_hits
        
        embeddings, batch_tokens = embedder.embed_batch(batch, is_query=False)
        all_embeddings.extend(embeddings)
        total_tokens += batch_tokens
    
    wall_time = time.time() - start_time
    avg_latency = getattr(embedder, 'total_latency', 0) / getattr(embedder, 'request_count', 1) if hasattr(embedder, 'request_count') and embedder.request_count > 0 else 0
    
    # Store in ChromaDB
    print_section("Storing in ChromaDB")
    
    chroma_client = chromadb.PersistentClient(path=config.chroma_dir)
    collection_name = get_collection_name(model_tag, embed_dim, chunk_size, chunk_overlap, mode)
    
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
    
    # Save report (only if not cache-only run)
    report_path = Path(config.data_dir) / f"embedding_report_{model_tag}_{mode}.json"
    
    # Check if this was a cache-only run (all embeddings from cache)
    if cache_hits == len(chunks):
        print_warning(f"Cache-only run (all {len(chunks)} chunks from cache)")
        if report_path.exists():
            print_warning(f"Skipping report overwrite to preserve previous results")
            print_warning(f"Existing report: {report_path}")
        else:
            # Still save report for first run
            report = {
                "backend": config.embed_backend,
                "model_id": actual_model,
                "mode": mode,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "collection_name": collection_name,
                "total_chunks": len(chunks),
                "total_tokens": total_tokens,
                "num_requests": getattr(embedder, 'request_count', 0),
                "failed_requests": getattr(embedder, 'failed_requests', 0),
                "retried_requests": getattr(embedder, 'retried_requests', 0),
                "cache_hits": cache_hits,
                "wall_time_seconds": wall_time,
                "tokens_per_second": total_tokens / wall_time if wall_time > 0 else 0,
                "avg_request_latency_seconds": avg_latency,
                "embed_dim": embed_dim,
                "query_prefix": config.query_prefix,
                "batch_size": config.batch_size,
                "model_context": model_context,
                "http_status_codes": getattr(embedder, 'http_status_codes', []),
                "manifest_papers": len(manifest) if manifest else 0
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to {report_path}")
            print(f"  Wall time: {wall_time:.1f}s")
            print(f"  Tokens/sec: {report['tokens_per_second']:.1f}")
            if config.embed_backend == "openrouter":
                print(f"  Avg request latency: {avg_latency:.2f}s")
                print(f"  Tokens estimated: {report.get('tokens_estimated', False)}")
                print(f"  HTTP status codes: {report.get('http_status_codes', [])}")
                print(f"  Failed requests: {report.get('failed_requests', 0)}")
                print(f"  Retried requests: {report.get('retried_requests', 0)}")
            print(f"  Manifest papers: {report['manifest_papers']}")
            print(f"  Cache hits: {cache_hits}")
            print(f"  Total tokens: {total_tokens:,}")
    else:
        # Normal run with actual embeddings, save report
        report = {
            "backend": config.embed_backend,
            "model_id": actual_model,
            "mode": mode,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "collection_name": collection_name,
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "num_requests": getattr(embedder, 'request_count', 0),
            "failed_requests": getattr(embedder, 'failed_requests', 0),
            "retried_requests": getattr(embedder, 'retried_requests', 0),
            "cache_hits": cache_hits,
            "wall_time_seconds": wall_time,
            "tokens_per_second": total_tokens / wall_time if wall_time > 0 else 0,
            "avg_request_latency_seconds": avg_latency,
            "embed_dim": embed_dim,
            "query_prefix": config.query_prefix,
            "batch_size": config.batch_size,
            "model_context": model_context,
            "http_status_codes": getattr(embedder, 'http_status_codes', []),
            "manifest_papers": len(manifest) if manifest else 0
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to {report_path}")
        print(f"  Wall time: {wall_time:.1f}s")
        print(f"  Tokens/sec: {report['tokens_per_second']:.1f}")
        if config.embed_backend == "openrouter":
            print(f"  Avg request latency: {avg_latency:.2f}s")
            print(f"  Tokens estimated: {report.get('tokens_estimated', False)}")
            print(f"  HTTP status codes: {report.get('http_status_codes', [])}")
            print(f"  Failed requests: {report.get('failed_requests', 0)}")
            print(f"  Retried requests: {report.get('retried_requests', 0)}")
        print(f"  Manifest papers: {report['manifest_papers']}")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Total tokens: {total_tokens:,}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Embed chunks")
    parser.add_argument("--model", required=True, help="Embedding model ID")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSONL file")
    parser.add_argument("--mode", choices=["vector", "enriched"], default="vector", help="Embedding mode")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--allow-truncation", action="store_true", help="Allow chunks exceeding model max sequence length (may truncate)")
    parser.add_argument("--query-prefix", type=str, default=None, help="Override query prefix (empty string for no prefix)")
    args = parser.parse_args()
    
    config = get_config()
    
    # Override query prefix if specified
    if args.query_prefix is not None:
        config.query_prefix = args.query_prefix
    
    # Temporarily override embed_model for this run
    original_model = config.embed_model
    config.embed_model = args.model
    
    embed_and_store(args.chunks, args.model, args.mode, config, confirm=not args.no_confirm, allow_truncation=args.allow_truncation)
    
    # Restore
    config.embed_model = original_model


if __name__ == "__main__":
    main()
