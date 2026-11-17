"""
Phase 4: Retrieval + QA.
Embeds query, returns top-k chunks with scores, optionally answers using LLM.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
import requests
import tiktoken

from lab_common import (
    get_config,
    get_collection_name,
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_query_terms(query: str, extractor_type: str = "yake") -> List[str]:
    """Extract terms from query using the same extractor as chunks."""
    if extractor_type == "yake":
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.7, top=10)
            keywords = kw_extractor.extract_keywords(query)
            return [kw for kw, score in keywords]
        except ImportError:
            print_warning("YAKE not available, using simple word extraction")
            # Fallback: simple word extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            return list(set(words))
    else:
        # Simple word extraction fallback
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        return list(set(words))


def normalize_scores(scores: List[float], method: str = "minmax") -> List[float]:
    """Normalize scores to [0,1] range."""
    scores = np.array(scores)
    if method == "minmax":
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val == 0:
            return np.zeros_like(scores).tolist()
        return ((scores - min_val) / (max_val - min_val)).tolist()
    elif method == "zscore":
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return np.zeros_like(scores).tolist()
        return ((scores - mean) / std).tolist()
    else:
        return scores.tolist()


def retrieve(
    query: str,
    collection,
    embedder,
    top_k: int = 5,
    mode: str = "vector",
    entity_boost: float = 0.15,
    entity_extractor: str = "yake",
    shared_terms: Optional[set] = None,
    manifest: Optional[Dict] = None
) -> Tuple[List[Dict], Dict]:
    """
    Retrieve top-k chunks for a query.
    Returns (results, timing_info).
    """
    timing = {}
    
    # Step 1: Embed query (cached)
    start = time.time()
    query_embeddings, query_tokens = embedder.embed_batch([query], is_query=True)
    query_embedding = query_embeddings[0]
    timing["embed"] = time.time() - start
    
    # Step 2: Vector search
    start = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,  # Get more for entity boost
        include=["embeddings", "documents", "metadatas", "distances"]
    )
    timing["search"] = time.time() - start
    
    # Process results
    chunks = []
    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        doc = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        embedding = results["embeddings"][0][i]
        
        # Convert distance to similarity (cosine distance -> cosine similarity)
        similarity = 1 - distance
        
        # Get paper title from manifest
        paper_id = metadata["paper_id"]
        title = manifest.get(paper_id, {}).get("title", paper_id) if manifest else paper_id
        
        # Parse entities
        entities_str = metadata.get("entities", "")
        entities = entities_str.split("|") if entities_str else []
        
        chunks.append({
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "title": title,
            "page_start": metadata["page_start"],
            "page_end": metadata["page_end"],
            "text": doc,
            "similarity": similarity,
            "embedding": np.array(embedding),
            "entities": entities
        })
    
    # Step 3: Apply entity boost if needed
    if mode == "entity_boost" and shared_terms:
        start = time.time()
        
        # Extract query terms
        query_terms = extract_query_terms(query, entity_extractor)
        query_terms_set = set(query_terms)
        
        # Get top 50 candidates for normalization
        top_n = 50
        if len(chunks) > top_n:
            chunks_for_norm = chunks[:top_n]
        else:
            chunks_for_norm = chunks
        
        # Compute boost for each chunk
        for chunk in chunks_for_norm:
            chunk_entities_set = set(chunk["entities"])
            shared = chunk_entities_set & query_terms_set & shared_terms
            if shared:
                boost = entity_boost * (len(shared) / max(1, len(query_terms_set)))
                chunk["similarity"] += boost
                chunk["entity_boost"] = boost
            else:
                chunk["entity_boost"] = 0.0
        
        # Normalize scores (min-max) for top N candidates
        all_similarities = [c["similarity"] for c in chunks_for_norm]
        normalized = normalize_scores(all_similarities, method="minmax")
        for chunk, norm_score in zip(chunks_for_norm, normalized):
            chunk["similarity"] = norm_score
        
        # Re-sort and take top-k
        chunks.sort(key=lambda x: x["similarity"], reverse=True)
        chunks = chunks[:top_k]
        
        timing["boost"] = time.time() - start
    else:
        chunks = chunks[:top_k]
        timing["boost"] = 0.0
    
    return chunks, timing


def answer_with_llm(
    query: str,
    chunks: List[Dict],
    model_id: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 2000
) -> Tuple[str, Dict]:
    """Answer query using LLM with retrieved chunks as context."""
    timing = {}
    
    # Build context from chunks
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[{chunk['paper_id']}, page {chunk['page_start']}]")
        context_parts.append(chunk["text"][:500])  # Limit snippet length
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Context from research papers:
{context}

Question: {query}

Answer based on the context. If the answer is not in the context, write "not found in the papers".
Include citations like [paper_id, page]. Be brief and direct."""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/omartt4/AI-project-group",
        "X-Title": "Graph RAG Lab"
    }
    
    start = time.time()
    
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0
    }
    
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )
    
    timing["llm"] = time.time() - start
    timing["http_status"] = response.status_code
    
    response.raise_for_status()
    result = response.json()
    
    # Extract answer - handle reasoning models
    choice = result["choices"][0]
    finish_reason = choice.get("finish_reason", "unknown")
    timing["finish_reason"] = finish_reason
    
    message = choice["message"]
    answer = message.get("content", "")
    
    # If content is empty, try reasoning field (for reasoning models)
    if not answer:
        answer = message.get("reasoning", "")
        if not answer:
            answer = message.get("reasoning_content", "")
    
    # If still empty, print raw response for debugging
    if not answer:
        print_error(f"LLM returned empty answer. Raw response:")
        print(f"  Status: {response.status_code}")
        print(f"  Finish reason: {finish_reason}")
        print(f"  Message keys: {list(message.keys())}")
        print(f"  Content: {message.get('content', 'None')}")
        print(f"  Reasoning: {message.get('reasoning', 'None')}")
        print(f"  Reasoning content: {message.get('reasoning_content', 'None')}")
        answer = "[Error: LLM returned empty response]"
    else:
        # If answer contains "thinking process" or similar, try to extract final answer
        # Look for patterns like "Answer:" or the content after the thinking
        lines = answer.split('\n')
        final_answer_lines = []
        in_thinking = False
        for line in lines:
            line_lower = line.lower()
            if 'thinking process' in line_lower or 'analysis' in line_lower:
                in_thinking = True
            elif in_thinking and (line.startswith('Answer:') or line.startswith('Final answer:') or line.startswith('Therefore')):
                in_thinking = False
                final_answer_lines.append(line)
            elif not in_thinking:
                final_answer_lines.append(line)
        
        if final_answer_lines and len(final_answer_lines) < len(lines):
            answer = '\n'.join(final_answer_lines).strip()
            # If still starts with thinking markers, take last paragraph
            if answer.lower().startswith(('here\'s a thinking', 'let me', 'i need to')):
                paragraphs = answer.split('\n\n')
                if len(paragraphs) > 1:
                    answer = paragraphs[-1].strip()
    
    # Estimate tokens
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        timing["llm_tokens"] = len(encoding.encode(prompt)) + len(encoding.encode(answer))
    except:
        timing["llm_tokens"] = 0
    
    return answer, timing


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Retrieval + QA")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--field", choices=["nlp", "ml", "cv"], help="Filter by field")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--mode", choices=["vector", "entity_boost", "enriched"], default="vector", help="Retrieval mode")
    parser.add_argument("--answer", action="store_true", help="Generate answer with LLM")
    parser.add_argument("--collection", help="Chroma collection name (default: auto-detect)")
    parser.add_argument("--entity-boost", type=float, default=0.15, help="Entity boost weight")
    args = parser.parse_args()
    
    config = get_config()
    
    # Verify models
    print_section("Verifying OpenRouter Models")
    verify_all_models()
    
    # Load manifest
    manifest_path = Path(config.data_dir) / "manifest.csv"
    manifest = {}
    if manifest_path.exists():
        import csv
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest[row["paper_id"]] = row
    else:
        print_warning(f"Manifest not found at {manifest_path}")
    
    # Load shared terms
    shared_terms_path = Path(config.data_dir) / "shared_terms.csv"
    shared_terms = set()
    if shared_terms_path.exists():
        import csv
        with open(shared_terms_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shared_terms.add(row["term"])
    else:
        print_warning(f"Shared terms not found at {shared_terms_path}")
    
    # Get collection
    chroma_client = chromadb.PersistentClient(path=config.chroma_dir)
    
    if args.collection:
        collection_name = args.collection
    else:
        # Auto-detect from config (always use vector collection for now)
        collection_name = get_collection_name(
            config.embed_model.replace("/", "_").replace(":", "_"),
            config.chunk_size,
            config.chunk_overlap,
            "vector"  # Always use vector collection, entity_boost is just retrieval method
        )
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        print_error(f"Collection not found: {collection_name}")
        sys.exit(1)
    
    print(f"Using collection: {collection_name}")
    print(f"Mode: {args.mode}")
    print(f"Top-k: {args.top_k}")
    
    # Create embedder
    from lab_embed import OpenRouterEmbedder
    embedder = OpenRouterEmbedder(
        model_id=config.embed_model,
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
        embed_dim=config.embed_dim,
        query_prefix=config.query_prefix,
        requests_per_min=config.requests_per_min,
        cache_dir=config.data_dir
    )
    
    # Retrieve
    print_section(f"Retrieval for: {args.question}")
    
    results, timing = retrieve(
        query=args.question,
        collection=collection,
        embedder=embedder,
        top_k=args.top_k,
        mode=args.mode,
        entity_boost=args.entity_boost,
        entity_extractor=config.entity_extractor,
        shared_terms=shared_terms,
        manifest=manifest
    )
    
    # Print results
    print(f"\nRetrieved {len(results)} chunks in {timing['search']:.3f}s")
    print(f"Embedding time: {timing['embed']:.3f}s")
    if timing.get("boost", 0) > 0:
        print(f"Entity boost time: {timing['boost']:.3f}s")
    
    print("\nTop chunks:")
    for i, chunk in enumerate(results, 1):
        print(f"\n{i}. [{chunk['chunk_id']}] Score: {chunk['similarity']:.4f}")
        print(f"   Paper: {chunk['title']}")
        print(f"   Pages: {chunk['page_start']}-{chunk['page_end']}")
        if chunk["entities"]:
            print(f"   Entities: {', '.join(chunk['entities'][:5])}")
        if "entity_boost" in chunk:
            print(f"   Entity boost: +{chunk['entity_boost']:.4f}")
        print(f"   Snippet: {chunk['text'][:200]}...")
    
    # Answer with LLM if requested
    if args.answer:
        print_section("Generating Answer")
        
        answer, llm_timing = answer_with_llm(
            query=args.question,
            chunks=results,
            model_id=config.chat_model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url
        )
        
        print(f"\nAnswer (in {llm_timing['llm']:.3f}s, {llm_timing['llm_tokens']} tokens):")
        print(answer)
        
        timing.update(llm_timing)
    
    # Print total timing
    timing_values = [v for v in timing.values() if isinstance(v, (int, float))]
    total_time = sum(timing_values)
    print(f"\nTotal time: {total_time:.3f}s")
    print(f"  Embed: {timing.get('embed', 0):.3f}s")
    print(f"  Search: {timing.get('search', 0):.3f}s")
    if "boost" in timing:
        print(f"  Boost: {timing['boost']:.3f}s")
    if "llm" in timing:
        print(f"  LLM: {timing['llm']:.3f}s ({timing.get('llm_tokens', 0)} tokens)")
        if "http_status" in timing:
            print(f"  HTTP status: {timing['http_status']}")
        if "finish_reason" in timing:
            print(f"  Finish reason: {timing['finish_reason']}")


if __name__ == "__main__":
    main()
