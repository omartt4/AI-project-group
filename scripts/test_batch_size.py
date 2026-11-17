"""
Test different batch sizes for embedding API to find optimal setting.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from lab_common import get_config
from lab_embed import OpenRouterEmbedder


def test_batch_size(chunks_file: str, model_id: str, batch_sizes: list[int], config):
    """Test embedding with different batch sizes."""
    # Load chunks
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    # Use first 20 chunks for testing
    test_chunks = chunks[:20]
    texts = [c["text"] for c in test_chunks]
    
    print(f"Testing with {len(texts)} texts")
    print(f"Model: {model_id}")
    print()
    
    results = []
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        embedder = OpenRouterEmbedder(
            model_id=model_id,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            embed_dim=config.embed_dim,
            query_prefix=config.query_prefix,
            requests_per_min=config.requests_per_min,
            cache_dir=config.data_dir
        )
        
        # Clear cache for fair test
        embedder.cache.cache = {}
        
        start = time.time()
        embeddings, tokens = embedder.embed_batch(texts, is_query=False)
        elapsed = time.time() - start
        
        result = {
            "batch_size": batch_size,
            "num_requests": embedder.request_count,
            "total_time": elapsed,
            "avg_latency": embedder.total_latency / embedder.request_count if embedder.request_count > 0 else 0,
            "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
            "tokens": tokens,
            "http_status": embedder.http_status_codes,
            "failed": embedder.failed_requests,
            "retried": embedder.retried_requests
        }
        results.append(result)
        
        print(f"  Requests: {result['num_requests']}")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Avg latency: {result['avg_latency']:.2f}s")
        print(f"  Tokens/sec: {result['tokens_per_second']:.1f}")
        print(f"  HTTP status: {result['http_status']}")
        print()
    
    # Print summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'Requests':<10} {'Time (s)':<10} {'Latency (s)':<12} {'Tokens/s':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['batch_size']:<12} {r['num_requests']:<10} {r['total_time']:<10.2f} {r['avg_latency']:<12.2f} {r['tokens_per_second']:<12.1f}")
    
    # Recommend best batch size
    best = min(results, key=lambda x: x['total_time'])
    print()
    print(f"Recommended batch size: {best['batch_size']} (fastest: {best['total_time']:.2f}s)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test embedding batch sizes")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSONL file")
    parser.add_argument("--model", default="nvidia/nemotron-3.5-embed-1b:free", help="Model to test")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32], help="Batch sizes to test")
    args = parser.parse_args()
    
    config = get_config()
    
    results = test_batch_size(args.chunks, args.model, args.batch_sizes, config)
    
    # Save results
    output_path = Path(config.data_dir) / "batch_size_test.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
