"""
Estimate sweep cost for stage 1a.
"""

import tiktoken
from lab_common import get_config


def estimate_sweep():
    """Estimate cost for stage 1a sweep."""
    config = get_config()
    
    # Stage 1a config
    model = "nvidia/nemotron-3.5-embed-1b:free"
    chunk_sizes = [256, 512, 1024]
    chunk_overlap = 51  # 10% of 512
    n_papers = 9  # 3 papers per field
    configs = len(chunk_sizes)
    
    # Estimate tokens per paper (average from quick mode: ~10k chars per paper)
    avg_chars_per_paper = 30000
    avg_tokens_per_char = 0.25  # Rough estimate
    avg_tokens_per_paper = int(avg_chars_per_paper * avg_tokens_per_char)
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    print("=" * 60)
    print("Stage 1a Sweep Cost Estimate")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Papers: {n_papers} (3 per field)")
    print(f"Configs: {configs} (chunk_size {chunk_sizes}, overlap 10%)")
    print(f"Batch size: {config.batch_size}")
    print(f"Rate limit: {config.requests_per_min} req/min")
    print()
    
    total_requests = 0
    total_tokens = 0
    
    for chunk_size in chunk_sizes:
        # Estimate chunks per paper
        chunks_per_paper = avg_tokens_per_paper // chunk_size
        total_chunks = chunks_per_paper * n_papers
        
        # Estimate tokens per chunk
        avg_tokens_per_chunk = chunk_size * 0.9  # 90% utilization
        
        # Total tokens for this config
        config_tokens = total_chunks * avg_tokens_per_chunk
        
        # Requests
        config_requests = (total_chunks + config.batch_size - 1) // config.batch_size
        
        # Time
        config_time = config_requests / config.requests_per_min
        
        print(f"Chunk size {chunk_size}:")
        print(f"  Chunks per paper: {chunks_per_paper}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Total tokens: {config_tokens:,}")
        print(f"  Requests: {config_requests}")
        print(f"  Time: {config_time:.1f} min")
        print()
        
        total_requests += config_requests
        total_tokens += config_tokens
    
    print("=" * 60)
    print("Total")
    print("=" * 60)
    print(f"Total requests: {total_requests}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {total_requests / config.requests_per_min:.1f} min ({(total_requests / config.requests_per_min) / 60:.2f} hours)")
    print()
    print(f"Note: Free model, so cost = $0")
    print(f"Rate limiting will dominate: {total_requests} requests at {config.requests_per_min} req/min")


if __name__ == "__main__":
    estimate_sweep()
