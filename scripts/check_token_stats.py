#!/usr/bin/env python3
"""Check bge-small-en-v1.5 tokenizer stats for chunks."""
import json
from pathlib import Path
from transformers import AutoTokenizer

def check_token_stats(chunks_file):
    """Check token stats using bge-small-en-v1.5 tokenizer."""
    chunks_path = Path(chunks_file)
    
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    # Load bge-small-en-v1.5 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    
    token_counts = []
    for chunk in chunks:
        text = chunk["text"]
        # Tokenize with special tokens included
        tokens = tokenizer(text, add_special_tokens=True)
        token_counts.append(len(tokens["input_ids"]))
    
    return {
        "total": len(chunks),
        "min": min(token_counts),
        "max": max(token_counts),
        "mean": sum(token_counts) / len(token_counts)
    }

if __name__ == "__main__":
    print("Checking 256 chunks with bge-small-en-v1.5 tokenizer...")
    stats_256 = check_token_stats('data/lab/chunks_256_26.jsonl')
    print(f"256: min={stats_256['min']}, mean={stats_256['mean']:.1f}, max={stats_256['max']}")
    
    print("\nChecking 384 chunks with bge-small-en-v1.5 tokenizer...")
    stats_384 = check_token_stats('data/lab/chunks_384_38.jsonl')
    print(f"384: min={stats_384['min']}, mean={stats_384['mean']:.1f}, max={stats_384['max']}")
