#!/usr/bin/env python3
"""Compute embedding cost for each config."""
import json
from pathlib import Path
from transformers import AutoTokenizer

def count_tokens(chunks_file, model_name="BAAI/bge-small-en-v1.5"):
    """Count total tokens using bge tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    total_tokens = 0
    for chunk in chunks:
        text = chunk['text']
        tokens = len(tokenizer(text, add_special_tokens=True)['input_ids'])
        total_tokens += tokens
    
    return len(chunks), total_tokens

if __name__ == "__main__":
    # Count tokens for both configs
    count_256, tokens_256 = count_tokens('data/lab/chunks_256_26.jsonl')
    count_384, tokens_384 = count_tokens('data/lab/chunks_384_38.jsonl')
    
    print(f"256/26: {count_256} chunks, {tokens_256} tokens")
    print(f"384/38: {count_384} chunks, {tokens_384} tokens")
