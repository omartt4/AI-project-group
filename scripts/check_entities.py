#!/usr/bin/env python3
"""Check if entities field exists in chunk files."""
import json
from pathlib import Path

def check_entities(chunks_file):
    """Check if entities field exists in all chunks."""
    chunks_path = Path(chunks_file)
    
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    with_entities = sum(1 for c in chunks if 'entities' in c)
    empty_entities = sum(1 for c in chunks if c.get('entities') == [])
    non_empty_entities = sum(1 for c in chunks if c.get('entities') and len(c.get('entities', [])) > 0)
    
    return len(chunks), with_entities, empty_entities, non_empty_entities

if __name__ == "__main__":
    total_256, with_256, empty_256, non_empty_256 = check_entities('data/lab/chunks_256_26.jsonl')
    print(f"256 chunks: {total_256} total, {with_256} have entities field, {empty_256} empty, {non_empty_256} non-empty")
    
    total_384, with_384, empty_384, non_empty_384 = check_entities('data/lab/chunks_384_38.jsonl')
    print(f"384 chunks: {total_384} total, {with_384} have entities field, {empty_384} empty, {non_empty_384} non-empty")
