#!/usr/bin/env python3
"""Print entity statistics."""
import json
from pathlib import Path
from collections import Counter

def load_chunks(chunks_file):
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

if __name__ == "__main__":
    chunks_256 = load_chunks('data/lab/chunks_256_26.jsonl')
    chunks_384 = load_chunks('data/lab/chunks_384_38.jsonl')
    
    # Count non-empty entities
    non_empty_256 = sum(1 for c in chunks_256 if c.get('entities') and len(c['entities']) > 0)
    non_empty_384 = sum(1 for c in chunks_384 if c.get('entities') and len(c['entities']) > 0)
    
    print(f"256 chunks: {non_empty_256}/{len(chunks_256)} have non-empty entities")
    print(f"384 chunks: {non_empty_384}/{len(chunks_384)} have non-empty entities")
    
    # 20 random chunks with entities
    import random
    random.seed(42)
    
    chunks_with_entities_256 = [c for c in chunks_256 if c.get('entities') and len(c['entities']) > 0]
    sample_256 = random.sample(chunks_with_entities_256, min(20, len(chunks_with_entities_256)))
    
    print("\n20 random chunks with entities (256):")
    for chunk in sample_256:
        print(f"  [{chunk['chunk_id']}]")
        print(f"    Text: {chunk['text'][:80]}...")
        print(f"    Entities: {chunk['entities'][:5]}")
    
    # Top 10 terms per paper
    paper_entities = {}
    for chunk in chunks_256:
        paper_id = chunk['paper_id']
        if paper_id not in paper_entities:
            paper_entities[paper_id] = Counter()
        for entity in chunk.get('entities', []):
            paper_entities[paper_id][entity] += 1
    
    print("\nTop 10 terms per paper:")
    for paper_id in sorted(paper_entities.keys()):
        print(f"\n  {paper_id}:")
        for term, count in paper_entities[paper_id].most_common(10):
            print(f"    {term}: {count}")
