"""
Show entity quality diagnostics.
"""

import json
import random
from collections import Counter
from pathlib import Path

chunks_file = Path("data/lab/chunks_512_50.jsonl")

# Load chunks
chunks = []
with open(chunks_file, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print("=" * 60)
print("20 Random Chunks (snippet + terms)")
print("=" * 60)
sample = random.sample(chunks, min(20, len(chunks)))
for chunk in sample:
    print(f"\n[{chunk['chunk_id']}]")
    print(f"  Text: {chunk['text'][:120]}...")
    print(f"  Terms: {chunk['entities'][:5]}")

# Count term frequency across corpus
term_freq = Counter()
for chunk in chunks:
    for term in chunk["entities"]:
        term_freq[term] += 1

print("\n" + "=" * 60)
print("30 Most Frequent Terms Across Corpus")
print("=" * 60)
for term, freq in term_freq.most_common(30):
    print(f"  {term}: {freq} occurrences")

# Per-paper top terms
paper_terms = {}
for chunk in chunks:
    paper_id = chunk["paper_id"]
    if paper_id not in paper_terms:
        paper_terms[paper_id] = Counter()
    for term in chunk["entities"]:
        paper_terms[paper_id][term] += 1

print("\n" + "=" * 60)
print("Top 15 Terms Per Paper")
print("=" * 60)
for paper_id, counter in sorted(paper_terms.items()):
    print(f"\n{paper_id}:")
    for term, freq in counter.most_common(15):
        print(f"  {term}: {freq}")
