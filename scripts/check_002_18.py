#!/usr/bin/env python3
"""Check if source snippet matches new chunk text for 002 chunk 18."""
import json
import re

def normalize_text(text):
    """Normalize whitespace and newlines."""
    return re.sub(r'\s+', ' ', text).strip()

# Load question
with open('eval_lab/questions.jsonl', 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f]

# Load new chunk
with open('data/lab/chunks_256_26.jsonl', 'r', encoding='utf-8') as f:
    chunks = {}
    for line in f:
        chunk = json.loads(line)
        chunks[chunk['chunk_id']] = chunk

chunk_id = '002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction_18'
question = [q for q in questions if q['chunk_id'] == chunk_id][0]
chunk = chunks[chunk_id]

source_snippet = question['source_snippet']
chunk_text = chunk['text']

print(f"Source snippet (original):")
print(repr(source_snippet[:200]))
print(f"\nChunk text (original):")
print(repr(chunk_text[:200]))

print(f"\nSource snippet (normalized):")
print(repr(normalize_text(source_snippet)[:200]))
print(f"\nChunk text (normalized):")
print(repr(normalize_text(chunk_text)[:200]))

if normalize_text(source_snippet) in normalize_text(chunk_text):
    print("\n✓ Source snippet matches normalized chunk text")
else:
    print("\n✗ Source snippet does NOT match normalized chunk text")
