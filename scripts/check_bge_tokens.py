#!/usr/bin/env python3
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

for fname in ['data/lab/chunks_256_26.jsonl', 'data/lab/chunks_384_38.jsonl']:
    chunks = [json.loads(line) for line in open(fname)]
    
    # Token lengths with bge tokenizer
    token_lens = [len(model.tokenizer(c['text'])['input_ids']) for c in chunks]
    exceeding = sum(1 for t in token_lens if t > 480)
    
    print("\n" + fname + ":")
    print("  Chunk count:", len(chunks))
    print("  Mean bge tokens:", round(sum(token_lens)/len(token_lens), 1))
    print("  Max bge tokens:", max(token_lens))
    print("  Exceeding 480:", exceeding)
