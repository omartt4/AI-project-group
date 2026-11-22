#!/usr/bin/env python3
import json
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

for fname in ['data/lab/chunks_256_26.jsonl', 'data/lab/chunks_384_38.jsonl']:
    chunks = [json.loads(line) for line in open(fname)]
    
    # Token lengths
    token_lens = [len(tokenizer.encode(c['text'])) for c in chunks]
    
    # Char lengths
    char_lens = [len(c['text']) for c in chunks]
    
    print("\n" + fname + ":")
    print("  Chunk count:", len(chunks))
    print("  Mean tokens:", round(sum(token_lens)/len(token_lens), 1))
    print("  Max tokens:", max(token_lens))
    print("  Mean chars:", round(sum(char_lens)/len(char_lens), 1))
    print("  Max chars:", max(char_lens))
