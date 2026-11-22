#!/usr/bin/env python3
import json

old_chunks = [json.loads(line) for line in open('data/lab/chunks_256_26.old.jsonl')]
new_chunks = [json.loads(line) for line in open('data/lab/chunks_256_26.jsonl')]

print("Old chunks:", len(old_chunks))
print("New chunks:", len(new_chunks))

if len(old_chunks) != len(new_chunks):
    print("ERROR: Different number of chunks!")
else:
    same_count = 0
    diff_count = 0
    for old, new in zip(old_chunks, new_chunks):
        if old['chunk_id'] != new['chunk_id']:
            print("Chunk ID mismatch:", old['chunk_id'], "vs", new['chunk_id'])
        if old['text'] == new['text']:
            same_count += 1
        else:
            diff_count += 1
            print("Text differs for:", new['chunk_id'])
    
    print("Same text:", same_count)
    print("Different text:", diff_count)
