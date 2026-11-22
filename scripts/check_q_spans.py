#!/usr/bin/env python3
import json

questions = [json.loads(line) for line in open('eval_lab/questions.jsonl')]
old_chunks = {c['chunk_id']: c for c in [json.loads(line) for line in open('data/lab/chunks_256_26.old.jsonl')]}
new_chunks = {c['chunk_id']: c for c in [json.loads(line) for line in open('data/lab/chunks_256_26.jsonl')]}

print("Total questions:", len(questions))

# Check which questions have changed chunk text
changed_q = []
for q in questions:
    chunk_id = q['chunk_id']
    if chunk_id in old_chunks and chunk_id in new_chunks:
        if old_chunks[chunk_id]['text'] != new_chunks[chunk_id]['text']:
            changed_q.append(q['paper_id'])
            print("Question chunk changed:", q['paper_id'], chunk_id)

print("\nQuestions with changed chunk text:", len(changed_q))
