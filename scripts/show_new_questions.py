#!/usr/bin/env python3
"""Show the 9 new questions from papers 074 and 090."""
import json

questions = []
with open('eval_lab/questions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        questions.append(json.loads(line))

new_questions = [q for q in questions if q['paper_id'].startswith('074_') or q['paper_id'].startswith('090_')]

print(f"Found {len(new_questions)} questions from papers 074 and 090")
print()

for q in new_questions:
    print(f"Chunk: {q['chunk_id']}")
    print(f"Question: {q['question']}")
    print()
