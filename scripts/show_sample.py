#!/usr/bin/env python3
import json
import random

data = [json.loads(line) for line in open('eval_lab/questions.jsonl')]
random.seed(42)
sample = random.sample(data, min(15, len(data)))

print('15 Random Questions (covering all 7 successful papers):')
for i, q in enumerate(sample, 1):
    print(f'\n{i}. Paper: {q["paper_id"]}')
    print(f'   Style: {q["style"]}')
    print(f'   Source: {q.get("source_snippet", "N/A")[:80]}...')
    print(f'   Question: {q["question"]}')
