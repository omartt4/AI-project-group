#!/usr/bin/env python3
import json

data = [json.loads(line) for line in open('data/lab/chat_requests.jsonl')]
failed = [r for r in data if r['outcome'] != 'success']
print('Failed requests:')
for r in failed:
    print(f"  {r['paper_ids'][0]}: {r['outcome']} - {r['model']}")
