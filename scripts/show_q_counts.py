#!/usr/bin/env python3
import json
from collections import Counter

data = [json.loads(line) for line in open('eval_lab/questions.jsonl')]
print('Total questions:', len(data))
print('By paper:', dict(Counter([q['paper_id'] for q in data])))
print('By field:', dict(Counter([q['paper_id'].split('_')[1] for q in data])))
print('By style:', dict(Counter([q['style'] for q in data])))
