#!/usr/bin/env python3
import json

questions = [json.loads(line) for line in open('eval_lab/questions.jsonl')]

# Count by paper
paper_counts = {}
for q in questions:
    pid = q['paper_id']
    paper_counts[pid] = paper_counts.get(pid, 0) + 1

print("Per paper question counts:")
for pid in sorted(paper_counts.keys()):
    print("  " + pid + ":", paper_counts[pid])

print("\nPer style counts:")
style_counts = {}
for q in questions:
    style = q['style']
    style_counts[style] = style_counts.get(style, 0) + 1
for style in sorted(style_counts.keys()):
    print("  " + style + ":", style_counts[style])
