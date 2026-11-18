#!/usr/bin/env python3
import json
from pathlib import Path

# Load questions
with open('data/lab/questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

# Create eval_lab directory
eval_dir = Path('eval_lab')
eval_dir.mkdir(exist_ok=True)

# Save as JSONL
with open(eval_dir / 'questions.jsonl', 'w', encoding='utf-8') as f:
    for q in questions:
        f.write(json.dumps(q) + '\n')

# Create metadata
by_paper = {}
for q in questions:
    p = q['paper_id']
    by_paper[p] = by_paper.get(p, 0) + 1

metadata = {
    'model': 'nvidia/nemotron-3.5-lightning:free',
    'seed': 42,
    'total_questions': len(questions),
    'papers': sorted(set([q['paper_id'] for q in questions])),
    'counts_by_paper': by_paper,
    'counts_by_style': {'factual': sum(1 for q in questions if q['style'] == 'factual'), 'conceptual': sum(1 for q in questions if q['style'] == 'conceptual'), 'paraphrased': sum(1 for q in questions if q['style'] == 'paraphrased')},
    'failed_papers': ['074_Machine_Learning_Unifying_Homophily_and_Heterophily_Network_Transformation_via_Motifs', '090_Machine_Learning_Adaptive_Precision_Training_for_Resource_Constrained_Devices']
}

with open(eval_dir / 'questions_meta.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved {len(questions)} questions to eval_lab/questions.jsonl")
print(f"Saved metadata to eval_lab/questions_meta.json")
