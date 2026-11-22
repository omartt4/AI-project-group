#!/usr/bin/env python3
"""Print per paper/field/style table for questions."""
import json
from pathlib import Path
from collections import defaultdict

def load_questions(questions_file):
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

if __name__ == "__main__":
    questions = load_questions('eval_lab/questions.jsonl')
    
    # Extract field from paper_id
    def get_field(paper_id):
        parts = paper_id.split('_')
        if len(parts) >= 2:
            return parts[1]
        return "Unknown"
    
    # Build table: paper -> field -> style -> count
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for q in questions:
        paper_id = q['paper_id']
        field = get_field(paper_id)
        style = q['style']
        table[paper_id][field][style] += 1
    
    # Print table
    print("Per paper / field / style table:")
    print("-" * 80)
    print(f"{'Paper':<60} {'Field':<15} {'Factual':<8} {'Conceptual':<10} {'Paraphrased':<12} {'Total':<6}")
    print("-" * 80)
    
    for paper_id in sorted(table.keys()):
        for field in sorted(table[paper_id].keys()):
            factual = table[paper_id][field]['factual']
            conceptual = table[paper_id][field]['conceptual']
            paraphrased = table[paper_id][field]['paraphrased']
            total = factual + conceptual + paraphrased
            print(f"{paper_id[:60]:<60} {field:<15} {factual:<8} {conceptual:<10} {paraphrased:<12} {total:<6}")
    
    # Totals
    print("-" * 80)
    total_factual = sum(q['style'] == 'factual' for q in questions)
    total_conceptual = sum(q['style'] == 'conceptual' for q in questions)
    total_paraphrased = sum(q['style'] == 'paraphrased' for q in questions)
    print(f"{'TOTAL':<60} {'':<15} {total_factual:<8} {total_conceptual:<10} {total_paraphrased:<12} {len(questions):<6}")
    print("-" * 80)
