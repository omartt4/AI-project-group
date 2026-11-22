#!/usr/bin/env python3
"""Classify failures and identify vaguest questions."""
import csv
import json
from pathlib import Path
from collections import defaultdict

def classify_failures():
    """Classify failures by type."""
    failures = []
    with open('eval_lab/failures.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            failures.append(row)
    
    # Group by question
    question_failures = defaultdict(list)
    for failure in failures:
        question_failures[failure['question']].append(failure)
    
    # Classify each unique question
    classified = []
    for question, fails in question_failures.items():
        # Check if any failure has correct paper in top_3
        has_correct_paper = False
        has_wrong_paper = False
        
        for fail in fails:
            top_3 = eval(fail['top_3'])
            gold_paper = fail['paper_id']
            
            # Check if any top_3 chunk is from correct paper
            for chunk_id in top_3:
                if gold_paper in chunk_id:
                    has_correct_paper = True
                    break
            
            # Check if all top_3 are from wrong papers
            if not has_correct_paper:
                all_wrong = all(gold_paper not in chunk_id for chunk_id in top_3)
                if all_wrong:
                    has_wrong_paper = True
        
        # Classify
        if has_correct_paper:
            classification = "right_paper_wrong_chunk"
        elif has_wrong_paper:
            classification = "wrong_paper"
        else:
            classification = "ambiguous_question"
        
        classified.append({
            'question': question,
            'paper_id': fails[0]['paper_id'],
            'style': fails[0]['style'],
            'classification': classification,
            'n_failures': len(fails)
        })
    
    return classified

def find_vaguest_questions():
    """Find 15 vaguest questions based on average rank across configs."""
    with open('eval_lab/per_question.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Group by question
    question_ranks = defaultdict(list)
    for row in rows:
        question_ranks[row['question']].append(int(row['rank']))
    
    # Calculate average rank (999 for failures)
    question_avg = []
    for question, ranks in question_ranks.items():
        avg_rank = sum(ranks) / len(ranks)
        question_avg.append({
            'question': question,
            'avg_rank': avg_rank,
            'min_rank': min(ranks),
            'max_rank': max(ranks),
            'n_configs': len(ranks)
        })
    
    # Sort by average rank (highest = vaguest)
    question_avg.sort(key=lambda x: x['avg_rank'], reverse=True)
    
    return question_avg[:15]

if __name__ == "__main__":
    # Classify failures
    classified = classify_failures()
    print("Failure Classification:")
    print("=" * 80)
    for item in classified:
        print(f"\nQuestion: {item['question'][:80]}...")
        print(f"  Paper: {item['paper_id']}")
        print(f"  Style: {item['style']}")
        print(f"  Classification: {item['classification']}")
        print(f"  Failures: {item['n_failures']}")
    
    # Find vaguest questions
    vaguest = find_vaguest_questions()
    print("\n\n15 Vaguest Questions:")
    print("=" * 80)
    for i, item in enumerate(vaguest, 1):
        print(f"\n{i}. Question: {item['question'][:80]}...")
        print(f"   Avg Rank: {item['avg_rank']:.2f}")
        print(f"   Min Rank: {item['min_rank']}")
        print(f"   Max Rank: {item['max_rank']}")
        print(f"   Configs: {item['n_configs']}")
    
    # Save to file
    with open('eval_lab/failure_classification.txt', 'w', encoding='utf-8') as f:
        f.write("Failure Classification:\n")
        f.write("=" * 80 + "\n\n")
        for item in classified:
            f.write(f"Question: {item['question']}\n")
            f.write(f"  Paper: {item['paper_id']}\n")
            f.write(f"  Style: {item['style']}\n")
            f.write(f"  Classification: {item['classification']}\n")
            f.write(f"  Failures: {item['n_failures']}\n\n")
        
        f.write("\n15 Vaguest Questions:\n")
        f.write("=" * 80 + "\n\n")
        for i, item in enumerate(vaguest, 1):
            f.write(f"{i}. Question: {item['question']}\n")
            f.write(f"   Avg Rank: {item['avg_rank']:.2f}\n")
            f.write(f"   Min Rank: {item['min_rank']}\n")
            f.write(f"   Max Rank: {item['max_rank']}\n")
            f.write(f"   Configs: {item['n_configs']}\n\n")
    
    print(f"\n\nClassification saved to eval_lab/failure_classification.txt")
