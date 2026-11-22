#!/usr/bin/env python3
"""Compute per style and per field results from per_question.csv."""
import csv
from pathlib import Path
from collections import defaultdict

def compute_metrics_by_group(csv_file):
    """Compute metrics grouped by style and field for each config."""
    configs = ['256_none', '256_instruction', '384_none', '384_instruction', 'BM25_256', 'BM25_384']
    groups = ['style', 'field']
    
    # Read per_question.csv
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Split by config (each question appears 6 times now)
    config_rows = {config: [] for config in configs}
    for row in rows:
        config = row.get('config')
        if config in config_rows:
            config_rows[config].append(row)
    
    # Compute metrics for each config and group
    results = []
    for config in configs:
        rows_config = config_rows[config]
        
        if not rows_config:
            continue
        
        for group in groups:
            grouped = defaultdict(list)
            
            for row in rows_config:
                if group == 'style':
                    key = row['style']
                else:  # field - extract from paper_id
                    paper_id = row['paper_id']
                    if 'NLP' in paper_id:
                        key = 'NLP'
                    elif 'Computer_Vision' in paper_id:
                        key = 'Computer'
                    elif 'Machine_Learning' in paper_id:
                        key = 'Machine'
                    else:
                        key = 'Unknown'
                
                rank = int(row['rank'])
                grouped[key].append(rank)
            
            for key, ranks in grouped.items():
                n = len(ranks)
                hit_at_1 = sum(1 for r in ranks if r == 1) / n
                hit_at_3 = sum(1 for r in ranks if r <= 3) / n
                hit_at_5 = sum(1 for r in ranks if r <= 5) / n
                hit_at_10 = sum(1 for r in ranks if r <= 10) / n
                mrr = sum(1.0 / r for r in ranks if r > 0) / n
                
                results.append({
                    'config': config,
                    'group_type': group,
                    'group_value': key,
                    'n': n,
                    'hit@1': round(hit_at_1, 4),
                    'hit@3': round(hit_at_3, 4),
                    'hit@5': round(hit_at_5, 4),
                    'hit@10': round(hit_at_10, 4),
                    'mrr': round(mrr, 4)
                })
    
    return results

if __name__ == "__main__":
    results = compute_metrics_by_group('eval_lab/per_question.csv')
    
    # Save to CSV
    with open('eval_lab/summary_by_group.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'group_type', 'group_value', 'n', 'hit@1', 'hit@3', 'hit@5', 'hit@10', 'mrr'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved {len(results)} rows to eval_lab/summary_by_group.csv")
