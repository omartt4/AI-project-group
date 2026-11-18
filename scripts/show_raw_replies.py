#!/usr/bin/env python3
import json

# Load chat requests
requests = [json.loads(line) for line in open('data/lab/chat_requests.jsonl')]

# Filter for papers 074 and 090 with parse_failed
target_papers = ['074_Machine_Learning_Unifying_Homophily_and_Heterophily_Network_Transformation_via_Motifs', 
                 '090_Machine_Learning_Adaptive_Precision_Training_for_Resource_Constrained_Devices']
parse_failed = [r for r in requests if any(p in target_papers for p in r['paper_ids']) and r['outcome'] == 'parse_failed']

print("=" * 80)
print("RAW REPLIES FOR PARSE FAILURES (074 and 090)")
print("=" * 80)

for r in parse_failed:
    paper = r['paper_ids'][0]
    print(f"\nPaper: {paper}")
    print(f"  Model: {r['model']}")
    print(f"  Raw reply (first 600 chars):")
    raw = r.get('raw_reply', '')
    if raw:
        print(f"  {raw[:600]}")
    else:
        print("  (no raw reply saved)")
