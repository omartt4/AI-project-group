#!/usr/bin/env python3
import json

# Load chat requests
requests = [json.loads(line) for line in open('data/lab/chat_requests.jsonl')]

# Filter for papers 074 and 090
target_papers = ['074_Machine_Learning_Unifying_Homophily_and_Heterophily_Network_Transformation_via_Motifs', 
                 '090_Machine_Learning_Adaptive_Precision_Training_for_Resource_Constrained_Devices']
failed = [r for r in requests if any(p in target_papers for p in r['paper_ids']) and r['outcome'] != 'success']

print("=" * 80)
print("DIAGNOSING FAILURES FOR PAPERS 074 AND 090")
print("=" * 80)

for r in failed:
    paper = r['paper_ids'][0]
    print(f"\nPaper: {paper}")
    print(f"  Status: {r['http_status']}")
    print(f"  Outcome: {r['outcome']}")
    print(f"  Model: {r['model']}")
    print(f"  Error: {r.get('error', 'N/A')[:200]}")
    
    # Extract provider info if available
    if r.get('error'):
        try:
            error_json = json.loads(r['error'])
            provider = error_json.get('error', {}).get('metadata', {}).get('provider_name', 'unknown')
            limit_source = error_json.get('error', {}).get('metadata', {}).get('limit_source', 'unknown')
            print(f"  Provider: {provider}")
            print(f"  Limit source: {limit_source}")
        except:
            pass

print("\n" + "=" * 80)
print("CHUNKS FOR PAPER 074")
print("=" * 80)

# Load chunks
chunks = [json.loads(line) for line in open('data/lab/chunks_256_26.jsonl')]
chunks_074 = [c for c in chunks if c['paper_id'].startswith('074')]
print(f"Total chunks for 074: {len(chunks_074)}")
for i, c in enumerate(chunks_074[:6]):  # First 6 chunks
    print(f"\nChunk {i+1}: {c['chunk_id']}")
    print(f"  Length: {len(c['text'])} chars")
    print(f"  First 200 chars: {c['text'][:200]}")
    has_formulas = '$' in c['text'] or '\\\\' in c['text']
    has_tables = '|' in c['text'] and c['text'].count('|') > 10
    print(f"  Has formulas: {has_formulas}")
    print(f"  Has tables: {has_tables}")

print("\n" + "=" * 80)
print("CHUNKS FOR PAPER 090")
print("=" * 80)

chunks_090 = [c for c in chunks if c['paper_id'].startswith('090')]
print(f"Total chunks for 090: {len(chunks_090)}")
for i, c in enumerate(chunks_090[:6]):  # First 6 chunks
    print(f"\nChunk {i+1}: {c['chunk_id']}")
    print(f"  Length: {len(c['text'])} chars")
    print(f"  First 200 chars: {c['text'][:200]}")
    has_formulas = '$' in c['text'] or '\\\\' in c['text']
    has_tables = '|' in c['text'] and c['text'].count('|') > 10
    print(f"  Has formulas: {has_formulas}")
    print(f"  Has tables: {has_tables}")
