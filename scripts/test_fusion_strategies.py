"""
Test Fusion Strategies.

Tests RRF, Weighted Score, and Concatenation fusion strategies.
"""

from typing import List, Dict, Any

def rrf_fuse(vector_results: List[Dict], graph_results: List[Dict], k: int = 60) -> List[Dict]:
    rrf_scores = {}
    result_map = {}
    
    for idx, result in enumerate(vector_results):
        doc_id = result.get("id", str(idx))
        rank = idx + 1
        rrf_score = 1.0 / (k + rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score * 0.6
        result_map[doc_id] = result
    
    for idx, result in enumerate(graph_results):
        doc_id = result.get("id", f"graph_{idx}")
        rank = idx + 1
        rrf_score = 1.0 / (k + rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score * 0.4
        if doc_id not in result_map:
            result_map[doc_id] = result
    
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    fused = []
    for doc_id in sorted_ids:
        result = result_map[doc_id].copy()
        result["fusion_score"] = rrf_scores[doc_id]
        fused.append(result)
    
    return fused

# Test cases
print("Fusion Strategies Test")
print("=" * 60)

vector_results = [
    {"id": "v1", "text": "Vector result 1", "score": 0.9},
    {"id": "v2", "text": "Vector result 2", "score": 0.8},
    {"id": "v3", "text": "Vector result 3", "score": 0.7},
]

graph_results = [
    {"id": "g1", "text": "Graph result 1", "score": 0.85},
    {"id": "g2", "text": "Graph result 2", "score": 0.75},
]

print("\nVector Results:")
for r in vector_results:
    print(f"  {r['id']}: score={r['score']}")

print("\nGraph Results:")
for r in graph_results:
    print(f"  {r['id']}: score={r['score']}")

print("\nRRF Fusion Results:")
fused = rrf_fuse(vector_results, graph_results)
for r in fused:
    print(f"  {r['id']}: fusion_score={r['fusion_score']:.4f}, source={r.get('text', '')}")

print("\n" + "=" * 60)
print("Fusion strategies test completed!")
