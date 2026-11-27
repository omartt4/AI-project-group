"""
Test Retrieval Evaluation Metrics.

Tests Recall@K, MRR, and nDCG metrics independently.
"""

from typing import List, Dict, Set
import math

def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = len(retrieved_at_k & relevant_ids)
    return relevant_retrieved / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def dcg_at_k(relevances: List[float], k: int) -> float:
    relevances_at_k = relevances[:k]
    dcg = 0.0
    for i, rel in enumerate(relevances_at_k, start=1):
        dcg += rel / math.log2(i + 1)
    return dcg

def ndcg_at_k(retrieved_ids: List[str], relevance_judgments: Dict[str, float], k: int) -> float:
    if not relevance_judgments:
        return 0.0
    relevances = [relevance_judgments.get(doc_id, 0.0) for doc_id in retrieved_ids]
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevance_judgments.values(), reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

# Test cases
print("Retrieval Evaluation Metrics Test")
print("=" * 60)

# Test 1: Recall@K
retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = {"doc1", "doc3", "doc5"}
print("\nTest 1: Recall@K")
print(f"Retrieved: {retrieved}")
print(f"Relevant: {relevant}")
for k in [1, 3, 5, 10]:
    recall = recall_at_k(retrieved, relevant, k)
    print(f"  Recall@{k}: {recall:.3f}")

# Test 2: MRR
print("\nTest 2: Mean Reciprocal Rank")
print(f"Retrieved: {retrieved}")
print(f"Relevant: {relevant}")
mrr = mean_reciprocal_rank(retrieved, relevant)
print(f"  MRR: {mrr:.3f}")

# Test 3: nDCG@K
relevance_judgments = {
    "doc1": 1.0,
    "doc2": 0.8,
    "doc3": 0.6,
    "doc4": 0.4,
    "doc5": 0.2,
}
print("\nTest 3: nDCG@K")
print(f"Retrieved: {retrieved}")
print(f"Relevance judgments: {relevance_judgments}")
for k in [1, 3, 5]:
    ndcg = ndcg_at_k(retrieved, relevance_judgments, k)
    print(f"  nDCG@{k}: {ndcg:.3f}")

# Test 4: Perfect ranking
print("\nTest 4: Perfect Ranking (should get nDCG=1.0)")
perfect_retrieved = ["doc1", "doc2", "doc3"]
ndcg_perfect = ndcg_at_k(perfect_retrieved, relevance_judgments, 3)
print(f"  nDCG@3: {ndcg_perfect:.3f}")

print("\n" + "=" * 60)
print("All evaluation metrics tests completed!")
