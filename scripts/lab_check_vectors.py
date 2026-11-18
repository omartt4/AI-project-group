"""
Phase 3: Vector sanity checks.
Validates ChromaDB collection with 6 checks: count, dimensions, duplicates, self-retrieval, neighbours, distribution.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import chromadb
import numpy as np

from lab_common import (
    get_config,
    print_section,
    print_warning,
    print_error
)


def get_collection_name(model_tag: str, embed_dim: int, chunk_size: int, chunk_overlap: int, mode: str) -> str:
    """Generate Chroma collection name with model and dimension."""
    safe_tag = model_tag.replace("/", "_").replace(":", "_")
    return f"retrieval_lab_{safe_tag}_{embed_dim}_{chunk_size}_{chunk_overlap}_{mode}"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def check_1_count_and_ids(collection) -> Dict:
    """Check 1: Chroma count matches expected, IDs are unique."""
    print("\n[CHECK 1] Count and ID uniqueness")
    
    count = collection.count()
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    
    ids = data["ids"]
    unique_ids = set(ids)
    
    print(f"  Total vectors: {count}")
    print(f"  Unique IDs: {len(unique_ids)}")
    
    passed = count == len(unique_ids)
    
    if passed:
        print("  ✓ PASS: All IDs are unique")
    else:
        print(f"  ✗ FAIL: {count - len(unique_ids)} duplicate IDs")
    
    return {
        "check": "count_and_ids",
        "passed": passed,
        "total_vectors": count,
        "unique_ids": len(unique_ids),
        "duplicates": count - len(unique_ids)
    }


def check_2_dimensions(collection) -> Dict:
    """Check 2: All vectors have same dimension, no NaN/inf, no all-zero vectors."""
    print("\n[CHECK 2] Dimensions and vector validity")
    
    data = collection.get(include=["embeddings"])
    embeddings = data["embeddings"]
    
    if len(embeddings) == 0:
        print("  ✗ FAIL: No embeddings found")
        return {"check": "dimensions", "passed": False}
    
    # Check dimensions
    dims = [len(emb) for emb in embeddings]
    unique_dims = set(dims)
    
    print(f"  Dimension: {list(unique_dims)}")
    
    if len(unique_dims) != 1:
        print(f"  ✗ FAIL: Inconsistent dimensions")
        return {"check": "dimensions", "passed": False}
    
    dim = list(unique_dims)[0]
    
    # Check for NaN/inf
    arr = np.array(embeddings)
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    
    # Check for all-zero vectors
    norms = np.linalg.norm(arr, axis=1)
    zero_count = int(np.sum(norms == 0))
    
    print(f"  All-zero vectors: {zero_count}")
    
    passed = (not has_nan) and (not has_inf) and (zero_count == 0)
    
    if passed:
        print("  ✓ PASS: All vectors valid")
    else:
        print("  ✗ FAIL: Invalid vectors found")
    
    return {
        "check": "dimensions",
        "passed": bool(passed),
        "dimension": dim,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "zero_vectors": zero_count
    }


def check_3_near_duplicates(collection, threshold: float = 0.98) -> Dict:
    """Check 3: Find near-duplicate vectors (cosine > threshold)."""
    print(f"\n[CHECK 3] Near-duplicates (cosine > {threshold})")
    
    data = collection.get(include=["embeddings"])
    embeddings = np.array(data["embeddings"])
    
    n = len(embeddings)
    if n > 500:
        # Sample for large collections
        print(f"  Sampling 500 vectors from {n}")
        indices = random.sample(range(n), 500)
        embeddings = embeddings[indices]
        n = 500
    
    # Compute pairwise similarities (upper triangle only)
    duplicates = []
    
    # Normalize for efficient cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    similarities = np.dot(normalized, normalized.T)
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarities[i, j]
            if sim > threshold:
                duplicates.append((i, j, sim))
    
    print(f"  Near-duplicate pairs: {len(duplicates)}")
    
    if len(duplicates) > 0:
        print(f"  Sample duplicates:")
        for idx1, idx2, sim in duplicates[:5]:
            print(f"    [{idx1}] <-> [{idx2}]: {sim:.4f}")
    
    passed = len(duplicates) < n * 0.01  # Less than 1% duplicates
    
    if passed:
        print("  ✓ PASS: Acceptable duplicate rate")
    else:
        print(f"  ✗ FAIL: Too many duplicates ({len(duplicates)}/{n})")
    
    return {
        "check": "near_duplicates",
        "passed": bool(passed),
        "duplicate_pairs": len(duplicates),
        "total_compared": n,
        "duplicate_rate": len(duplicates) / n if n > 0 else 0
    }


def check_4_self_retrieval(collection, sample_size: int = 100) -> Dict:
    """Check 4: Self-retrieval - 100 random chunks should return themselves at rank 1."""
    print(f"\n[CHECK 4] Self-retrieval (sample {sample_size})")
    
    data = collection.get(include=["embeddings"])
    embeddings = np.array(data["embeddings"])
    
    n = len(embeddings)
    if n < sample_size:
        sample_size = n
    
    indices = random.sample(range(n), sample_size)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    correct = 0
    
    for idx in indices:
        query = normalized[idx]
        # Compute similarities
        similarities = np.dot(normalized, query)
        # Get top-k (excluding self from consideration? No, include it)
        top_indices = np.argsort(similarities)[::-1][:5]
        
        # Check if self is rank 1
        if top_indices[0] == idx:
            correct += 1
    
    accuracy = correct / sample_size
    print(f"  Self-retrieval accuracy: {correct}/{sample_size} ({100*accuracy:.1f}%)")
    
    passed = accuracy >= 0.95
    
    if passed:
        print("  ✓ PASS: Self-retrieval works")
    else:
        print("  ✗ FAIL: Self-retrieval accuracy too low")
    
    return {
        "check": "self_retrieval",
        "passed": bool(passed),
        "correct": correct,
        "sample_size": sample_size,
        "accuracy": float(accuracy)
    }


def check_5_neighbour_analysis(collection, sample_size: int = 20) -> Dict:
    """Check 5: Show 20 random chunks with their top-3 neighbours and papers."""
    print(f"\n[CHECK 5] Neighbour analysis (sample {sample_size})")
    
    data = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(data["embeddings"])
    metadatas = data["metadatas"]
    
    n = len(embeddings)
    if n < sample_size:
        sample_size = n
    
    indices = random.sample(range(n), sample_size)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    print("  Sample chunks with top-3 neighbours:")
    
    for idx in indices[:5]:  # Show first 5
        query_paper = metadatas[idx]["paper_id"]
        
        similarities = np.dot(normalized, normalized[idx])
        top_indices = np.argsort(similarities)[::-1][1:4]  # Skip self
        
        neighbours = []
        for nidx in top_indices:
            neighbours.append({
                "index": nidx,
                "paper": metadatas[nidx]["paper_id"],
                "similarity": float(similarities[nidx])
            })
        
        print(f"    [{idx}] ({query_paper})")
        for nb in neighbours:
            print(f"      -> [{nb['index']}] ({nb['paper']}): {nb['similarity']:.4f}")
    
    print("  ✓ INFO: Neighbour analysis complete")
    
    return {
        "check": "neighbour_analysis",
        "passed": True,  # This is informational
        "sample_size": sample_size
    }


def check_6_similarity_distribution(collection, sample_size: int = 100) -> Dict:
    """Check 6: Distribution of top-1 similarity scores."""
    print(f"\n[CHECK 6] Similarity distribution (sample {sample_size})")
    
    data = collection.get(include=["embeddings"])
    embeddings = np.array(data["embeddings"])
    
    n = len(embeddings)
    if n < sample_size:
        sample_size = n
    
    indices = random.sample(range(n), sample_size)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    top1_scores = []
    
    for idx in indices:
        query = normalized[idx]
        similarities = np.dot(normalized, query)
        # Get top-1 (excluding self)
        similarities[idx] = -1  # Exclude self
        top1 = np.max(similarities)
        top1_scores.append(top1)
    
    top1_scores = np.array(top1_scores)
    
    print(f"  Top-1 similarity stats:")
    print(f"    Min: {np.min(top1_scores):.4f}")
    print(f"    Mean: {np.mean(top1_scores):.4f}")
    print(f"    Max: {np.max(top1_scores):.4f}")
    print(f"    Std: {np.std(top1_scores):.4f}")
    
    # Check if distribution looks reasonable (not all near 0 or 1)
    passed = (np.mean(top1_scores) > 0.3) and (np.mean(top1_scores) < 0.99)
    
    if passed:
        print("  ✓ PASS: Similarity distribution looks reasonable")
    else:
        print("  ✗ FAIL: Similarity distribution looks abnormal")
    
    return {
        "check": "similarity_distribution",
        "passed": bool(passed),
        "min": float(np.min(top1_scores)),
        "mean": float(np.mean(top1_scores)),
        "max": float(np.max(top1_scores)),
        "std": float(np.std(top1_scores))
    }


def run_all_checks(collection_name: str, config) -> Dict:
    """Run all 6 checks and return results."""
    print_section("Phase 3: Vector Sanity Checks")
    print(f"Collection: {collection_name}\n")
    
    chroma_client = chromadb.PersistentClient(path=config.chroma_dir)
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        print_error(f"Collection not found: {collection_name}")
        sys.exit(1)
    
    results = {
        "collection_name": collection_name,
        "checks": []
    }
    
    # Run all checks
    results["checks"].append(check_1_count_and_ids(collection))
    results["checks"].append(check_2_dimensions(collection))
    results["checks"].append(check_3_near_duplicates(collection))
    results["checks"].append(check_4_self_retrieval(collection))
    results["checks"].append(check_5_neighbour_analysis(collection))
    results["checks"].append(check_6_similarity_distribution(collection))
    
    # Summary
    print_section("Summary")
    passed = sum(1 for c in results["checks"] if c["passed"])
    total = len(results["checks"])
    
    print(f"Passed: {passed}/{total}")
    
    for check in results["checks"]:
        status = "✓" if check["passed"] else "✗"
        print(f"  {status} {check['check']}")
    
    if passed == total:
        print("\n✓ ALL CHECKS PASSED")
    else:
        print(f"\n✗ {total - passed} CHECK(S) FAILED")
    
    # Save results
    results_path = Path(config.data_dir) / f"vector_checks_{collection_name}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Vector sanity checks")
    parser.add_argument("--collection", required=True, help="Chroma collection name")
    args = parser.parse_args()
    
    config = get_config()
    
    results = run_all_checks(args.collection, config)
    
    # Exit with error if any check failed
    if not all(c["passed"] for c in results["checks"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
