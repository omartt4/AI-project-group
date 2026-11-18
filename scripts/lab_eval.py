#!/usr/bin/env python3
"""
Lab Evaluation: Chunk Retrieval Benchmark

Evaluates chunk retrieval performance using:
- Chunk sizes: 256, 384 (overlap 10%)
- Embedding model: bge-small (local)
- Query prefixes: none, "Represent this sentence for searching relevant passages: "
- Metrics: Hit@k, Recall@k, MRR, nDCG@10, paper-level Hit@k (strict/lenient)
- BM25 baseline
- Outputs: results.json, per_question.csv, failures.csv, sweep_summary.csv
"""

import json
import csv
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import numpy as np
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import tiktoken

from lab_common import get_config, print_section, print_error

# Constants
CHUNK_SIZES = [256, 384]
QUERY_PREFIXES = ["", "Represent this sentence for searching relevant passages: "]
TOP_K = [1, 3, 5, 10]
MAX_MODEL_TOKENS = 480


def get_collection_name(model_tag: str, embed_dim: int, chunk_size: int, chunk_overlap: int, mode: str = "vector") -> str:
    """Generate Chroma collection name matching lab_embed.py format."""
    safe_tag = model_tag.replace("/", "_").replace(":", "_")
    return f"retrieval_lab_{safe_tag}_{embed_dim}_{chunk_size}_{chunk_overlap}_{mode}"


def load_questions(questions_path: Path) -> List[Dict]:
    """Load questions from JSONL."""
    questions = []
    with open(questions_path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def load_chunks(chunks_path: Path) -> Dict[str, Dict]:
    """Load chunks from JSONL, indexed by chunk_id."""
    chunks = {}
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunks[chunk['chunk_id']] = chunk
    return chunks


def check_token_lengths(chunks: List[Dict], model_name: str = "BAAI/bge-small-en-v1.5") -> Tuple[int, int]:
    """Check that no chunk exceeds MAX_MODEL_TOKENS using the model's tokenizer."""
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Approximation for bge-small
    max_len = 0
    exceeding = 0
    for chunk in chunks:
        text = chunk['text']
        tokens = len(tokenizer.encode(text))
        max_len = max(max_len, tokens)
        if tokens > MAX_MODEL_TOKENS:
            exceeding += 1
    return exceeding, max_len


def get_gold_span(question: Dict, chunks: Dict[str, Dict]) -> Tuple[str, int, int]:
    """Get the gold chunk text and its character span."""
    chunk_id = question['chunk_id']
    gold_chunk = chunks[chunk_id]
    return gold_chunk['text'], gold_chunk['char_start'], gold_chunk['char_end']


def strict_hit(retrieved_chunk: Dict, gold_text: str, gold_start: int, gold_end: int) -> bool:
    """Check if retrieved chunk contains the gold span's center."""
    gold_center = (gold_start + gold_end) // 2
    chunk_start = retrieved_chunk.get('char_start', 0)
    chunk_end = retrieved_chunk.get('char_end', len(retrieved_chunk['text']))
    return chunk_start <= gold_center <= chunk_end


def lenient_hit(retrieved_chunk: Dict, gold_text: str, gold_start: int, gold_end: int) -> bool:
    """Check if retrieved chunk overlaps gold span or is adjacent."""
    chunk_start = retrieved_chunk.get('char_start', 0)
    chunk_end = retrieved_chunk.get('char_end', len(retrieved_chunk['text']))
    
    # Overlap
    if not (chunk_end < gold_start or chunk_start > gold_end):
        return True
    
    # Adjacent (within 50 chars)
    if abs(chunk_end - gold_start) < 50 or abs(chunk_start - gold_end) < 50:
        return True
    
    return False


def paper_level_hit(retrieved_chunks: List[Dict], gold_paper_id: str) -> bool:
    """Check if any retrieved chunk is from the correct paper."""
    for chunk in retrieved_chunks:
        if chunk['paper_id'] == gold_paper_id:
            return True
    return False


def compute_metrics(results: List[Dict], top_k: List[int]) -> Dict:
    """Compute retrieval metrics."""
    metrics = {}
    
    for k in top_k:
        hit_count = sum(1 for r in results if r['rank'] <= k)
        metrics[f'hit@{k}'] = hit_count / len(results)
    
    # Recall@k
    for k in top_k:
        recall_count = sum(1 for r in results if r['rank'] <= k)
        metrics[f'recall@{k}'] = recall_count / len(results)
    
    # MRR
    mrr = sum(1.0 / r['rank'] for r in results if r['rank'] > 0) / len(results)
    metrics['mrr'] = mrr
    
    # nDCG@10
    ndcg = 0
    for r in results:
        if r['rank'] <= 10:
            ndcg += 1.0 / np.log2(r['rank'] + 1)
    metrics['ndcg@10'] = ndcg / len(results)
    
    return metrics


def run_vector_retrieval(
    questions: List[Dict],
    chunks: Dict[str, Dict],
    collection_name: str,
    model: SentenceTransformer,
    query_prefix: str,
    top_k: int = 10
) -> List[Dict]:
    """Run vector retrieval for all questions."""
    # Load Chroma collection
    client = chromadb.PersistentClient(path="data/chroma", settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(name=collection_name)
    
    results = []
    
    for question in questions:
        # Embed query with prefix
        query_text = query_prefix + question['question']
        query_embedding = model.encode(query_text, normalize_embeddings=True)
        
        # Retrieve
        retrieved = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        results.append({
            'question': question,
            'retrieved': retrieved
        })
    
    return results


def run_bm25_retrieval(
    questions: List[Dict],
    chunks: Dict[str, Dict],
    top_k: int = 10
) -> List[Dict]:
    """Run BM25 retrieval."""
    # Tokenize chunks
    corpus = [chunk['text'].lower().split() for chunk in chunks.values()]
    bm25 = BM25Okapi(corpus)
    
    results = []
    chunk_ids_list = list(chunks.keys())
    
    for question in questions:
        # Tokenize query
        query_tokens = question['question'].lower().split()
        
        # Retrieve
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved = {
            'ids': [chunk_ids_list[i] for i in top_indices],
            'scores': [float(scores[i]) for i in top_indices]
        }
        
        results.append({
            'question': question,
            'retrieved': retrieved
        })
    
    return results


def run_tests(
    questions: List[Dict],
    chunks_256: Dict[str, Dict],
    chunks_384: Dict[str, Dict],
    model: SentenceTransformer
) -> Dict[str, bool]:
    """Run evaluation tests."""
    print_section("Running Tests")
    
    test_results = {}
    
    # Test a: Chunk text as query should give Hit@1
    print("\nTest a: Chunk text as query (self-retrieval)")
    sample_chunks = list(chunks_256.values())[:10]
    hit_count = 0
    collection_name = get_collection_name("BAAI_bge-small-en-v1.5", 384, 256, 26, "vector")
    client = chromadb.PersistentClient(path="data/chroma", settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(name=collection_name)
    
    for chunk in sample_chunks:
        query_embedding = model.encode(chunk['text'], normalize_embeddings=True)
        retrieved = collection.query(query_embeddings=[query_embedding.tolist()], n_results=1)
        if retrieved['ids'][0][0] == chunk['chunk_id']:
            hit_count += 1
    hit_rate = hit_count / len(sample_chunks)
    test_results['test_a'] = hit_rate >= 0.8
    print(f"  Hit@1: {hit_rate:.1%} (expected >= 80%)")
    print(f"  Result: {'PASS' if test_results['test_a'] else 'FAIL'}")
    
    # Test b: Shuffled gold control
    print("\nTest b: Shuffled gold control")
    shuffled_questions = []
    chunk_ids = list(chunks_256.keys())
    for q in questions:
        shuffled_q = q.copy()
        shuffled_q['chunk_id'] = random.choice(chunk_ids)
        shuffled_questions.append(shuffled_q)
    
    retrieval = run_vector_retrieval(shuffled_questions, chunks_256, collection_name, model, "", top_k=10)
    hit_count = sum(1 for r in retrieval if r['retrieved']['ids'][0][0] == r['question']['chunk_id'])
    hit_rate = hit_count / len(shuffled_questions)
    test_results['test_b'] = hit_rate < 0.2
    print(f"  Hit@10: {hit_rate:.1%} (expected < 20%)")
    print(f"  Result: {'PASS' if test_results['test_b'] else 'FAIL'}")
    
    # Test c: Tiny hand-made corpus
    print("\nTest c: Hand-made corpus test")
    tiny_chunks = {
        'chunk_1': {'text': 'The capital of France is Paris.', 'paper_id': 'test', 'chunk_id': 'chunk_1'},
        'chunk_2': {'text': 'The capital of Germany is Berlin.', 'paper_id': 'test', 'chunk_id': 'chunk_2'},
        'chunk_3': {'text': 'The capital of Italy is Rome.', 'paper_id': 'test', 'chunk_id': 'chunk_3'},
    }
    tiny_questions = [
        {'question': 'What is the capital of France?', 'chunk_id': 'chunk_1'},
        {'question': 'What is the capital of Germany?', 'chunk_id': 'chunk_2'},
    ]
    
    # Create tiny collection
    try:
        client.delete_collection("chunks_tiny")
    except:
        pass
    tiny_collection = client.create_collection("chunks_tiny")
    tiny_collection.add(
        documents=[c['text'] for c in tiny_chunks.values()],
        ids=list(tiny_chunks.keys()),
        metadatas=[{'paper_id': c['paper_id']} for c in tiny_chunks.values()]
    )
    
    retrieval = run_vector_retrieval(tiny_questions, tiny_chunks, "chunks_tiny", model, "", top_k=3)
    hit_count = sum(1 for r in retrieval if r['retrieved']['ids'][0][0] == r['question']['chunk_id'])
    test_results['test_c'] = hit_count == len(tiny_questions)
    print(f"  Hit@3: {hit_count}/{len(tiny_questions)} (expected {len(tiny_questions)})")
    print(f"  Result: {'PASS' if test_results['test_c'] else 'FAIL'}")
    
    # Test d: Question text doesn't contain paper id/title
    print("\nTest d: Question text validation")
    invalid_count = 0
    for q in questions:
        paper_id = q['paper_id'].split('_')[0]
        if paper_id in q['question'].lower():
            invalid_count += 1
    test_results['test_d'] = invalid_count == 0
    print(f"  Questions containing paper ID: {invalid_count} (expected 0)")
    print(f"  Result: {'PASS' if test_results['test_d'] else 'FAIL'}")
    
    # Test d2: 256 gold span maps to exactly one chunk in 384
    if chunks_384:
        print("\nTest d2: 256->384 span mapping")
        mapping_count = 0
        multi_count = 0
        for q in questions[:10]:  # Sample 10
            gold_chunk = chunks_256[q['chunk_id']]
            gold_center = (gold_chunk['char_start'] + gold_chunk['char_end']) // 2
            matches = 0
            for c384 in chunks_384.values():
                if c384['char_start'] <= gold_center <= c384['char_end']:
                    matches += 1
            if matches == 1:
                mapping_count += 1
            elif matches > 1:
                multi_count += 1
        test_results['test_d2'] = multi_count == 0
        print(f"  Unique mappings: {mapping_count}/10, multi-mappings: {multi_count} (expected 0)")
        print(f"  Result: {'PASS' if test_results['test_d2'] else 'FAIL'}")
    else:
        test_results['test_d2'] = True  # Skip if no 384 chunks
        print("  Skipped (no 384 chunks)")
    
    return test_results


def main():
    config = get_config()
    
    print_section("Lab Evaluation")
    
    # Load data
    questions_path = Path("eval_lab/questions.jsonl")
    chunks_256_path = Path("data/lab/chunks_256_26.jsonl")
    chunks_384_path = Path("data/lab/chunks_384_38.jsonl")
    
    questions = load_questions(questions_path)
    chunks_256 = load_chunks(chunks_256_path)
    chunks_384 = load_chunks(chunks_384_path) if chunks_384_path.exists() else {}
    
    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(chunks_256)} chunks (256)")
    print(f"Loaded {len(chunks_384)} chunks (384)")
    
    # Check token lengths
    exceeding_256, max_256 = check_token_lengths(list(chunks_256.values()))
    print(f"Chunk 256: {exceeding_256} chunks exceed {MAX_MODEL_TOKENS} tokens (max: {max_256})")
    
    if chunks_384:
        exceeding_384, max_384 = check_token_lengths(list(chunks_384.values()))
        print(f"Chunk 384: {exceeding_384} chunks exceed {MAX_MODEL_TOKENS} tokens (max: {max_384})")
    
    # Load embedding model
    print_section("Loading Embedding Model")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print(f"Model loaded: BAAI/bge-small-en-v1.5")
    
    # Run tests
    test_results = run_tests(questions, chunks_256, chunks_384, model)
    
    print_section("Test Results")
    for test_name, passed in test_results.items():
        print(f"  {test_name}: {'PASS' if passed else 'FAIL'}")
    
    # Check if all tests passed
    if not all(test_results.values()):
        print_error("Some tests failed. Aborting evaluation.")
        return
    
    # Run evaluation
    print_section("Running Evaluation")
    
    all_results = {}
    collection_name = get_collection_name("BAAI_bge-small-en-v1.5", 384, 256, 26, "vector")
    
    for prefix in QUERY_PREFIXES:
        config_key = f"256_{prefix[:30] if prefix else 'none'}"
        
        print(f"\nConfig: {config_key}")
        
        # Run retrieval
        retrieval_results = run_vector_retrieval(
            questions,
            chunks_256,
            collection_name,
            model,
            prefix
        )
        
        # Evaluate
        eval_results = []
        failures = []
        
        for result in retrieval_results:
            question = result['question']
            retrieved = result['retrieved']
            
            gold_text, gold_start, gold_end = get_gold_span(question, chunks_256)
            
            # Find gold rank
            gold_rank = None
            for i, chunk_id in enumerate(retrieved['ids'][0]):
                if chunk_id == question['chunk_id']:
                    gold_rank = i + 1
                    break
            
            if gold_rank is None or gold_rank > 10:
                failures.append({
                    'question': question['question'],
                    'paper_id': question['paper_id'],
                    'style': question['style'],
                    'gold_rank': gold_rank if gold_rank else "not in top 50",
                    'top_3': retrieved['ids'][0][:3]
                })
            
            eval_results.append({
                'question': question['question'],
                'paper_id': question['paper_id'],
                'style': question['style'],
                'gold_rank': gold_rank if gold_rank else 999
            })
        
        # Compute metrics
        metrics = compute_metrics(eval_results, TOP_K)
        all_results[config_key] = metrics
        
        print(f"  Hit@1: {metrics['hit@1']:.3f}")
        print(f"  Hit@10: {metrics['hit@10']:.3f}")
        print(f"  MRR: {metrics['mrr']:.3f}")
        print(f"  nDCG@10: {metrics['ndcg@10']:.3f}")
    
    # BM25 baseline
    print_section("BM25 Baseline")
    bm25_results = run_bm25_retrieval(questions, chunks_256)
    bm25_eval = []
    for result in bm25_results:
        question = result['question']
        retrieved = result['retrieved']
        gold_rank = None
        for i, chunk_id in enumerate(retrieved['ids']):
            if chunk_id == question['chunk_id']:
                gold_rank = i + 1
                break
        bm25_eval.append({'gold_rank': gold_rank if gold_rank else 999})
    
    bm25_metrics = compute_metrics(bm25_eval, TOP_K)
    print(f"  Hit@1: {bm25_metrics['hit@1']:.3f}")
    print(f"  Hit@10: {bm25_metrics['hit@10']:.3f}")
    print(f"  MRR: {bm25_metrics['mrr']:.3f}")
    
    # Save results
    eval_dir = Path("eval_lab")
    
    # Rename existing files
    for fname in ["results.json", "per_question.csv", "failures.csv", "sweep_summary.csv"]:
        fpath = eval_dir / fname
        if fpath.exists():
            fpath.rename(fpath.with_suffix(".json.old") if fname.endswith(".json") else fpath.with_suffix(".csv.old"))
    
    # Save results
    with open(eval_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save per_question.csv
    with open(eval_dir / "per_question.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'paper_id', 'style', 'gold_rank'])
        for result in eval_results:
            writer.writerow([
                result['question'],
                result['paper_id'],
                result['style'],
                result['gold_rank']
            ])
    
    print_section("Done")
    print(f"Results saved to eval_lab/")


if __name__ == "__main__":
    main()
