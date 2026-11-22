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
    """Check that no chunk exceeds MAX_MODEL_TOKENS using the bge-small tokenizer with special tokens included."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = 0
    exceeding = 0
    for chunk in chunks:
        text = chunk['text']
        tokens = len(tokenizer(text, add_special_tokens=True)['input_ids'])
        max_len = max(max_len, tokens)
        if tokens > MAX_MODEL_TOKENS:
            exceeding += 1
    return exceeding, max_len


def get_gold_span(question: Dict, chunks: Dict[str, Dict]) -> Tuple[str, int, int]:
    """Get the gold chunk text and its character span."""
    chunk_id = question['chunk_id']
    gold_chunk = chunks[chunk_id]
    return gold_chunk['text'], gold_chunk['char_start'], gold_chunk['char_end']


def strict_hit(retrieved_chunk: Dict, gold_paper_id: str, gold_text: str, gold_start: int, gold_end: int) -> bool:
    """Check if retrieved chunk is from correct paper and contains the gold span's center."""
    # Must be from correct paper
    if retrieved_chunk.get('paper_id') != gold_paper_id:
        return False
    gold_center = (gold_start + gold_end) // 2
    chunk_start = retrieved_chunk.get('char_start', 0)
    chunk_end = retrieved_chunk.get('char_end', len(retrieved_chunk['text']))
    return chunk_start <= gold_center <= chunk_end


def lenient_hit(retrieved_chunk: Dict, gold_paper_id: str, gold_text: str, gold_start: int, gold_end: int) -> bool:
    """Check if retrieved chunk is from correct paper and overlaps gold span or is adjacent."""
    # Must be from correct paper
    if retrieved_chunk.get('paper_id') != gold_paper_id:
        return False
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


def compute_metrics_by_group(results: List[Dict], group_key: str, top_k: List[int]) -> Dict[str, Dict]:
    """Compute metrics grouped by a key (style or field)."""
    groups = {}
    for r in results:
        key = r[group_key]
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    group_metrics = {}
    for key, group_results in groups.items():
        group_metrics[key] = compute_metrics(group_results, top_k)
        group_metrics[key]['n'] = len(group_results)
    
    return group_metrics


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
    config = get_config()
    client = chromadb.PersistentClient(path=config.chroma_dir, settings=Settings(anonymized_telemetry=False))
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
    config = get_config()
    client = chromadb.PersistentClient(path=config.chroma_dir, settings=Settings(anonymized_telemetry=False))
    
    # Test a: Chunk text as query should give Hit@1 (for both 256 and 384)
    print("\nTest a: Chunk text as query (self-retrieval)")
    for chunk_size in [256, 384]:
        chunks = chunks_256 if chunk_size == 256 else chunks_384
        if not chunks:
            continue
        sample_chunks = list(chunks.values())[:10]
        hit_count = 0
        collection_name = get_collection_name("BAAI_bge-small-en-v1.5", 384, chunk_size, 26 if chunk_size == 256 else 38, "vector")
        collection = client.get_collection(name=collection_name)
        
        for chunk in sample_chunks:
            query_embedding = model.encode(chunk['text'], normalize_embeddings=True)
            retrieved = collection.query(query_embeddings=[query_embedding.tolist()], n_results=1)
            if retrieved['ids'][0][0] == chunk['chunk_id']:
                hit_count += 1
        hit_rate = hit_count / len(sample_chunks)
        test_results[f'test_a_{chunk_size}'] = hit_rate >= 0.8
        print(f"  {chunk_size}: Hit@1:", round(hit_rate, 1), "(expected >= 80%)")
        print(f"  Result:", "PASS" if test_results[f'test_a_{chunk_size}'] else "FAIL")
    
    # Test b: Shuffled gold control (for both 256 and 384)
    print("\nTest b: Shuffled gold control")
    for chunk_size in [256, 384]:
        chunks = chunks_256 if chunk_size == 256 else chunks_384
        if not chunks:
            continue
        shuffled_questions = []
        chunk_ids = list(chunks.keys())
        for q in questions:
            shuffled_q = q.copy()
            shuffled_q['chunk_id'] = random.choice(chunk_ids)
            shuffled_questions.append(shuffled_q)
        
        collection_name = get_collection_name("BAAI_bge-small-en-v1.5", 384, chunk_size, 26 if chunk_size == 256 else 38, "vector")
        retrieval = run_vector_retrieval(shuffled_questions, chunks, collection_name, model, "", top_k=10)
        hit_count = sum(1 for r in retrieval if r['retrieved']['ids'][0][0] == r['question']['chunk_id'])
        hit_rate = hit_count / len(shuffled_questions)
        test_results[f'test_b_{chunk_size}'] = hit_rate < 0.2
        print(f"  {chunk_size}: Hit@10:", round(hit_rate, 1), "(expected < 20%)")
        print(f"  Result:", "PASS" if test_results[f'test_b_{chunk_size}'] else "FAIL")
    
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
    
    # Create tiny collection (use config client)
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
    print("  Hit@3:", hit_count, "/", len(tiny_questions), "(expected", len(tiny_questions), ")")
    print("  Result: " + ("PASS" if test_results['test_c'] else "FAIL"))
    
    # Test d: Question text doesn't contain paper id/title
    print("\nTest d: Question text validation")
    invalid_count = 0
    for q in questions:
        paper_id = q['paper_id'].split('_')[0]
        if paper_id in q['question'].lower():
            invalid_count += 1
    test_results['test_d'] = invalid_count == 0
    print("  Questions containing paper ID:", invalid_count, "(expected 0)")
    print("  Result: " + ("PASS" if test_results['test_d'] else "FAIL"))
    
    # Test d2: 256 gold span maps to one or two same-paper 384 chunks.
    # Two matches are valid only when the center is in the 384-chunk overlap.
    print("\nTest d2: 256->384 span mapping")
    unique_count = 0
    overlap_count = 0
    invalid_count = 0
    zero_count = 0

    if chunks_384:
        for q in questions:
            gold_chunk = chunks_256[q['chunk_id']]
            gold_center = (gold_chunk['char_start'] + gold_chunk['char_end']) // 2
            matches = [
                c384 for c384 in chunks_384.values()
                if c384['paper_id'] == gold_chunk['paper_id']
                and c384['char_start'] <= gold_center <= c384['char_end']
            ]
            if len(matches) == 1:
                unique_count += 1
            elif len(matches) == 2:
                overlap_count += 1
            elif len(matches) == 0:
                zero_count += 1
            else:
                invalid_count += 1
    else:
        invalid_count = len(questions)

    test_results['test_d2'] = (
        bool(chunks_384)
        and zero_count == 0
        and invalid_count == 0
    )
    print("  Unique mappings:", unique_count, "/" + str(len(questions)))
    print("  Overlap mappings:", overlap_count, "/" + str(len(questions)) + " (allowed)")
    print("  Invalid mappings:", invalid_count, "/" + str(len(questions)) + " (must be 0)")
    print("  Zero-mappings:", zero_count, "/" + str(len(questions)) + " (must be 0)")
    print("  Result: " + ("PASS" if test_results['test_d2'] else "FAIL"))
    
    # Unit test: strict_hit should reject chunks from other papers
    print("\nUnit test: strict_hit paper_id check")
    test_chunk_other_paper = {
        'paper_id': 'other_paper',
        'char_start': 0,
        'char_end': 1000,
        'text': 'some text'
    }
    test_result = strict_hit(test_chunk_other_paper, 'gold_paper', 'gold text', 500, 600)
    if not test_result:
        print("  PASS: Chunk from other paper rejected even with correct offset")
    else:
        print("  FAIL: Chunk from other paper incorrectly accepted")
    
    test_chunk_correct_paper = {
        'paper_id': 'gold_paper',
        'char_start': 0,
        'char_end': 1000,
        'text': 'some text'
    }
    test_result = strict_hit(test_chunk_correct_paper, 'gold_paper', 'gold text', 500, 600)
    if test_result:
        print("  PASS: Chunk from correct paper accepted with correct offset")
    else:
        print("  FAIL: Chunk from correct paper incorrectly rejected")
    
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
    
    print("Loaded", len(questions), "questions")
    print("Loaded", len(chunks_256), "chunks (256)")
    print("Loaded", len(chunks_384), "chunks (384)")
    
    # Check token lengths
    exceeding_256, max_256 = check_token_lengths(list(chunks_256.values()))
    print("Chunk 256:", exceeding_256, "chunks exceed 480 tokens (bge-small tokenizer, max:", max_256, ")")
    print("  Note: Chunk size is defined in tiktoken (cl100k_base) tokens")
    
    if chunks_384:
        exceeding_384, max_384 = check_token_lengths(list(chunks_384.values()))
        print("Chunk 384:", exceeding_384, "chunks exceed 480 tokens (bge-small tokenizer, max:", max_384, ")")
    
    # Load embedding model
    print_section("Loading Embedding Model")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Model loaded: BAAI/bge-small-en-v1.5")
    
    # Run tests
    test_results = run_tests(questions, chunks_256, chunks_384, model)
    
    print_section("Test Results")
    for test_name, passed in test_results.items():
        print("  " + test_name + ":", "PASS" if passed else "FAIL")
    
    # Check if all tests passed
    if not all(test_results.values()):
        print_error("Some tests failed. Aborting evaluation.")
        return
    
    # Run evaluation
    print_section("Running Evaluation")
    
    all_results = {}
    all_eval_results = []
    all_failures = []
    
    for chunk_size in [256, 384]:
        chunks = chunks_256 if chunk_size == 256 else chunks_384
        if not chunks:
            continue
        collection_name = get_collection_name("BAAI_bge-small-en-v1.5", 384, chunk_size, 26 if chunk_size == 256 else 38, "vector")
        
        for prefix in QUERY_PREFIXES:
            if prefix:
                config_key = f"{chunk_size}_instruction"
            else:
                config_key = f"{chunk_size}_none"
            
            print("\nConfig:", config_key)
            
            # Run retrieval
            retrieval_results = run_vector_retrieval(
                questions,
                chunks,
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
                
                # Always use 256 chunks for gold span (questions reference 256 chunk_ids)
                gold_text, gold_start, gold_end = get_gold_span(question, chunks_256)
                
                # Find gold rank using hit rule (strict or lenient)
                gold_rank = None
                for i, chunk_id in enumerate(retrieved['ids'][0]):
                    if chunk_id in chunks:
                        retrieved_chunk = chunks[chunk_id]
                        # For 256 config, use exact chunk_id match
                        # For 384 config, use strict hit rule (gold span center in retrieved chunk)
                        if chunk_size == 256:
                            if chunk_id == question['chunk_id']:
                                gold_rank = i + 1
                                break
                        else:  # 384 config
                            if strict_hit(retrieved_chunk, question['paper_id'], gold_text, gold_start, gold_end):
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
                    'field': question['paper_id'].split('_')[1],
                    'style': question['style'],
                    'rank': gold_rank if gold_rank else 999,
                    'config': config_key
                })
            
            all_eval_results.extend(eval_results)
            all_failures.extend(failures)
            
            # Compute metrics
            metrics = compute_metrics(eval_results, TOP_K)
            all_results[config_key] = metrics
            
            print("  Hit@1:", round(metrics['hit@1'], 3))
            print("  Hit@3:", round(metrics['hit@3'], 3))
            print("  Hit@5:", round(metrics['hit@5'], 3))
            print("  Hit@10:", round(metrics['hit@10'], 3))
            print("  Recall@10:", round(metrics['recall@10'], 3))
            print("  MRR:", round(metrics['mrr'], 3))
            print("  nDCG@10:", round(metrics['ndcg@10'], 3))
    
    # BM25 baseline for both chunk sizes
    print_section("BM25 Baseline")
    
    for chunk_size in [256, 384]:
        chunks = chunks_256 if chunk_size == 256 else chunks_384
        if not chunks:
            continue
        
        print(f"\nBM25 {chunk_size}:")
        bm25_results = run_bm25_retrieval(questions, chunks)
        bm25_eval = []
        
        for result in bm25_results:
            question = result['question']
            retrieved = result['retrieved']
            
            # Always use 256 chunks for gold span
            gold_text, gold_start, gold_end = get_gold_span(question, chunks_256)
            
            # Find gold rank using strict hit rule
            gold_rank = None
            for i, chunk_id in enumerate(retrieved['ids']):
                if chunk_id in chunks:
                    retrieved_chunk = chunks[chunk_id]
                    if strict_hit(retrieved_chunk, question['paper_id'], gold_text, gold_start, gold_end):
                        gold_rank = i + 1
                        break
            
            bm25_eval.append({
                'rank': gold_rank if gold_rank else 999,
                'field': question['paper_id'].split('_')[1],
                'style': question['style']
            })
        
        bm25_metrics = compute_metrics(bm25_eval, TOP_K)
        all_results[f"BM25_{chunk_size}"] = bm25_metrics
        print("  Hit@1:", round(bm25_metrics['hit@1'], 3))
        print("  Hit@3:", round(bm25_metrics['hit@3'], 3))
        print("  Hit@5:", round(bm25_metrics['hit@5'], 3))
        print("  Hit@10:", round(bm25_metrics['hit@10'], 3))
        print("  Recall@10:", round(bm25_metrics['recall@10'], 3))
        print("  MRR:", round(bm25_metrics['mrr'], 3))
        print("  nDCG@10:", round(bm25_metrics['ndcg@10'], 3))
    
    # Print grouped metrics (skip BM25 configs here)
    for config_key in all_results.keys():
        if config_key.startswith('BM25'):
            continue
        config_eval = [r for r in all_eval_results if r.get('config') == config_key]
        
        print("\n--- " + config_key + " by Style (strict) ---")
        style_metrics = compute_metrics_by_group(config_eval, 'style', TOP_K)
        for style in ['factual', 'conceptual', 'paraphrased']:
            if style in style_metrics:
                m = style_metrics[style]
                print("  " + style + ": Hit@1=" + str(round(m['hit@1'], 3)) + ", Hit@10=" + str(round(m['hit@10'], 3)) + ", MRR=" + str(round(m['mrr'], 3)) + " (n=" + str(m['n']) + ")")
        
        print("\n--- " + config_key + " by Field (strict) ---")
        field_metrics = compute_metrics_by_group(config_eval, 'field', TOP_K)
        for field in ['NLP', 'Computer', 'Machine']:
            if field in field_metrics:
                m = field_metrics[field]
                print("  " + field + ": Hit@1=" + str(round(m['hit@1'], 3)) + ", Hit@10=" + str(round(m['hit@10'], 3)) + ", MRR=" + str(round(m['mrr'], 3)) + " (n=" + str(m['n']) + ")")
    
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
        writer.writerow(['question', 'paper_id', 'style', 'rank'])
        for result in all_eval_results:
            writer.writerow([
                result['question'],
                result['paper_id'],
                result['style'],
                result['rank']
            ])
    
    # Save failures.csv
    if all_failures:
        with open(eval_dir / "failures.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'paper_id', 'style', 'gold_rank', 'top_3'])
            for failure in all_failures:
                writer.writerow([
                    failure['question'],
                    failure['paper_id'],
                    failure['style'],
                    failure['gold_rank'],
                    failure['top_3']
                ])
    
    # Save sweep_summary.csv
    with open(eval_dir / "sweep_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'hit@1', 'hit@3', 'hit@5', 'hit@10', 'recall@10', 'mrr', 'ndcg@10'])
        for config_key, metrics in all_results.items():
            writer.writerow([
                config_key,
                round(metrics['hit@1'], 4),
                round(metrics.get('hit@3', 0), 4),
                round(metrics.get('hit@5', 0), 4),
                round(metrics['hit@10'], 4),
                round(metrics.get('recall@10', 0), 4),
                round(metrics['mrr'], 4),
                round(metrics.get('ndcg@10', 0), 4)
            ])
    
    print_section("Done")
    print("Results saved to eval_lab/")


if __name__ == "__main__":
    main()
