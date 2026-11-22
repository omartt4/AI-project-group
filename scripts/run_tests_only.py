
#!/usr/bin/env python3
"""Run only the lab evaluation tests (test d2), not the full evaluation."""
import json
from pathlib import Path

def load_questions(questions_path: Path):
    """Load questions from JSONL."""
    questions = []
    with open(questions_path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

def load_chunks(chunks_path: Path):
    """Load chunks from JSONL, indexed by chunk_id."""
    chunks = {}
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunks[chunk['chunk_id']] = chunk
    return chunks

def run_tests(questions, chunks_256, chunks_384):
    """Run tests d and d2."""
    test_results = {}
    
    # Test d: Questions don't contain paper ID
    print("Test d: Questions containing paper ID")
    invalid_count = 0
    for q in questions:
        paper_id = q['paper_id']
        if paper_id in q['question'].lower():
            invalid_count += 1
    test_results['test_d'] = invalid_count == 0
    print("  Questions containing paper ID:", invalid_count, "(expected 0)")
    print("  Result: " + ("PASS" if test_results['test_d'] else "FAIL"))
    
    # Test d2: 256 gold span maps to one or two same-paper 384 chunks.
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
    
    return test_results

if __name__ == "__main__":
    questions_path = Path("eval_lab/questions.jsonl")
    chunks_256_path = Path("data/lab/chunks_256_26.jsonl")
    chunks_384_path = Path("data/lab/chunks_384_38.jsonl")
    
    questions = load_questions(questions_path)
    chunks_256 = load_chunks(chunks_256_path)
    chunks_384 = load_chunks(chunks_384_path) if chunks_384_path.exists() else {}
    
    print("Loaded", len(questions), "questions")
    print("Loaded", len(chunks_256), "chunks (256)")
    print("Loaded", len(chunks_384), "chunks (384)")
    
    # Check token lengths using bge tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    
    max_256 = 0
    exceeding_256 = 0
    for chunk in chunks_256.values():
        tokens = len(tokenizer(chunk['text'], add_special_tokens=True)['input_ids'])
        max_256 = max(max_256, tokens)
        if tokens > 480:
            exceeding_256 += 1
    
    max_384 = 0
    exceeding_384 = 0
    for chunk in chunks_384.values():
        tokens = len(tokenizer(chunk['text'], add_special_tokens=True)['input_ids'])
        max_384 = max(max_384, tokens)
        if tokens > 480:
            exceeding_384 += 1
    
    print(f"Chunk 256: {exceeding_256} chunks exceed 480 tokens (bge tokenizer, max: {max_256})")
    print(f"Chunk 384: {exceeding_384} chunks exceed 480 tokens (bge tokenizer, max: {max_384})")
    print("  Note: Chunk size is defined in tiktoken (cl100k_base) tokens")
    
    test_results = run_tests(questions, chunks_256, chunks_384)
    
    print("\nTest Results:")
    for test_name, passed in test_results.items():
        print("  " + test_name + ":", "PASS" if passed else "FAIL")
