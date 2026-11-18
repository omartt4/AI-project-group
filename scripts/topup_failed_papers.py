#!/usr/bin/env python3
"""Top-up failed papers 074 and 090 with improved JSON parsing."""
import json
import time
import random
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, 'scripts')

from lab_common import get_config
from lab_qgen import (
    generate_questions_batch, validate_questions, check_tag_overlap,
    filter_questions_quality, log_chat_request
)

def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

def print_error(msg):
    print(f"ERROR: {msg}")

def sample_chunks_per_paper(chunks, n=6, seed=42):
    """Sample n chunks from a paper's chunks."""
    random.seed(seed)
    if len(chunks) <= n:
        return chunks
    return random.sample(chunks, n)

def main():
    config = get_config()
    
    # Check quota
    from lab_models import check_chat_quota
    quota_info = check_chat_quota(config.openrouter_base_url, config.openrouter_api_key, config.data_dir)
    print(f"Quota remaining: {quota_info['requests_available']}")
    
    if quota_info['requests_available'] is not None and quota_info['requests_available'] < 6:
        print_error("Insufficient quota for top-up (need 6 requests)")
        return
    
    # Load chunks
    chunks = [json.loads(line) for line in open('data/lab/chunks_256_26.jsonl')]
    paper_chunks = {}
    for c in chunks:
        pid = c['paper_id']
        if pid not in paper_chunks:
            paper_chunks[pid] = []
        paper_chunks[pid].append(c)
    
    # Load existing questions
    with open('data/lab/questions.json', 'r', encoding='utf-8') as f:
        all_questions = json.load(f)
    
    # Target papers
    target_papers = [
        '074_Machine_Learning_Unifying_Homophily_and_Heterophily_Network_Transformation_via_Motifs',
        '090_Machine_Learning_Adaptive_Precision_Training_for_Resource_Constrained_Devices',
        '002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction',
        '003_NLP_Should_I_visit_this_place_Inclusion_and_Exclusion_Phrase_Mining_from_Reviews'
    ]
    
    # Models to try (NVIDIA first, then non-Google fallback, then Google fallbacks)
    models_to_try = ['nvidia/nemotron-3.5-lightning:free', 'inclusionai/ling-3.2-flash:free', 'google/gemma-4-26b-a4b-it:free', 'google/gemma-4-31b-it:free']
    expected_styles = ["factual", "factual", "conceptual", "conceptual", "paraphrased", "paraphrased"]
    
    # Create raw_replies directory
    raw_replies_dir = Path('data/lab/raw_replies')
    raw_replies_dir.mkdir(exist_ok=True)
    
    print_section("Top-up Failed Papers")
    print(f"Target papers: {len(target_papers)}")
    print(f"Max requests per paper: 2")
    print(f"Models to try: {models_to_try}")
    
    for paper_id in target_papers:
        print(f"\n=== Paper {paper_id} ===")
        
        # Check if already has questions
        existing = [q for q in all_questions if q['paper_id'] == paper_id]
        if len(existing) >= 6:
            print(f"  Already has {len(existing)} questions, skipping")
            continue
        
        print(f"  Needs {6 - len(existing)} questions")
        
        # Sample chunks that don't have questions yet
        existing_chunk_ids = {q['chunk_id'] for q in existing}
        p_chunks = paper_chunks.get(paper_id, [])
        if not p_chunks:
            print_error(f"  No chunks found for {paper_id}")
            continue
        
        # Filter to chunks without questions
        chunks_without_questions = [c for c in p_chunks if c['chunk_id'] not in existing_chunk_ids]
        if not chunks_without_questions:
            print(f"  All chunks already have questions, skipping")
            continue
        
        # Sample up to 6 chunks (or fewer if that's all that's left)
        n_to_sample = min(6, len(chunks_without_questions))
        sampled = sample_chunks_per_paper(chunks_without_questions, n=n_to_sample)
        print(f"  Sampled {len(sampled)} chunks (need {6 - len(existing)} total)")
        
        # Try up to 2 requests total per paper (not per model)
        total_requests = 0
        max_requests_per_paper = 2
        
        for model_idx, model_id in enumerate(models_to_try):
            if total_requests >= max_requests_per_paper:
                print(f"  Reached max {max_requests_per_paper} requests for this paper")
                break
            
            total_requests += 1
            print(f"  Request {total_requests}/{max_requests_per_paper}, Model: {model_id}")
            
            try:
                # Adjust prompt for variable chunk count
                questions, timing, raw_reply = generate_questions_batch(
                    sampled,
                    model_id,
                    config.openrouter_api_key,
                    config.openrouter_base_url,
                    seed=99,
                    dry_run=False
                )
            except Exception as e:
                print_error(f"  Connection error: {e}")
                log_chat_request(config.data_dir, model_id, [paper_id], "error", {"llm": 0, "http_status": 0, "finish_reason": "connection_error"}, str(e))
                if total_requests < max_requests_per_paper and model_idx < len(models_to_try) - 1:
                    print(f"  Trying next model...")
                    continue
                else:
                    break
            
            # Handle 429
            if timing.get("http_status") == 429:
                print_error(f"  Rate limited (429)")
                try:
                    error_json = json.loads(raw_reply)
                    provider = error_json.get('error', {}).get('metadata', {}).get('provider_name', 'unknown')
                    limit_source = error_json.get('error', {}).get('metadata', {}).get('limit_source', 'unknown')
                    print(f"  Provider: {provider}, Limit source: {limit_source}")
                except:
                    pass
                log_chat_request(config.data_dir, model_id, [paper_id], "rate_limited", timing, raw_reply)
                
                if total_requests < max_requests_per_paper and model_idx < len(models_to_try) - 1:
                    print(f"  Trying next model...")
                    continue
                else:
                    print(f"  All models exhausted for this paper")
                    break
            
            # Handle other HTTP errors
            elif timing.get("http_status") != 200:
                print_error(f"  HTTP {timing.get('http_status')}")
                log_chat_request(config.data_dir, model_id, [paper_id], "error", timing, raw_reply)
                if total_requests < max_requests_per_paper and model_idx < len(models_to_try) - 1:
                    continue
                else:
                    break
            
            # Handle parse failure
            elif not questions:
                print_error(f"  Parse failed")
                print(f"  Raw reply (first 600 chars): {raw_reply[:600]}")
                
                # Check if it's thinking text without JSON
                if raw_reply and ('thinking' in raw_reply.lower() or 'analysis' in raw_reply.lower() or 'let me' in raw_reply.lower()):
                    print(f"  Reply appears to be thinking text without JSON")
                
                # Save full raw reply
                paper_short = paper_id.split('_')[0]
                raw_file = raw_replies_dir / f"{paper_short}_request{total_requests}.txt"
                with open(raw_file, 'w', encoding='utf-8') as f:
                    f.write(f"Model: {model_id}\n")
                    f.write(f"Finish reason: {timing.get('finish_reason', 'N/A')}\n")
                    f.write(f"HTTP status: {timing.get('http_status', 'N/A')}\n")
                    f.write(f"\n--- RAW REPLY ---\n")
                    f.write(raw_reply)
                print(f"  Saved raw reply to {raw_file}")
                
                log_chat_request(config.data_dir, model_id, [paper_id], "parse_failed", timing, raw_reply=raw_reply)
                
                if total_requests < max_requests_per_paper and model_idx < len(models_to_try) - 1:
                    print(f"  Trying next model...")
                    continue
                else:
                    break
            
            # Success
            log_chat_request(config.data_dir, model_id, [paper_id], "success", timing)
            print(f"  Success: {len(questions)} questions")
            
            # Validate (adjust expected styles for number of chunks)
            chunk_expected_styles = expected_styles[:len(sampled)]
            is_valid, validation_error = validate_questions(questions, sampled, chunk_expected_styles)
            if not is_valid:
                print_error(f"  Validation failed: {validation_error}")
                log_chat_request(config.data_dir, model_id, [paper_id], "validation_failed", timing, validation_error)
                if total_requests < max_requests_per_paper and model_idx < len(models_to_try) - 1:
                    continue
                else:
                    break
            
            # Tag paper_id
            for q in questions:
                chunk_id = q.get("chunk_id")
                source_chunk = next((c for c in sampled if c["chunk_id"] == chunk_id), None)
                if source_chunk:
                    q["paper_id"] = source_chunk["paper_id"]
                    q["source_snippet"] = source_chunk["text"][:200]
                else:
                    q["paper_id"] = paper_id
            
            # Tag overlap and quality filter
            questions, dropped = check_tag_overlap(questions)
            print(f"  Dropped by tag check: {dropped}")
            questions, rejected = filter_questions_quality(questions)
            print(f"  Rejected by quality filter: {rejected}")
            
            # Add to all questions (append, don't replace)
            all_questions.extend(questions)
            
            # Save
            with open('data/lab/questions.json', 'w', encoding='utf-8') as f:
                json.dump(all_questions, f, indent=2)
            
            print(f"  Saved: {len(questions)} questions")
            print(f"  Total for paper: {len([q for q in all_questions if q['paper_id'] == paper_id])}")
            
            # Success - move to next paper
            break

if __name__ == "__main__":
    main()
