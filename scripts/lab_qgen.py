"""
Phase 5: Question Generation.
Batched LLM question generation from chunks.
One request per paper, returns 6 questions (2 factual, 2 conceptual, 2 paraphrased) as JSON.
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import tiktoken

from lab_common import (
    get_config,
    print_section,
    print_warning,
    print_error
)


def verify_all_models():
    """Import and run model verification."""
    from lab_models import main as verify_models_main
    old_argv = sys.argv
    sys.argv = ["lab_models.py"]
    try:
        try:
            verify_models_main()
        except SystemExit as e:
            if e.code != 0:
                sys.exit(e.code)
    finally:
        sys.argv = old_argv


def check_rate_limit(api_key: str, base_url: str) -> Dict:
    """Check rate limit info from /key endpoint."""
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(f"{base_url}/key", headers=headers)
    r.raise_for_status()
    data = r.json()
    
    # Print rate limit info
    print(f"Free tier: {data['data']['is_free_tier']}")
    print(f"Daily requests: {data['data']['free_model_daily_requests']['used']}/{data['data']['free_model_daily_requests']['limit']}")
    print(f"Remaining: {data['data']['free_model_daily_requests']['remaining']}")
    
    # Check for reset time if available
    if 'limit_reset' in data['data'] and data['data']['limit_reset']:
        reset_time = datetime.fromtimestamp(data['data']['limit_reset'], tz=timezone.utc)
        cairo_offset = timezone(timedelta(hours=2))
        cairo_time = reset_time.astimezone(cairo_offset)
        print(f"Limit reset: {reset_time} UTC / {cairo_time} UTC+2")
    
    return data


def sample_chunks_per_paper(chunks: List[Dict], n: int = 6) -> List[Dict]:
    """Sample n chunks per paper, distributed across page regions."""
    # Group by paper
    paper_chunks = {}
    for chunk in chunks:
        paper_id = chunk["paper_id"]
        if paper_id not in paper_chunks:
            paper_chunks[paper_id] = []
        paper_chunks[paper_id].append(chunk)
    
    # Sample n chunks per paper, preferring different page ranges
    sampled = []
    for paper_id, p_chunks in paper_chunks.items():
        if len(p_chunks) <= n:
            sampled.extend(p_chunks)
        else:
            # Sort by page_start to get distribution
            p_chunks_sorted = sorted(p_chunks, key=lambda x: x["page_start"])
            # Sample evenly distributed
            indices = [int(i * len(p_chunks_sorted) / n) for i in range(n)]
            sampled.extend([p_chunks_sorted[i] for i in indices])
    
    return sampled


def validate_questions(questions: List[Dict], chunks: List[Dict], expected_styles: List[str]) -> tuple[bool, str]:
    """
    Validate questions against expected chunks and styles.
    Returns (is_valid, error_message).
    """
    if len(questions) != len(chunks):
        return False, f"Expected {len(chunks)} questions, got {len(questions)}"
    
    expected_ids = {c["chunk_id"] for c in chunks}
    returned_ids = {q.get("chunk_id") for q in questions}
    
    if returned_ids != expected_ids:
        missing = expected_ids - returned_ids
        extra = returned_ids - expected_ids
        return False, f"Chunk ID mismatch. Missing: {missing}, Extra: {extra}"
    
    # Validate styles
    for i, q in enumerate(questions):
        expected_style = expected_styles[i % len(expected_styles)]
        if q.get("style") != expected_style:
            return False, f"Style mismatch for chunk {i}: expected {expected_style}, got {q.get('style')}"
    
    return True, ""


def check_tag_overlap(questions: List[Dict]) -> tuple[List[Dict], int]:
    """
    Check for word overlap between questions of the same request.
    Drop questions that clearly overlap more with another question in the batch.
    Returns (filtered_questions, dropped_count).
    """
    filtered = []
    dropped = 0
    
    for i, q in enumerate(questions):
        q_words = set(q.get("question", "").lower().split())
        if len(q_words) < 3:
            continue
        
        # Check overlap with other questions
        max_overlap = 0
        for j, other in enumerate(questions):
            if i == j:
                continue
            other_words = set(other.get("question", "").lower().split())
            if q_words and other_words:
                overlap = len(q_words & other_words) / len(q_words | other_words)
                max_overlap = max(max_overlap, overlap)
        
        # Drop if overlap > 0.7 with another question
        if max_overlap > 0.7:
            dropped += 1
        else:
            filtered.append(q)
    
    return filtered, dropped


def filter_questions_quality(questions: List[Dict]) -> tuple[List[Dict], int]:
    """
    Apply automatic quality filters to questions.
    Returns (filtered_questions, rejected_count).
    Filters:
    - >50% of question's content words appear in chunk
    - Shorter than 6 words
    - Near-duplicate (word-set cosine > 0.8)
    """
    filtered = []
    rejected = 0
    
    for q in questions:
        question_text = q.get("question", "").lower()
        words = set(question_text.split())
        
        # Filter: shorter than 6 words
        if len(words) < 6:
            rejected += 1
            continue
        
        # Filter: >50% of content words appear in chunk (if chunk text available)
        if "source_snippet" in q:
            chunk_words = set(q["source_snippet"].lower().split())
            if words and chunk_words:
                overlap = len(words & chunk_words) / len(words)
                if overlap > 0.5:
                    rejected += 1
                    continue
        
        # Filter: near-duplicate (simple check for identical questions)
        existing_texts = [fq.get("question", "").lower() for fq in filtered]
        for existing in existing_texts:
            existing_words = set(existing.split())
            if words and existing_words:
                # Simple word-set overlap as proxy for cosine
                overlap = len(words & existing_words) / len(words | existing_words)
                if overlap > 0.8:
                    rejected += 1
                    break
        else:
            filtered.append(q)
    
    return filtered, rejected


def log_chat_request(data_dir: str, model: str, paper_ids: List[str], outcome: str, timing: Dict, error: str = None, raw_reply: str = None):
    """Log a chat request to chat_requests.jsonl."""
    log_path = Path(data_dir) / "chat_requests.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "paper_ids": paper_ids,
        "outcome": outcome,  # "success", "rate_limited", "error", "parse_failed"
        "llm_latency": timing.get("llm", 0),
        "http_status": timing.get("http_status"),
        "finish_reason": timing.get("finish_reason"),
        "headers": timing.get("headers", {}),
        "error": error,
        "raw_reply": raw_reply if outcome == "parse_failed" else None  # Only save raw on parse failure
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def generate_questions_batch(
    chunks: List[Dict],
    model_id: str,
    api_key: str,
    base_url: str,
    seed: int = 42,
    dry_run: bool = False
) -> Tuple[List[Dict], Dict, str]:
    """
    Generate questions for chunks in one batched request.
    Returns (questions, timing_info).
    """
    random.seed(seed)
    
    # Build prompt with chunks
    # Styles per chunk order: factual, factual, conceptual, conceptual, paraphrased, paraphrased
    styles = ["factual", "factual", "conceptual", "conceptual", "paraphrased", "paraphrased"]
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        style = styles[i % len(styles)]
        chunks_text += f"\nChunk {i+1} (ID: {chunk['chunk_id']}, Paper: {chunk['paper_id']}, Page {chunk['page_start']}):\n"
        chunks_text += f"Style: {style}\n"
        chunks_text += chunk["text"][:1200] + "\n"  # Limit to 1200 chars
    
    prompt = f"""Generate exactly 1 question for each of these {len(chunks)} research paper chunks. Return a JSON array with exactly {len(chunks)} items, each with chunk_id, style, and question.

{chunks_text}

For each chunk:
- chunk_id: use the exact chunk_id from the chunk
- style: use the specified style for that chunk
- question: a self-contained question answerable from that chunk (no "this paper", "the authors", no figure/section numbers, no yes/no questions)

Output format:
[
  {{"chunk_id": "chunk_id", "style": "style", "question": "..."}},
  ...
]

Replace "chunk_id" and "style" with actual values from the chunks above.
Questions must be answerable from the chunk text.
Include paper-specific terminology.
Output valid JSON only, no markdown, no explanations."""

    if dry_run:
        # In dry-run, just show prompt size and return empty
        print(f"  Prompt size: {len(prompt)} characters")
        print(f"  Estimated tokens: ~{len(prompt.split())} words")
        print(f"  First 200 chars of prompt: {prompt[:200]}...")
        return [], {"llm": 0, "http_status": 0, "finish_reason": "dry_run"}, ""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/omartt4/AI-project-group",
        "X-Title": "Graph RAG Lab"
    }
    
    start = time.time()
    
    # Different system prompt for reasoning models to avoid thinking output
    if "nemotron" in model_id.lower():
        system_prompt = "You are a JSON generator. Output ONLY the JSON array. Do NOT include any thinking process, analysis, or explanations. Start your response immediately with '['."
    else:
        system_prompt = "Generate JSON array only. No explanations."
    
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0
    }
    
    # Try to disable reasoning for Nemotron
    if "nemotron" in model_id.lower():
        data["reasoning"] = {"max_tokens": 0}  # Disable reasoning output
    else:
        data["response_format"] = {"type": "json_object"}  # Force JSON for non-reasoning models
    
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )
    
    timing = {
        "llm": time.time() - start,
        "http_status": response.status_code,
        "finish_reason": None,
        "headers": dict(response.headers)  # Capture headers for rate limit info
    }
    
    # Handle 429 before raise_for_status
    if response.status_code == 429:
        print_error(f"HTTP 429 Rate Limited!")
        print_error(f"Response body: {response.text}")
        print_error(f"Headers: {dict(response.headers)}")
        print_error(f"Full response details:")
        try:
            error_json = response.json()
            print_error(f"  JSON: {json.dumps(error_json, indent=2)}")
        except:
            pass
        return [], timing, response.text
    
    response.raise_for_status()
    result = response.json()
    timing["finish_reason"] = result["choices"][0].get("finish_reason", "unknown")
    
    # Extract answer
    message = result["choices"][0]["message"]
    answer = message.get("content", "")
    
    # If content is empty, try reasoning field
    if not answer:
        answer = message.get("reasoning", "")
        if not answer:
            answer = message.get("reasoning_content", "")
    
    # If still empty, print error
    if not answer:
        print_error(f"LLM returned empty answer. Finish reason: {timing['finish_reason']}")
        print(f"Message keys: {list(message.keys())}")
        return [], timing, ""
    
    # If answer contains "thinking process", try to extract final JSON
    lines = answer.split('\n')
    final_lines = []
    in_thinking = False
    for line in lines:
        line_lower = line.lower()
        # Detect start of thinking
        if 'thinking process' in line_lower or 'analysis' in line_lower or 'here\'s a thinking' in line_lower or 'let me think' in line_lower:
            in_thinking = True
        # Detect end of thinking (start of JSON)
        elif in_thinking and ('```json' in line or line.strip().startswith('[') or line.strip().startswith('{')):
            in_thinking = False
            final_lines.append(line)
        elif not in_thinking:
            final_lines.append(line)
    
    if final_lines and len(final_lines) < len(lines):
        answer = '\n'.join(final_lines).strip()
        # If still starts with thinking, take last paragraph
        if answer.lower().startswith(('here\'s a thinking', 'let me', 'i need to', '1.', '2.', '3.', '4.', '5.', '6.')):
            paragraphs = answer.split('\n\n')
            # Find first paragraph that looks like JSON
            for para in paragraphs:
                if para.strip().startswith('[') or para.strip().startswith('{'):
                    answer = para.strip()
                    break
            else:
                answer = paragraphs[-1].strip()
    
    # Aggressive fallback: find first [ or { in entire response
    if not answer.strip().startswith('[') and not answer.strip().startswith('{'):
        bracket_pos = answer.find('[')
        brace_pos = answer.find('{')
        if bracket_pos >= 0 and (brace_pos < 0 or bracket_pos < brace_pos):
            answer = answer[bracket_pos:]
        elif brace_pos >= 0:
            answer = answer[brace_pos:]
    
    # Strip and parse JSON
    answer = answer.strip()
    if answer.startswith("```"):
        answer = re.sub(r"^```(?:json)?\n", "", answer)
        answer = re.sub(r"\n```$", "", answer)
    
    try:
        result_obj = json.loads(answer)
        # Handle both array and object with "questions" key
        if isinstance(result_obj, dict) and "questions" in result_obj:
            questions = result_obj["questions"]
        elif isinstance(result_obj, list):
            questions = result_obj
        else:
            print_error(f"Unexpected JSON format: {type(result_obj)}")
            return [], timing, answer
    except json.JSONDecodeError as e:
        print_error(f"Failed to parse JSON: {e}")
        print(f"Raw answer: {answer[:500]}")
        
        # Try to find the last ] or } to truncate trailing text
        last_bracket = max(answer.rfind(']'), answer.rfind('}'))
        if last_bracket > 0:
            truncated = answer[:last_bracket + 1]
            try:
                result_obj = json.loads(truncated)
                if isinstance(result_obj, dict) and "questions" in result_obj:
                    questions = result_obj["questions"]
                elif isinstance(result_obj, list):
                    questions = result_obj
                else:
                    return [], timing, answer
                print("Successfully parsed after truncating trailing text")
            except json.JSONDecodeError:
                return [], timing, answer
        else:
            return [], timing, answer
    
    return questions, timing, answer


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Question Generation")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSONL file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing questions file")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts without making API calls")
    parser.add_argument("--force", action="store_true", help="Bypass local quota check (use with caution)")
    parser.add_argument("--max-batches", type=int, default=None, help="Maximum number of batches to process")
    parser.add_argument("--topup", action="store_true", help="Top up papers with <5 questions")
    args = parser.parse_args()
    
    config = get_config()
    
    # Check rate limit (skip in dry-run or with --force)
    if not args.dry_run and not args.force:
        print_section("Checking Rate Limits")
        from lab_models import check_chat_quota
        quota_info = check_chat_quota(config.openrouter_base_url, config.openrouter_api_key, config.data_dir)
        print(f"  Requests in last 24h: {quota_info['requests_24h']}")
        if quota_info['requests_available'] is not None:
            print(f"  Requests available: {quota_info['requests_available']}")
        if quota_info['reset_time_utc']:
            print(f"  Reset time (UTC): {quota_info['reset_time_utc']}")
            print(f"  Reset time (Cairo, UTC+2): {quota_info['reset_time_cairo']}")
        
        remaining = quota_info['requests_available'] or 0
        if remaining < 3:
            print_error(f"Insufficient requests remaining: {remaining} < 3. Stop and wait for reset.")
            print_error("Use --force to bypass this check (not recommended).")
            sys.exit(1)
    elif args.force:
        print_warning("Bypassing local quota check (--force flag used)")
        print_section("Fetching Key Info (for reference)")
        from lab_models import get_openrouter_key_info
        try:
            key_info = get_openrouter_key_info(config.openrouter_base_url, config.openrouter_api_key)
            # Redact API key for display
            import copy
            key_info_redacted = copy.deepcopy(key_info)
            if "data" in key_info_redacted and "label" in key_info_redacted["data"]:
                key_info_redacted["data"]["label"] = "REDACTED"
            print(f"  Raw key info structure:")
            print(f"    Top-level keys: {list(key_info_redacted.keys())}")
            if "data" in key_info_redacted:
                print(f"    data keys: {list(key_info_redacted['data'].keys())}")
                if "free_model_daily_requests" in key_info_redacted["data"]:
                    print(f"    free_model_daily_requests keys: {list(key_info_redacted['data']['free_model_daily_requests'].keys())}")
                    print(f"    free_model_daily_requests values: {key_info_redacted['data']['free_model_daily_requests']}")
            # Fix quota parser based on real structure
            print(f"  Field names used for quota: key_info['data']['free_model_daily_requests']['remaining']")
            # Also fix the quota parser in lab_models.py
            remaining = 999  # Dummy value
        except Exception as e:
            print(f"  Warning: Could not fetch key info: {e}")
            remaining = 999  # Dummy value
    else:
        print_section("Dry Run Mode (no API calls)")
        remaining = 999  # Dummy value for dry-run
    
    # Verify models
    print_section("Verifying OpenRouter Models")
    verify_all_models()
    
    # Load chunks
    chunks_path = Path(args.chunks)
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    
    # Group by paper
    paper_chunks = {}
    for chunk in chunks:
        paper_id = chunk["paper_id"]
        if paper_id not in paper_chunks:
            paper_chunks[paper_id] = []
        paper_chunks[paper_id].append(chunk)
    
    print(f"Found {len(paper_chunks)} papers")
    
    # Load existing questions if resuming
    questions_file = Path(config.data_dir) / "questions.json"
    existing_questions = []
    if args.resume and questions_file.exists():
        with open(questions_file, "r", encoding="utf-8") as f:
            existing_questions = json.load(f)
        print(f"Loaded {len(existing_questions)} existing questions")
    
    # Track which papers are done
    done_papers = {q["paper_id"] for q in existing_questions}
    
    # Group papers into batches of 1 (one request per paper)
    paper_ids_sorted = sorted(paper_chunks.keys())
    paper_batches = [[pid] for pid in paper_ids_sorted]  # One paper per batch
    
    print(f"Grouped {len(paper_ids_sorted)} papers into {len(paper_batches)} batches (1 paper per request)")
    
    # Expected styles for validation
    expected_styles = ["factual", "factual", "conceptual", "conceptual", "paraphrased", "paraphrased"]
    
    # Fallback models
    fallback_models = [m.strip() for m in config.qgen_fallbacks.split(",")]
    models_to_try = [config.qgen_model] + fallback_models
    
    # Track papers 001 and 002 for initial test
    test_papers = ["001_NLP_End-to-End_Speaker_Diarization_as_Post-Processing", "002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction"]
    test_complete = False
    
    # Generate questions per batch
    print_section("Generating Questions")
    
    all_questions = existing_questions.copy()
    rejected_count = 0
    regenerated_count = 0
    dropped_tag_count = 0
    requests_used = 0
    max_rounds = 2  # Max rounds (one attempt per model per round)
    
    for round_idx in range(max_rounds):
        if round_idx > 0:
            print(f"\n=== Round {round_idx + 1} ===")
            time.sleep(60)  # 60s pause between rounds
        
        for batch_idx, paper_batch in enumerate(paper_batches):
            # Skip if all papers in batch are done
            if all(p in done_papers for p in paper_batch):
                print(f"Skipping {paper_batch[0]} (already done)")
                continue
            
            # Filter to only papers not done
            papers_to_process = [p for p in paper_batch if p not in done_papers]
            if not papers_to_process:
                continue
            
            paper_id = papers_to_process[0]  # Only 1 paper per batch
            
            # Only stop after test papers for initial review if not yet complete
            if not test_complete and paper_id not in test_papers:
                print_section(f"Test papers complete ({test_papers}) - Stopping for review")
                break
            
            print(f"\n=== Paper {paper_id} (Round {round_idx + 1}) ===")
            
            # Sample 6 chunks
            p_chunks = paper_chunks[paper_id]
            sampled = sample_chunks_per_paper(p_chunks, n=6)
            
            print(f"  Chunks: {len(sampled)}")
            
            # Try each model once this round
            for model_idx, model_id in enumerate(models_to_try):
                print(f"  Trying model: {model_id}")
                
                # Generate batch (one request per paper)
                questions, timing, raw_reply = generate_questions_batch(
                    sampled,
                    model_id,
                    config.openrouter_api_key,
                    config.openrouter_base_url,
                    seed=42 + round_idx * 10,  # Different seed per round
                    dry_run=args.dry_run
                )
                
                requests_used += 1
                
                # Handle 429 - log provider info and try next model
                if timing.get("http_status") == 429:
                    print_error(f"HTTP 429 on {model_id}")
                    print_error(f"Response body: {raw_reply}")
                    print_error(f"Headers: {dict(timing.get('headers', {}))}")
                    
                    # Extract provider info
                    provider_name = "unknown"
                    limit_source = "unknown"
                    try:
                        error_json = json.loads(raw_reply)
                        provider_name = error_json.get("error", {}).get("metadata", {}).get("provider_name", "unknown")
                        limit_source = error_json.get("error", {}).get("metadata", {}).get("limit_source", "unknown")
                    except:
                        pass
                    
                    log_chat_request(config.data_dir, model_id, [paper_id], "rate_limited", timing, 
                                   f"Provider: {provider_name}, Limit source: {limit_source}")
                    
                    print_error(f"  Provider: {provider_name}, Limit source: {limit_source}")
                    
                    # Try next model this round
                    if model_idx < len(models_to_try) - 1:
                        print(f"  Trying next model...")
                        continue
                    else:
                        print_error(f"All models exhausted for {paper_id} this round")
                        break
                
                # Handle other HTTP errors
                elif timing.get("http_status") != 200:
                    log_chat_request(config.data_dir, model_id, [paper_id], "error", timing, raw_reply)
                    print_error(f"HTTP {timing.get('http_status')}: {raw_reply}")
                    if model_idx < len(models_to_try) - 1:
                        print(f"  Trying next model...")
                        continue
                    else:
                        break
                
                # Handle parse failure
                elif not questions:
                    log_chat_request(config.data_dir, model_id, [paper_id], "parse_failed", timing, raw_reply=raw_reply)
                    print_error(f"Parse failed, raw reply: {raw_reply[:500]}")
                    if model_idx < len(models_to_try) - 1:
                        print(f"  Trying next model...")
                        continue
                    else:
                        break
                
                # Success - validate and process
                log_chat_request(config.data_dir, model_id, [paper_id], "success", timing)
                print(f"  Success with {model_id}")
                
                # Print raw reply for paper 001
                if paper_id == test_papers[0] and round_idx == 0:
                    print_section(f"{paper_id} Raw Reply")
                    print(raw_reply[:1000] if len(raw_reply) > 1000 else raw_reply)
                
                # Validate chunk_id and styles
                is_valid, validation_error = validate_questions(questions, sampled, expected_styles)
                if not is_valid:
                    print_error(f"Validation failed: {validation_error}")
                    print(f"Raw reply: {raw_reply[:500]}")
                    # Try next model
                    if model_idx < len(models_to_try) - 1:
                        print(f"  Trying next model due to validation failure...")
                        continue
                    else:
                        log_chat_request(config.data_dir, model_id, [paper_id], "validation_failed", timing, validation_error)
                        break
                
                print(f"  Validation: PASS (count={len(questions)}, ids match, styles match)")
                
                # Tag paper_id from chunk_id (lookup in chunks)
                chunk_id_to_paper = {c["chunk_id"]: c["paper_id"] for c in sampled}
                for q in questions:
                    chunk_id = q.get("chunk_id")
                    if chunk_id in chunk_id_to_paper:
                        q["paper_id"] = chunk_id_to_paper[chunk_id]
                    else:
                        q["paper_id"] = paper_id  # Fallback
                    # Add source snippet
                    source_chunk = next((c for c in sampled if c["chunk_id"] == chunk_id), None)
                    if source_chunk:
                        q["source_snippet"] = source_chunk["text"][:200]
                
                # Check tag overlap
                questions, dropped = check_tag_overlap(questions)
                dropped_tag_count += dropped
                print(f"  Tag overlap check: {dropped} questions dropped")
                
                # Apply quality filter
                questions, rejected_batch = filter_questions_quality(questions)
                rejected_count += rejected_batch
                print(f"  Quality filter: {rejected_batch} questions rejected")
                
                all_questions.extend(questions)
                done_papers.add(paper_id)
                
                # Save incrementally
                with open(questions_file, "w", encoding="utf-8") as f:
                    json.dump(all_questions, f, indent=2)
                
                print(f"  Generated {len(questions)} questions (final) in {timing['llm']:.2f}s")
                print(f"  Provider/Model: {model_id}")
                print(f"  Requests used: {requests_used}")
                print(f"  Saved to {questions_file}")
                
                # Break out of model loop on success
                break
        
        # Check if test papers are complete
        if test_complete:
            break
        # After completing both test papers, mark as complete and continue with remaining papers
        if all(p in done_papers for p in test_papers):
            test_complete = True
            print_section("Test Papers Complete - Review Results")
            print(f"Paper 001: Generated, validated, filtered")
            print(f"Paper 002: Generated, validated, filtered")
            print(f"Total so far: {len(all_questions)} questions")
            print(f"Rejected: {rejected_count}, Dropped by tag check: {dropped_tag_count}")
            print("Test papers passed. Continuing with remaining 7 papers...")
            # Don't break - continue with remaining papers
    
    # Top-up papers with <5 questions
    if args.topup:
        print_section("Top-up Mode")
        
        # Reload questions to get latest state
        with open(questions_file, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
        
        # Count questions per paper
        from collections import Counter
        by_paper = Counter([q['paper_id'] for q in all_questions])
        
        # Find papers with <5 questions
        papers_to_topup = [p for p, c in by_paper.items() if c < 5]
        papers_to_topup.sort()
        
        print(f"Papers needing top-up: {len(papers_to_topup)}")
        for p in papers_to_topup:
            print(f"  {p}: {by_paper[p]} questions")
        
        # Cap at 5 papers
        papers_to_topup = papers_to_topup[:5]
        
        if papers_to_topup:
            # Check quota
            from lab_models import check_chat_quota
            quota_info = check_chat_quota(config.openrouter_base_url, config.openrouter_api_key, config.data_dir)
            print(f"  Quota remaining: {quota_info['requests_available']}")
            
            if quota_info['requests_available'] is not None and quota_info['requests_available'] < len(papers_to_topup):
                print_error(f"Insufficient quota for top-up: {quota_info['requests_available']} < {len(papers_to_topup)}")
            else:
                # Top-up each paper
                for paper_id in papers_to_topup:
                    print(f"\n=== Top-up: {paper_id} ===")
                    
                    # Get existing questions for this paper
                    paper_questions = [q for q in all_questions if q['paper_id'] == paper_id]
                    existing_chunk_ids = {q['chunk_id'] for q in paper_questions}
                    existing_styles_by_chunk = {q['chunk_id']: q['style'] for q in paper_questions}
                    
                    # Sample 6 chunks
                    p_chunks = paper_chunks[paper_id]
                    sampled = sample_chunks_per_paper(p_chunks, n=6)
                    
                    # Find chunks that need questions (missing styles or no question)
                    chunks_to_process = []
                    for chunk in sampled:
                        if chunk['chunk_id'] not in existing_chunk_ids:
                            # No question for this chunk
                            chunks_to_process.append(chunk)
                        else:
                            # Has question, check if missing other styles
                            existing_style = existing_styles_by_chunk[chunk['chunk_id']]
                            # Expected styles: factual, factual, conceptual, conceptual, paraphrased, paraphrased
                            chunk_idx = sampled.index(chunk)
                            expected_style = expected_styles[chunk_idx % len(expected_styles)]
                            if existing_style != expected_style:
                                # Wrong style, need to regenerate
                                chunks_to_process.append(chunk)
                    
                    if not chunks_to_process:
                        print(f"  No chunks need top-up")
                        continue
                    
                    print(f"  Chunks to process: {len(chunks_to_process)}")
                    
                    # Generate for missing chunks
                    questions, timing, raw_reply = generate_questions_batch(
                        chunks_to_process,
                        "nvidia/nemotron-3.5-lightning:free",  # Use NVIDIA for top-up
                        config.openrouter_api_key,
                        config.openrouter_base_url,
                        seed=99,
                        dry_run=args.dry_run
                    )
                    
                    requests_used += 1
                    
                    if timing.get("http_status") == 200 and questions:
                        log_chat_request(config.data_dir, "nvidia/nemotron-3.5-lightning:free", [paper_id], "success", timing)
                        print(f"  Success: {len(questions)} questions")
                        
                        # Validate and tag
                        is_valid, validation_error = validate_questions(questions, chunks_to_process, expected_styles)
                        if not is_valid:
                            print_error(f"Validation failed: {validation_error}")
                            continue
                        
                        # Tag paper_id
                        chunk_id_to_paper = {c["chunk_id"]: c["paper_id"] for c in chunks_to_process}
                        for q in questions:
                            chunk_id = q.get("chunk_id")
                            if chunk_id in chunk_id_to_paper:
                                q["paper_id"] = chunk_id_to_paper[chunk_id]
                            else:
                                q["paper_id"] = paper_id
                            source_chunk = next((c for c in chunks_to_process if c["chunk_id"] == chunk_id), None)
                            if source_chunk:
                                q["source_snippet"] = source_chunk["text"][:200]
                        
                        # Tag overlap and quality filter
                        questions, dropped = check_tag_overlap(questions)
                        dropped_tag_count += dropped
                        questions, rejected_batch = filter_questions_quality(questions)
                        rejected_count += rejected_batch
                        
                        # Remove old questions for these chunks and add new ones
                        all_questions = [q for q in all_questions if q['paper_id'] != paper_id or q['chunk_id'] not in [c['chunk_id'] for c in chunks_to_process]]
                        all_questions.extend(questions)
                        
                        # Save
                        with open(questions_file, "w", encoding="utf-8") as f:
                            json.dump(all_questions, f, indent=2)
                        
                        print(f"  Saved: {len(questions)} questions")
                    else:
                        log_chat_request(config.data_dir, "nvidia/nemotron-3.5-lightning:free", [paper_id], "error", timing, raw_reply)
                        print_error(f"Top-up failed for {paper_id}")
    
    # Show sample
    print_section("Sample Questions (15 random)")
    sample = random.sample(all_questions, min(15, len(all_questions)))
    for q in sample:
        print(f"\nPaper: {q['paper_id']}")
        print(f"Style: {q['style']}")
        print(f"Chunk ID: {q.get('chunk_id', 'N/A')}")
        print(f"Source snippet: {q.get('source_snippet', 'N/A')[:100]}...")
        print(f"Question: {q['question']}")
    
    print(f"\nTotal questions: {len(all_questions)}")
    print(f"Rejected: {rejected_count}")
    print(f"Regenerated: {regenerated_count}")
    print(f"Requests remaining: {remaining}")
    
    print_section("Done")


if __name__ == "__main__":
    main()
