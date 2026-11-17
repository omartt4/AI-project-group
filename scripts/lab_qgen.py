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


def generate_questions_batch(
    chunks: List[Dict],
    model_id: str,
    api_key: str,
    base_url: str,
    seed: int = 42
) -> Tuple[List[Dict], Dict]:
    """
    Generate questions for chunks in one batched request.
    Returns (questions, timing_info).
    """
    random.seed(seed)
    
    # Build prompt with chunks
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"\nChunk {i+1} (ID: {chunk['chunk_id']}, Paper: {chunk['paper_id']}, Page {chunk['page_start']}):\n"
        chunks_text += chunk["text"][:800] + "\n"  # Limit to 800 chars
    
    prompt = f"""Generate 6 questions from these research paper chunks. Return a JSON object with a "questions" array.

{chunks_text}

For each chunk, generate 1 question (factual or conceptual).
Return ONLY this JSON format, no other text:
{{"questions": [
  {{"chunk_id": "chunk_id", "style": "factual", "question": "..."}},
  {{"chunk_id": "chunk_id", "style": "factual", "question": "..."}},
  {{"chunk_id": "chunk_id", "style": "conceptual", "question": "..."}},
  {{"chunk_id": "chunk_id", "style": "conceptual", "question": "..."}},
  {{"chunk_id": "chunk_id", "style": "factual", "question": "..."}},
  {{"chunk_id": "chunk_id", "style": "conceptual", "question": "..."}}
]}}

Replace "chunk_id" with actual chunk IDs and "..." with actual questions.
Questions must be answerable from the chunk text.
No yes/no questions.
Include paper-specific terminology.
Output valid JSON only, no markdown, no explanations."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/omartt4/AI-project-group",
        "X-Title": "Graph RAG Lab"
    }
    
    start = time.time()
    
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a JSON generator. Output ONLY valid JSON arrays. No explanations, no thinking process, no markdown."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )
    
    timing = {
        "llm": time.time() - start,
        "http_status": response.status_code,
        "finish_reason": None
    }
    
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
        return [], timing
    
    # If answer contains "thinking process", try to extract final JSON
    lines = answer.split('\n')
    final_lines = []
    in_thinking = False
    for line in lines:
        line_lower = line.lower()
        if 'thinking process' in line_lower or 'analysis' in line_lower or 'here\'s a thinking' in line_lower:
            in_thinking = True
        elif in_thinking and ('```json' in line or line.startswith('[') or line.startswith('{')):
            in_thinking = False
            final_lines.append(line)
        elif not in_thinking:
            final_lines.append(line)
    
    if final_lines and len(final_lines) < len(lines):
        answer = '\n'.join(final_lines).strip()
        # If still starts with thinking, take last paragraph
        if answer.lower().startswith(('here\'s a thinking', 'let me', 'i need to')):
            paragraphs = answer.split('\n\n')
            if len(paragraphs) > 1:
                answer = paragraphs[-1].strip()
    
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
            return [], timing
    except json.JSONDecodeError as e:
        print_error(f"Failed to parse JSON: {e}")
        print(f"Raw answer: {answer[:500]}")
        return [], timing
    
    return questions, timing


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Question Generation")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSONL file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing questions file")
    args = parser.parse_args()
    
    config = get_config()
    
    # Check rate limit
    print_section("Checking Rate Limits")
    key_info = check_rate_limit(config.openrouter_api_key, config.openrouter_base_url)
    print(f"Free tier: {key_info['data']['is_free_tier']}")
    print(f"Daily requests: {key_info['data']['free_model_daily_requests']['used']}/{key_info['data']['free_model_daily_requests']['limit']} (remaining: {key_info['data']['free_model_daily_requests']['remaining']})")
    
    remaining = key_info['data']['free_model_daily_requests']['remaining']
    if remaining < 3:
        print_error(f"Insufficient requests remaining: {remaining} < 3. Stop and wait for reset.")
        sys.exit(1)
    
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
    
    # Generate questions per paper
    print_section("Generating Questions")
    
    all_questions = existing_questions.copy()
    rejected_count = 0
    regenerated_count = 0
    
    for paper_id in sorted(paper_chunks.keys()):
        if paper_id in done_papers:
            print(f"Skipping {paper_id} (already done)")
            continue
        
        if remaining < 3:
            print_error(f"Stopping: only {remaining} requests remaining")
            break
        
        p_chunks = paper_chunks[paper_id]
        # Sample 6 chunks
        sampled = sample_chunks_per_paper(p_chunks, n=6)
        
        print(f"\nGenerating questions for {paper_id} ({len(sampled)} chunks)")
        
        # Generate batch
        questions, timing = generate_questions_batch(
            sampled,
            config.chat_model,
            config.openrouter_api_key,
            config.openrouter_base_url,
            seed=42
        )
        
        remaining -= 1
        print(f"  Generated {len(questions)} questions in {timing['llm']:.2f}s")
        print(f"  Requests remaining: {remaining}")
        
        # Validate and tag questions
        for q in questions:
            q["paper_id"] = paper_id
            # Find source chunk
            chunk_id = q.get("chunk_id")
            source_chunk = next((c for c in sampled if c["chunk_id"] == chunk_id), None)
            if source_chunk:
                q["source_snippet"] = source_chunk["text"][:200]
        
        all_questions.extend(questions)
        
        # Save incrementally
        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, indent=2)
        
        print(f"  Saved to {questions_file}")
    
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
