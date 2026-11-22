#!/usr/bin/env python3
"""Verify char_start/char_end offsets match cleaned text slices exactly."""
import json
from pathlib import Path

def verify_offsets(chunks_file, cleaned_dir):
    """Verify chunk text matches cleaned text slice exactly."""
    chunks_path = Path(chunks_file)
    cleaned_path_base = Path(cleaned_dir)
    
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    passed = 0
    failed = []
    
    for chunk in chunks:
        paper_id = chunk['paper_id']
        cleaned_file = cleaned_path_base / f"{paper_id}.txt"
        
        if not cleaned_file.exists():
            failed.append((chunk['chunk_id'], 'cleaned file missing'))
            continue
        
        cleaned_text = cleaned_file.read_text(encoding='utf-8')
        char_start = chunk['char_start']
        char_end = chunk['char_end']
        chunk_text = chunk['text']
        
        slice_text = cleaned_text[char_start:char_end]
        
        if slice_text == chunk_text:
            passed += 1
        else:
            failed.append((chunk['chunk_id'], f'slice mismatch (len(slice)={len(slice_text)}, len(chunk)={len(chunk_text)})'))
    
    return passed, len(chunks), failed

if __name__ == "__main__":
    print("Verifying 256 chunks...")
    passed_256, total_256, failed_256 = verify_offsets('data/lab/chunks_256_26.jsonl', 'data/lab/cleaned')
    print(f"256: {passed_256}/{total_256} passed (exact slice equality)")
    if failed_256:
        print(f"Failed: {len(failed_256)}")
        for chunk_id, reason in failed_256[:10]:
            print(f"  {chunk_id}: {reason}")
    
    print("\nVerifying 384 chunks...")
    passed_384, total_384, failed_384 = verify_offsets('data/lab/chunks_384_38.jsonl', 'data/lab/cleaned')
    print(f"384: {passed_384}/{total_384} passed (exact slice equality)")
    if failed_384:
        print(f"Failed: {len(failed_384)}")
        for chunk_id, reason in failed_384[:10]:
            print(f"  {chunk_id}: {reason}")
