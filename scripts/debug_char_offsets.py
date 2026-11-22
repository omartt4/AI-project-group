#!/usr/bin/env python3
"""Debug char offset mismatches."""
import json
from pathlib import Path

def debug_chunk(chunk_id):
    """Debug a specific chunk."""
    chunks_path = Path('data/lab/chunks_256_26.jsonl')
    cleaned_dir = Path('data/lab/cleaned')
    
    # Find the chunk
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            if chunk['chunk_id'] == chunk_id:
                break
    
    paper_id = chunk['paper_id']
    cleaned_file = cleaned_dir / f"{paper_id}.txt"
    cleaned_text = cleaned_file.read_text(encoding='utf-8')
    
    char_start = chunk['char_start']
    char_end = chunk['char_end']
    chunk_text = chunk['text']
    
    slice_text = cleaned_text[char_start:char_end]
    
    print(f"Chunk: {chunk_id}")
    print(f"char_start: {char_start}, char_end: {char_end}")
    print(f"len(slice): {len(slice_text)}, len(chunk): {len(chunk_text)}")
    print(f"Equal: {slice_text == chunk_text}")
    
    # Show first 100 chars of each
    print(f"\nSlice (first 100):")
    print(repr(slice_text[:100]))
    print(f"\nChunk (first 100):")
    print(repr(chunk_text[:100]))
    
    # Show last 100 chars of each
    print(f"\nSlice (last 100):")
    print(repr(slice_text[-100:]))
    print(f"\nChunk (last 100):")
    print(repr(chunk_text[-100:]))
    
    # Find first difference
    for i, (c1, c2) in enumerate(zip(slice_text, chunk_text)):
        if c1 != c2:
            print(f"\nFirst difference at position {i}:")
            print(f"  Slice: {repr(slice_text[max(0,i-10):i+10])}")
            print(f"  Chunk: {repr(chunk_text[max(0,i-10):i+10])}")
            break

if __name__ == "__main__":
    import sys
    chunk_id = sys.argv[1] if len(sys.argv) > 1 else '001_NLP_End-to-End_Speaker_Diarization_as_Post-Processing_2'
    debug_chunk(chunk_id)
