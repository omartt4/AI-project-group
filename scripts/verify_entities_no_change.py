#!/usr/bin/env python3
"""Verify that lab_entities.py did not change chunk text, ids, or char spans."""
import json
from pathlib import Path

def compare_chunks(original_file, modified_file):
    """Compare original and modified chunk files."""
    original_path = Path(original_file)
    modified_path = Path(modified_file)
    
    original_chunks = []
    with open(original_path, 'r', encoding='utf-8') as f:
        for line in f:
            original_chunks.append(json.loads(line))
    
    modified_chunks = []
    with open(modified_path, 'r', encoding='utf-8') as f:
        for line in f:
            modified_chunks.append(json.loads(line))
    
    if len(original_chunks) != len(modified_chunks):
        print(f"ERROR: Chunk count differs: {len(original_chunks)} vs {len(modified_chunks)}")
        return False
    
    text_changed = []
    id_changed = []
    char_start_changed = []
    char_end_changed = []
    
    for orig, mod in zip(original_chunks, modified_chunks):
        if orig['text'] != mod['text']:
            text_changed.append(orig['chunk_id'])
        if orig['chunk_id'] != mod['chunk_id']:
            id_changed.append(orig['chunk_id'])
        if orig['char_start'] != mod['char_start']:
            char_start_changed.append(orig['chunk_id'])
        if orig['char_end'] != mod['char_end']:
            char_end_changed.append(orig['chunk_id'])
    
    print(f"Total chunks: {len(original_chunks)}")
    print(f"Text changed: {len(text_changed)}")
    print(f"ID changed: {len(id_changed)}")
    print(f"char_start changed: {len(char_start_changed)}")
    print(f"char_end changed: {len(char_end_changed)}")
    
    if text_changed:
        print(f"Text changed in: {text_changed[:5]}")
    if id_changed:
        print(f"ID changed in: {id_changed[:5]}")
    if char_start_changed:
        print(f"char_start changed in: {char_start_changed[:5]}")
    if char_end_changed:
        print(f"char_end changed in: {char_end_changed[:5]}")
    
    return (len(text_changed) == 0 and len(id_changed) == 0 and 
            len(char_start_changed) == 0 and len(char_end_changed) == 0)

if __name__ == "__main__":
    print("Verifying 256 chunks...")
    ok_256 = compare_chunks('data/lab/chunks_256_26.jsonl.backup', 'data/lab/chunks_256_26.jsonl')
    print(f"256 chunks unchanged: {ok_256}")
    
    print("\nVerifying 384 chunks...")
    ok_384 = compare_chunks('data/lab/chunks_384_38.jsonl.backup', 'data/lab/chunks_384_38.jsonl')
    print(f"384 chunks unchanged: {ok_384}")
    
    if ok_256 and ok_384:
        print("\n✓ All chunk text, IDs, and char spans unchanged")
    else:
        print("\n✗ Some chunk data changed")
