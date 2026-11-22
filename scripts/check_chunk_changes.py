#!/usr/bin/env python3
"""Check difflib ratio for changed chunks in paper 002."""
import json
from pathlib import Path
import difflib

def load_chunks(chunks_file):
    chunks = {}
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunks[chunk['chunk_id']] = chunk
    return chunks

def compare_chunks(old_chunks, new_chunks, chunk_ids):
    for chunk_id in chunk_ids:
        if chunk_id in old_chunks and chunk_id in new_chunks:
            old_text = old_chunks[chunk_id]['text']
            new_text = new_chunks[chunk_id]['text']
            
            ratio = difflib.SequenceMatcher(None, old_text, new_text).ratio()
            
            print(f"\nChunk: {chunk_id}")
            print(f"Difflib ratio: {ratio:.4f}")
            print(f"Old length: {len(old_text)}, New length: {len(new_text)}")
            print(f"Old char_start: {old_chunks[chunk_id]['char_start']}, New char_start: {new_chunks[chunk_id]['char_start']}")
            print(f"Old char_end: {old_chunks[chunk_id]['char_end']}, New char_end: {new_chunks[chunk_id]['char_end']}")
            
            # Show first difference
            if old_text != new_text:
                for i, (c1, c2) in enumerate(zip(old_text, new_text)):
                    if c1 != c2:
                        print(f"First difference at position {i}:")
                        print(f"  Old: {repr(old_text[max(0,i-10):i+10])}")
                        print(f"  New: {repr(new_text[max(0,i-10):i+10])}")
                        break

if __name__ == "__main__":
    # Load backup (old) and current (new) chunks
    old_chunks = load_chunks('data/lab/chunks_256_26.jsonl.backup')
    new_chunks = load_chunks('data/lab/chunks_256_26.jsonl')
    
    # Check the changed chunks in paper 002 (from verification output)
    changed_chunks = [
        '002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction_15',
        '002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction_16'
    ]
    
    compare_chunks(old_chunks, new_chunks, changed_chunks)
