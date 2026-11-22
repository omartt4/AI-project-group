#!/usr/bin/env python3
"""Compare Chroma stored documents with current chunk files."""
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings

def load_chunks(chunks_file):
    chunks = {}
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunks[chunk['chunk_id']] = chunk
    return chunks

def check_collection(collection_name, chunks_file):
    """Check if Chroma collection matches chunk file."""
    chunks = load_chunks(chunks_file)
    
    config = {
        'chroma_dir': 'data/chroma_lab'
    }
    client = chromadb.PersistentClient(path=config['chroma_dir'], settings=Settings(anonymized_telemetry=False))
    
    # List all collections first
    collections = client.list_collections()
    collection_names = [col.name for col in collections]
    
    print(f"\nCollection: {collection_name}")
    print(f"  Available collections: {collection_names}")
    
    if collection_name not in collection_names:
        print(f"  ERROR: Collection does not exist")
        return False
    
    collection = client.get_collection(name=collection_name)
    
    # Get all documents from Chroma
    chroma_data = collection.get(include=['documents', 'metadatas'])
    
    chroma_ids = set(chroma_data['ids'])
    chunk_ids = set(chunks.keys())
    
    print(f"  Chroma documents: {len(chroma_ids)}")
    print(f"  Chunk file entries: {len(chunk_ids)}")
    
    # Check ID mismatch
    missing_in_chroma = chunk_ids - chroma_ids
    extra_in_chroma = chroma_ids - chunk_ids
    
    if missing_in_chroma:
        print(f"  Missing in Chroma: {len(missing_in_chroma)}")
        if len(missing_in_chroma) <= 5:
            print(f"    {missing_in_chroma}")
    if extra_in_chroma:
        print(f"  Extra in Chroma: {len(extra_in_chroma)}")
        if len(extra_in_chroma) <= 5:
            print(f"    {extra_in_chroma}")
    
    # Check text mismatch for common IDs
    common_ids = chroma_ids & chunk_ids
    text_mismatches = []
    
    for chunk_id in common_ids:
        chroma_idx = chroma_data['ids'].index(chunk_id)
        chroma_text = chroma_data['documents'][chroma_idx]
        chunk_text = chunks[chunk_id]['text']
        
        if chroma_text != chunk_text:
            text_mismatches.append(chunk_id)
    
    if text_mismatches:
        print(f"  Text mismatches: {len(text_mismatches)}")
        if len(text_mismatches) <= 5:
            for chunk_id in text_mismatches:
                print(f"    {chunk_id}")
    else:
        print(f"  Text matches: {len(common_ids)}/{len(common_ids)}")
    
    return len(missing_in_chroma) == 0 and len(extra_in_chroma) == 0 and len(text_mismatches) == 0

if __name__ == "__main__":
    ok_256 = check_collection("retrieval_lab_BAAI_bge-small-en-v1.5_384_256_26_vector", "data/lab/chunks_256_26.jsonl")
    ok_384 = check_collection("retrieval_lab_BAAI_bge-small-en-v1.5_384_384_38_vector", "data/lab/chunks_384_38.jsonl")
    
    print("\n" + "="*60)
    print(f"256 collection matches: {ok_256}")
    print(f"384 collection matches: {ok_384}")
