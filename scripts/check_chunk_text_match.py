#!/usr/bin/env python3
"""Check that stored ChromaDB text matches chunk files."""
import json
import chromadb
from chromadb.config import Settings

def check_collection(collection_name, chunks_path):
    client = chromadb.PersistentClient(path="data/chroma_lab", settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(name=collection_name)
    
    # Load chunks
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]
    
    # Get all stored documents
    stored_docs = collection.get(include=['documents', 'metadatas'])
    
    # Check match
    matches = 0
    mismatches = 0
    for chunk in chunks:
        chunk_id = chunk['chunk_id']
        chunk_text = chunk['text']
        
        # Find in stored docs
        idx = stored_docs['ids'].index(chunk_id)
        stored_text = stored_docs['documents'][idx]
        
        if chunk_text == stored_text:
            matches += 1
        else:
            mismatches += 1
            print(f"Mismatch: {chunk_id}")
            print(f"  Chunk length: {len(chunk_text)}")
            print(f"  Stored length: {len(stored_text)}")
    
    print(f"\n{collection_name}:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Matches: {matches}")
    print(f"  Mismatches: {mismatches}")
    print(f"  Result: {'PASS' if mismatches == 0 else 'FAIL'}")

if __name__ == "__main__":
    check_collection('retrieval_lab_BAAI_bge-small-en-v1.5_384_256_26_vector', 'data/lab/chunks_256_26.jsonl')
    check_collection('retrieval_lab_BAAI_bge-small-en-v1.5_384_384_38_vector', 'data/lab/chunks_384_38.jsonl')
