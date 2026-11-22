#!/usr/bin/env python3
"""Check if question text matches new chunk text for specific chunks."""
import json
from pathlib import Path

def load_questions(questions_file):
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

def load_chunks(chunks_file):
    chunks = {}
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunks[chunk['chunk_id']] = chunk
    return chunks

if __name__ == "__main__":
    questions = load_questions('eval_lab/questions.jsonl')
    chunks = load_chunks('data/lab/chunks_256_26.jsonl')
    
    # Check the specific chunks in paper 002
    specific_chunks = [
        '002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction_18',
        '002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction_24'
    ]
    
    for chunk_id in specific_chunks:
        # Find questions for this chunk
        chunk_questions = [q for q in questions if q['chunk_id'] == chunk_id]
        
        print(f"\nChunk: {chunk_id}")
        print(f"Number of questions: {len(chunk_questions)}")
        
        if chunk_id in chunks:
            chunk_text = chunks[chunk_id]['text']
            print(f"Chunk text (first 100 chars): {repr(chunk_text[:100])}")
            
            for q in chunk_questions:
                print(f"\n  Question: {q['question']}")
                # Check if question text is in chunk text
                if q['question'] in chunk_text:
                    print(f"  ✓ Question text found in chunk")
                else:
                    print(f"  ✗ Question text NOT found in chunk")
                # Check if source snippet is in chunk text
                if q['source_snippet'] in chunk_text:
                    print(f"  ✓ Source snippet found in chunk")
                else:
                    print(f"  ✗ Source snippet NOT found in chunk")
