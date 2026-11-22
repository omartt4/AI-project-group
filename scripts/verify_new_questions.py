#!/usr/bin/env python3
"""Verify the 9 new questions from papers 074 and 090."""
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
            paper_id = json.loads(line)['paper_id']
            chunk = json.loads(line)
            chunks[chunk['chunk_id']] = chunk
    return chunks

if __name__ == "__main__":
    questions = load_questions('eval_lab/questions.jsonl')
    chunks = load_chunks('data/lab/chunks_256_26.jsonl')
    
    # Get questions from papers 074 and 090
    new_questions = [q for q in questions if q['paper_id'].startswith('074_') or q['paper_id'].startswith('090_')]
    
    print(f"Found {len(new_questions)} questions from papers 074 and 090")
    print()
    
    for q in new_questions:
        chunk_id = q['chunk_id']
        paper_id = q['paper_id']
        question_text = q['question']
        
        print(f"Chunk: {chunk_id}")
        print(f"Paper: {paper_id}")
        
        # 1. Check chunk_id exists in chunks_256_26
        if chunk_id in chunks:
            print(f"  ✓ chunk_id exists in chunks_256_26")
        else:
            print(f"  ✗ chunk_id NOT in chunks_256_26")
        
        # 2. Check no paper id/title in the question text
        paper_id_in_question = paper_id in question_text
        title_in_question = any(word in question_text.lower() for word in paper_id.replace('_', ' ').lower().split())
        if not paper_id_in_question and not title_in_question:
            print(f"  ✓ No paper ID/title in question text")
        else:
            print(f"  ✗ Paper ID/title found in question text")
        
        # 3. Check not a near-duplicate (compare with other questions)
        is_duplicate = False
        for other_q in questions:
            if other_q['chunk_id'] != chunk_id:
                # Simple similarity check
                words_q = set(question_text.lower().split())
                words_other = set(other_q['question'].lower().split())
                overlap = len(words_q & words_other) / len(words_q | words_other) if words_q | words_other else 0
                if overlap > 0.7:
                    is_duplicate = True
                    print(f"  ✗ Near-duplicate detected with: {other_q['chunk_id']}")
                    break
        if not is_duplicate:
            print(f"  ✓ Not a near-duplicate")
        
        # 4. Word-overlap filter (check if question words overlap with chunk text)
        chunk_text = chunks.get(chunk_id, {}).get('text', '')
        question_words = set(question_text.lower().split())
        chunk_words = set(chunk_text.lower().split())
        overlap = len(question_words & chunk_words) / len(question_words) if question_words else 0
        if overlap > 0.3:  # Reasonable threshold
            print(f"  ✓ Word-overlap filter passed (overlap: {overlap:.2f})")
        else:
            print(f"  ✗ Word-overlap filter failed (overlap: {overlap:.2f})")
        
        print()
