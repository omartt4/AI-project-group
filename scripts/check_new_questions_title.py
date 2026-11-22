#!/usr/bin/env python3
"""Check the 9 new questions for title terms."""
import json
import re

def get_paper_id_from_chunk_id(chunk_id):
    """Extract paper ID from chunk ID."""
    # chunk_id format: 001_NLP_..._2
    parts = chunk_id.split('_')
    # Paper ID is everything before the last underscore and number
    # e.g., 001_NLP_End-to-End_Speaker_Diarization_as_Post-Processing_2
    # paper_id = 001_NLP_End-to-End_Speaker_Diarization_as_Post-Processing
    if parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return chunk_id

def extract_title_terms(paper_id):
    """Extract title terms from paper ID."""
    # Remove the numeric prefix
    title_part = paper_id.split('_', 1)[1] if '_' in paper_id else paper_id
    # Split by underscore and lowercase
    terms = [t.lower() for t in title_part.split('_')]
    return terms

def check_title_in_question(question, title_terms):
    """Check if question contains paper ID or 3+ consecutive title words."""
    question_text = question['question'].lower()
    paper_id = question['paper_id'].lower()
    
    # Check paper ID
    if paper_id in question_text:
        return f"Contains paper ID: {paper_id}"
    
    # Check 3+ consecutive title words
    for i in range(len(title_terms) - 2):
        consecutive = ' '.join(title_terms[i:i+3])
        if consecutive in question_text:
            return f"Contains 3+ consecutive title words: {consecutive}"
    
    return None

# Load questions
with open('eval_lab/questions.jsonl', 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f]

# Get 074/090 questions
new_questions = [q for q in questions if q['paper_id'].startswith('074_') or q['paper_id'].startswith('090_')]

print(f"Found {len(new_questions)} questions from papers 074 and 090\n")

for q in new_questions:
    paper_id = q['paper_id']
    title_terms = extract_title_terms(paper_id)
    
    print(f"Chunk: {q['chunk_id']}")
    print(f"Paper ID: {paper_id}")
    print(f"Title terms: {title_terms}")
    print(f"Question: {q['question']}")
    
    # My check
    my_result = check_title_in_question(q, title_terms)
    if my_result:
        print(f"  My check: FAIL - {my_result}")
    else:
        print(f"  My check: PASS")
    
    # Test d check (from lab_eval.py)
    # Test d only checks if the numeric prefix (e.g., "074") is in the question
    numeric_prefix = paper_id.split('_')[0]
    if numeric_prefix in q['question'].lower():
        print(f"  Test d check: FAIL - contains {numeric_prefix}")
    else:
        print(f"  Test d check: PASS")
    
    print()
