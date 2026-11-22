#!/usr/bin/env python3
"""Filter questions according to the rules."""
import json
import re

def normalize_text(text):
    """Normalize whitespace and newlines."""
    return re.sub(r'\s+', ' ', text).strip()

def get_paper_id_from_chunk_id(chunk_id):
    """Extract paper ID from chunk ID."""
    parts = chunk_id.split('_')
    if parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return chunk_id

def extract_title_terms(paper_id):
    """Extract title terms from paper ID."""
    title_part = paper_id.split('_', 1)[1] if '_' in paper_id else paper_id
    terms = [t.lower() for t in title_part.split('_')]
    return terms

def check_title_in_question(question, title_terms, chunk_text):
    """Check if question contains paper ID or 3+ consecutive title words (excluding terms in chunk text)."""
    question_text = question['question'].lower()
    paper_id = question['paper_id'].lower()
    chunk_text_lower = chunk_text.lower()
    
    # Check paper ID
    if paper_id in question_text:
        return f"Contains paper ID: {paper_id}"
    
    # Check 3+ consecutive title words, but exclude if they appear in chunk text
    for i in range(len(title_terms) - 2):
        consecutive = ' '.join(title_terms[i:i+3])
        if consecutive in question_text:
            # Check if this phrase appears in chunk text
            if consecutive not in chunk_text_lower:
                return f"Contains 3+ consecutive title words (not in chunk): {consecutive}"
    
    return None

# Load questions
with open('eval_lab/questions.jsonl', 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f]

# Load chunks
with open('data/lab/chunks_256_26.jsonl', 'r', encoding='utf-8') as f:
    chunks = {}
    for line in f:
        chunk = json.loads(line)
        chunks[chunk['chunk_id']] = chunk

# Filter questions
filtered_questions = []
dropped_questions = []

for q in questions:
    chunk_id = q['chunk_id']
    paper_id = q['paper_id']
    
    # Rule 1: Drop 002 chunk 24
    if chunk_id == '002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction_24':
        dropped_questions.append((q, "002 chunk 24 (ratio 0.46)"))
        continue
    
    # Rule 2: Drop 002 chunk 18 if source snippet doesn't match
    if chunk_id == '002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction_18':
        chunk = chunks[chunk_id]
        source_snippet = q['source_snippet']
        chunk_text = chunk['text']
        if normalize_text(source_snippet) not in normalize_text(chunk_text):
            dropped_questions.append((q, "002 chunk 18 (source snippet doesn't match)"))
            continue
    
    # Rule 3: Drop if contains paper ID or 3+ consecutive title words (not in chunk)
    title_terms = extract_title_terms(paper_id)
    chunk_text = chunks.get(chunk_id, {}).get('text', '')
    title_check = check_title_in_question(q, title_terms, chunk_text)
    if title_check:
        dropped_questions.append((q, title_check))
        continue
    
    # Rule 4: Drop if failed word-overlap filter
    chunk_text = chunks.get(chunk_id, {}).get('text', '')
    question_words = set(q['question'].lower().split())
    chunk_words = set(chunk_text.lower().split())
    overlap = len(question_words & chunk_words) / len(question_words) if question_words else 0
    if overlap < 0.3:
        dropped_questions.append((q, f"Word-overlap filter failed (overlap: {overlap:.2f})"))
        continue
    
    filtered_questions.append(q)

print(f"Original questions: {len(questions)}")
print(f"Dropped questions: {len(dropped_questions)}")
print(f"Filtered questions: {len(filtered_questions)}")
print()

print("Dropped questions:")
for q, reason in dropped_questions:
    print(f"  {q['chunk_id']}: {reason}")
    print(f"    Question: {q['question']}")
    print()

# Save filtered questions
with open('eval_lab/questions.jsonl', 'w', encoding='utf-8') as f:
    for q in filtered_questions:
        f.write(json.dumps(q) + '\n')

print(f"Saved {len(filtered_questions)} questions to eval_lab/questions.jsonl")
