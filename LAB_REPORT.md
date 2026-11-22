# Lab Retrieval Pipeline Report

## 1. Goal and Scope Changes

**Original Goal:** Compare multiple embedding models and implement entity-based retrieval.

**Actual Scope:**
- **Model comparison dropped:** Due to daily API limits on free OpenRouter models, switched to local embeddings only (BAAI/bge-small-en-v1.5).
- **Entity retrieval dropped:** Entities are extracted and stored but not used in retrieval (stored only for potential future use).
- **Local embeddings:** Used BAAI/bge-small-en-v1.5 (384 dimensions) via sentence-transformers for zero-cost embeddings.
- **Batched question generation:** Due to free daily API limits, questions were generated in batches across multiple days.

## 2. Pipeline Diagram

```
PDFs (pdfs/)
  ↓
Cleaning (lab_chunk.py --use-cleaned)
  → data/lab/cleaned/*.txt (CRLF line endings)
  ↓
Chunking (lab_chunk.py)
  → data/lab/chunks_256_26.jsonl (356 chunks)
  → data/lab/chunks_384_38.jsonl (237 chunks)
  ↓
Entity Extraction (lab_entities.py)
  → entities field added to chunks (YAKE, NLTK stopwords)
  ↓
Embedding (lab_embed.py)
  → ChromaDB collections (data/chroma_lab/)
  → retrieval_lab_BAAI_bge-small-en-v1.5_384_256_26_vector
  → retrieval_lab_BAAI_bge-small-en-v1.5_384_384_38_vector
  ↓
Vector Search (lab_eval.py)
  → Retrieval with/without query prefix
  ↓
Question Generation (lab_qgen.py)
  → eval_lab/questions.jsonl (39 questions)
  ↓
Evaluation (lab_eval.py)
  → results.json, per_question.csv, failures.csv, sweep_summary.csv
```

## 3. File-by-File Table

| Path | Purpose | Inputs | Outputs | How to Run |
|------|---------|--------|---------|------------|
| scripts/lab_chunk.py | PDF parsing, cleaning, chunking | PDFs from pdfs/ | data/lab/cleaned/*.txt, data/lab/chunks_*.jsonl | `uv run python scripts/lab_chunk.py --use-cleaned --chunk-size 256 --chunk-overlap 26` |
| scripts/lab_entities.py | Entity extraction (YAKE) | data/lab/chunks_*.jsonl | entities field in chunks | `uv run python scripts/lab_entities.py --chunks data/lab/chunks_256_26.jsonl` |
| scripts/lab_embed.py | Embedding generation | data/lab/chunks_*.jsonl | ChromaDB collections | `uv run python scripts/lab_embed.py --model BAAI/bge-small-en-v1.5 --chunks data/lab/chunks_256_26.jsonl` |
| scripts/lab_qgen.py | Question generation | ChromaDB collections | eval_lab/questions.jsonl | `uv run python scripts/lab_qgen.py --model google/gemma-4-31b-it:free` |
| scripts/lab_eval.py | Evaluation | eval_lab/questions.jsonl, chunks, ChromaDB | results.json, per_question.csv, failures.csv, sweep_summary.csv | `uv run python scripts/lab_eval.py` |
| scripts/lab_common.py | Shared configuration | .env file | N/A | N/A (imported by other scripts) |

**LAB_* Variables from .env.example:**
- `LAB_EMBED_BACKEND`: local or openrouter
- `LAB_EMBED_MODEL`: OpenRouter embedding model
- `LAB_EMBED_MODEL_LOCAL`: Local embedding model (BAAI/bge-small-en-v1.5)
- `LAB_CHAT_MODEL`: Chat model for question generation
- `LAB_QGEN_MODEL`: Question generation model
- `LAB_QGEN_FALLBACKS`: Fallback models for question generation
- `LAB_EXTRACT_ENTITIES`: true/false
- `LAB_ENTITY_EXTRACTOR`: yake, spacy_noun_chunks, keybert, llm_open
- `LAB_MAX_ENTITIES_PER_CHUNK`: Max entities per chunk
- `LAB_ENTITY_BOOST`: Entity boost score
- `LAB_QUERY_PREFIX`: Query prefix for embeddings
- `PDF_DIR`: Directory containing PDFs
- `LAB_REQUESTS_PER_MIN`: API rate limit

## 4. Data

**Papers:** 9 papers across 3 fields
- NLP: 3 papers (001, 002, 003)
- Computer Vision: 3 papers (035, 041, 045)
- Machine Learning: 3 papers (072, 074, 090)

**Chunks per config:**
- 256/26: 356 chunks
- 384/38: 237 chunks

**Chunk size definition:** Chunk size is defined in tiktoken (cl100k_base) tokens, not bge-small tokens.

**BGE tokenizer statistics:**
- 256 config: mean 239.6 tokens, max 294 tokens
- 384 config: mean 357.4 tokens, max 423 tokens
- Model max context: 512 tokens

**Token guard:** 480-token limit enforced using bge-small tokenizer (including special tokens).

## 5. Embedding Model

**Model:** BAAI/bge-small-en-v1.5
- Dimensions: 384
- Max sequence length: 512
- Query prefix: "Represent this sentence for searching relevant passages: " (used for queries only)
- Normalization: L2 normalization applied to embeddings
- Backend: Local (sentence-transformers)

## 6. Vector Checks

**ChromaDB verification:**
- 256 collection: 356/356 documents match chunk file (same IDs, same text)
- 384 collection: 237/237 documents match chunk file (same IDs, same text)
- All checks passed (6/6)

## 7. Entities

**Extractor:** YAKE (Yet Another Keyword Extractor)
- Stopwords: NLTK stopwords
- Max entities per chunk: 8
- Max entities per paper: 50
- Min paper frequency: 2

**Storage:** Entities stored in chunk JSON files but not used in retrieval.

**Sample chunks with entities:** 20 random chunks all have non-empty entities.

**Top terms per paper:** (from check_entities_summary.py output)
- Paper-specific top 10 terms extracted and stored.

## 8. Question Generation

**Model:** google/gemma-4-31b-it:free (via OpenRouter)
- Chat requests used: ~49 questions generated
- Styles: factual, conceptual, paraphrased
- Filter counts:
  - 49 generated
  - 10 removed (2 from 002 chunks 18/24 due to text mismatch, 8 due to title-words filter)
  - 39 final questions

**15 sample questions:** (from lab_qgen.py output)
- Distributed across papers and styles

## 9. Results

**6-Config Table (strict hit type):**

| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@10 | MRR | nDCG@10 |
|--------|-------|-------|-------|--------|-----------|-----|--------|
| 256_none | 0.590 | 0.795 | 0.897 | 0.949 | 0.949 | 0.710 | 0.768 |
| 256_instruction | 0.564 | 0.872 | 0.897 | 0.923 | 0.923 | 0.693 | 0.750 |
| 384_none | 0.590 | 0.821 | 0.923 | 0.949 | 0.949 | 0.723 | 0.779 |
| 384_instruction | 0.641 | 0.821 | 0.846 | 0.949 | 0.949 | 0.746 | 0.795 |
| BM25_256 | 0.718 | 0.846 | 0.923 | 1.000 | 1.000 | 0.810 | 0.856 |
| BM25_384 | 0.641 | 0.821 | 0.974 | 1.000 | 1.000 | 0.772 | 0.828 |

**Per Style (from summary_by_group.csv):**

| Config | Style | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|--------|-------|---|-------|-------|-------|--------|-----|
| 256_none | factual | 15 | 0.467 | 0.667 | 0.800 | 0.933 | 0.602 |
| 256_none | conceptual | 13 | 0.615 | 0.769 | 0.846 | 0.923 | 0.727 |
| 256_none | paraphrased | 11 | 0.727 | 1.000 | 1.000 | 1.000 | 0.836 |
| 256_instruction | factual | 15 | 0.467 | 0.733 | 0.800 | 0.933 | 0.611 |
| 256_instruction | conceptual | 13 | 0.615 | 0.846 | 0.846 | 0.846 | 0.695 |
| 256_instruction | paraphrased | 11 | 0.636 | 1.000 | 1.000 | 1.000 | 0.803 |
| 384_none | factual | 15 | 0.667 | 0.733 | 0.867 | 0.933 | 0.780 |
| 384_none | conceptual | 13 | 0.538 | 0.769 | 0.923 | 0.923 | 0.646 |
| 384_none | paraphrased | 11 | 0.545 | 0.909 | 1.000 | 1.000 | 0.735 |
| 384_instruction | factual | 15 | 0.667 | 0.733 | 0.867 | 0.933 | 0.778 |
| 384_instruction | conceptual | 13 | 0.615 | 0.846 | 0.923 | 0.923 | 0.673 |
| 384_instruction | paraphrased | 11 | 0.636 | 0.909 | 1.000 | 1.000 | 0.791 |

**Per Field (from summary_by_group.csv):**

| Config | Field | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|--------|-------|---|-------|-------|-------|--------|-----|
| 256_none | NLP | 14 | 0.571 | 0.786 | 0.857 | 0.929 | 0.697 |
| 256_none | Computer | 13 | 0.538 | 0.769 | 0.846 | 0.923 | 0.655 |
| 256_none | Machine | 12 | 0.667 | 0.833 | 1.000 | 1.000 | 0.785 |
| 256_instruction | NLP | 14 | 0.500 | 0.786 | 0.857 | 0.929 | 0.657 |
| 256_instruction | Computer | 13 | 0.538 | 0.923 | 0.923 | 0.846 | 0.641 |
| 256_instruction | Machine | 12 | 0.667 | 0.917 | 0.917 | 1.000 | 0.792 |
| 384_none | NLP | 14 | 0.643 | 0.786 | 0.929 | 0.929 | 0.741 |
| 384_none | Computer | 13 | 0.538 | 0.769 | 0.923 | 0.923 | 0.676 |
| 384_none | Machine | 12 | 0.583 | 0.917 | 0.917 | 1.000 | 0.753 |
| 384_instruction | NLP | 14 | 0.786 | 0.857 | 0.929 | 0.929 | 0.805 |
| 384_instruction | Computer | 13 | 0.538 | 0.769 | 0.923 | 0.923 | 0.682 |
| 384_instruction | Machine | 12 | 0.583 | 0.833 | 0.667 | 1.000 | 0.748 |

**BM25 vs Vector:**
- BM25_256 outperforms vector configs on Hit@1 (0.718 vs 0.590)
- BM25_384 outperforms vector configs on Hit@1 (0.641 vs 0.590)
- BM25 achieves perfect Hit@10 (1.000) for both configs

**Failure Analysis:**
- Total failures: 10 (across all configs)
- 5 unique questions failing consistently:
  1. "What is the first step in the proposed method to obtain initial diarization results?" (001_NLP, factual) - gold rank: not in top 50
  2. "How does the feature extraction network output a tensor of size a*b*M for each layer, and what do a*b and M represent?" (041_CV, conceptual) - gold rank: not in top 50
  3. "What fixed learning rate and batch size were used to train the localization network..." (041_CV, conceptual) - gold rank: not in top 50

## 10. Conclusions and Limitations

**Conclusions:**
- Local embeddings (bge-small) provide competitive performance at zero cost
- BM25 baseline outperforms vector retrieval on this small dataset
- Query prefix has mixed effects (helps 384, hurts 256)
- 384 chunks show slightly better performance with query prefix

**Limitations:**
1. **Small question set:** The final question set has only 39 questions (49 generated, 10 removed by a title-words filter and the 002 changes). With only 9 papers and 3-6 questions each, statistical significance is limited.
2. **Thin coverage:** ML field and some styles (conceptual) have fewer questions, making comparisons less reliable.
3. **Question sensitivity:** One question is about 2.6 percentage points (1/39 ≈ 2.56%), so differences of one to three questions (256 vs 384, prefix vs none) are not statistically significant.
4. **No lenient/paper-level metrics:** Only strict hit metrics were computed; lenient and paper-level Hit@k were not included in the evaluation outputs.
5. **Entities unused:** Entities were extracted but not used in retrieval, so their impact on performance is unknown.
6. **No model comparison:** Only one embedding model was evaluated due to API limits.
