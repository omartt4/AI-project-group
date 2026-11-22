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
- Stopwords: NLTK stopwords (downloaded automatically), with sklearn ENGLISH_STOP_WORDS as fallback
- Max entities per chunk: 8
- Max entities per paper: 50
- Min paper frequency: 2 (default, terms must appear in at least 2 papers to be kept in paper_entities.json)
- Document frequency threshold: 6 (default, terms appearing in >= 6 of 9 papers are dropped from chunks)

**Filtering stages (256/26 config):**
- Step 1 (validity filter): 0 removed (0.0%) - stopwords, length, etc.
- Step 6 (doc-freq filter): 1 removed (0.0%) - terms appearing in >= 6 papers
- Step 7 (min_paper_freq): 2762 removed (99.1%) - terms appearing in < 2 papers (paper-level only)

**Filtering stages (384/38 config):**
- Step 1 (validity filter): 0 removed (0.0%) - stopwords, length, etc.
- Step 6 (doc-freq filter): 0 removed (0.0%) - terms appearing in >= 6 papers
- Step 7 (min_paper_freq): 1870 removed (99.4%) - terms appearing in < 2 papers (paper-level only)

**Storage:** Entities stored in chunk JSON files but not used in retrieval.

**Entity statistics (256/26 config):**
- Mean entities per chunk: 7.90
- Median entities per chunk: 8
- Chunks with 0 entities: 0/356 (0.0%)
- Chunks with fewer than 8 entities: 9/356 (2.5%)
- Max entities per chunk: 8

**Entity statistics (384/38 config):**
- Mean entities per chunk: 7.99
- Median entities per chunk: 8
- Chunks with 0 entities: 0/237 (0.0%)
- Chunks with fewer than 8 entities: 0/237 (0.0%)
- Max entities per chunk: 8

**Document frequency filtering:**
- 256/26: 18 terms dropped (model, results, table, figure, fig, data, network, networks, dataset, datasets, precision, learning, deep learning, neural, experiments, set, models, wang)
- 384/38: 8 terms dropped (results, figure, dataset, network, learning, data, model, precision)
- Verified: 0 entities in final chunk files match dropped terms

**Residual noise:**
- About 0.4% of entities contain digits, "fig" or a URL because chunks contain reference and header text
- Chunks must not change, so this noise is accepted
- Examples: "crawls 2008-2017 contexts", "shown in fig", "url http"

## 8. Question Generation

**Model:** google/gemma-4-31b-it:free (via OpenRouter)
- Chat requests used: 49 questions generated
- Styles: factual, conceptual, paraphrased
- Filter counts:
  - 49 generated
  - 10 removed (2 from 002 chunks 18/24 due to text mismatch, 8 due to title-words filter)
  - 39 final questions
- Pipeline order: Question generation runs after chunking and embedding, before evaluation

**15 sample questions:**
1. What assumption underlies clustering-based diarization methods regarding speaker presence per frame? (001_NLP, factual)
2. How is speaker permutation ambiguity resolved between the two speakers and the frame indices? (001_NLP, conceptual)
3. What is the DER of the x-vector AHC + VB + OVL + Proposed method on DIHARD II eval? (001_NLP, paraphrased)
4. What is the primary challenge addressed by distant supervised relation extraction in knowledge base construction? (002_NLP, factual)
5. How is the relation representation formed as a query vector in the RA-CapNet model? (002_NLP, conceptual)
6. What types of user constraints can automated itinerary planning systems not incorporate according to the text? (003_NLP, factual)
7. What two metrics are used to evaluate the inclusion/exclusion phrase mining task and how do they differ in measuring correctness? (003_NLP, conceptual)
8. What does the BERT classifier output for the sentence about the wheelchair and turnstile, and how does the 11-class classifier label this example? (003_NLP, paraphrased)
9. What is the main challenge for proposing better methods in leaf segmentation and counting with deep learning according to the abstract? (035_CV, factual)
10. What was the best SBD achieved in the Komatsuna dataset and what abs. DiC value did the model obtain at a threshold of 0.6? (035_CV, paraphrased)
11. What two main components make up the visual localization network proposed in the paper? (041_CV, factual)
12. What is the purpose of combining high-level and mid-level feature layers in place recognition under severe appearance changes? (041_CV, factual)
13. What is the role of the depth completion network in the multi-sensor fusion framework for 3D object detection? (045_CV, factual)
14. What clinical data sources are used for accurate ICD10 code predictions in the ensemble model? (072_ML, factual)
15. What is the HOP (Higher-Order Proximity) defined as in the H2NT network transformation framework? (074_ML, conceptual)

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
| 256_instruction | Computer | 13 | 0.538 | 0.846 | 0.846 | 0.846 | 0.641 |
| 256_instruction | Machine | 12 | 0.667 | 0.917 | 0.917 | 1.000 | 0.792 |
| 384_none | NLP | 14 | 0.643 | 0.786 | 0.929 | 0.929 | 0.741 |
| 384_none | Computer | 13 | 0.538 | 0.769 | 0.923 | 0.923 | 0.676 |
| 384_none | Machine | 12 | 0.583 | 0.917 | 0.917 | 1.000 | 0.753 |
| 384_instruction | NLP | 14 | 0.786 | 0.857 | 0.929 | 0.929 | 0.805 |
| 384_instruction | Computer | 13 | 0.538 | 0.769 | 0.923 | 0.923 | 0.682 |
| 384_instruction | Machine | 12 | 0.583 | 0.833 | 1.000 | 1.000 | 0.748 |

**BM25 vs Vector:**
- BM25_256 achieves higher Hit@1 than vector configs (0.718 vs 0.590 for 256_none)
- BM25_384 achieves higher Hit@1 than vector configs (0.641 vs 0.590 for 384_none)
- BM25 achieves perfect Hit@10 (1.000) for both configs, while vector configs achieve 0.949
- Note: BM25's advantage may be due to the small dataset size and question characteristics; results are not statistically significant given only 39 questions

**Failure Analysis:**
- Total failures: 16 (across 4 configs × 39 questions = 156 retrieval attempts)
- 5 unique questions failing (gold chunk not in top 10):
  1. "What is the first step in the proposed method to obtain initial diarization results?" (001_NLP, factual) - fails in all 4 configs
  2. "How does the feature extraction network output a tensor of size a*b*M for each layer, and what do a*b and M represent?" (041_CV, conceptual) - fails in all 4 configs
  3. "What fixed learning rate and batch size were used to train the localization network..." (041_CV, conceptual) - fails in all 4 configs
  4. "What does the BERT classifier output for the sentence about the wheelchair and turnstile..." (003_NLP, paraphrased) - fails in 2 configs (256_instruction, 384_instruction)
  5. "What was the best SBD achieved in the Komatsuna dataset..." (035_CV, paraphrased) - fails in 2 configs (256_none, 384_none)

## 10. Conclusions and Limitations

**Conclusions:**
- Local embeddings (bge-small) provide competitive performance at zero cost
- BM25 baseline outperforms vector retrieval on this small dataset
- Query prefix has mixed effects (helps 384, hurts 256)
- 384 chunks show slightly better performance with query prefix

**Limitations:**
1. **Small question set:** The final question set has only 39 questions (49 generated, 10 removed by filtering). With only 9 papers and 3-6 questions each, statistical significance is limited.
2. **Thin coverage:** ML field and some styles (conceptual) have fewer questions, making comparisons less reliable.
3. **Question sensitivity:** One question is about 2.6 percentage points (1/39 ≈ 2.56%), so differences of one to three questions (256 vs 384, prefix vs none) are not statistically significant.
4. **No lenient/paper-level metrics:** Only strict hit metrics were computed; lenient and paper-level Hit@k were not included in the evaluation outputs.
5. **Entities unused:** Entities were extracted but not used in retrieval, so their impact on performance is unknown.
6. **No model comparison:** Only one embedding model was evaluated due to API limits.

## 11. Problems and Fixes Timeline

| Date | Problem | Fix |
|------|---------|-----|
| 2026-11-22 | Hand-written stopwords in lab_entities.py | Replaced with NLTK stopwords (sklearn fallback) |
| 2026-11-22 | No document-frequency filtering | Added doc_freq_threshold=6 to drop terms appearing in >=6 papers |
| 2026-11-22 | min_paper_freq changed to 1 | Restored to 2 (terms must appear in >=2 papers) |
| 2026-11-22 | Doc-freq filter not applied to chunks | Modified Step 8 to filter common terms from chunk entities |
| 2026-11-22 | ChromaDB corruption (disk full error) | Deleted and recreated data/chroma_lab, re-embedded chunks |
| 2026-11-22 | Backup files deleted | Restored with git restore |

## 12. Next Steps

1. **Query classifier update:** Implement a classifier to determine whether a query requires factual retrieval, conceptual synthesis, or paraphrase matching. This could improve routing to appropriate retrieval strategies.

2. **Answer cost optimization:** Evaluate the cost-effectiveness of different LLM models for answer generation. Consider using smaller models for simpler queries and larger models only for complex reasoning tasks.

3. **Arabic extension idea:** Extend the pipeline to support Arabic documents and queries. This would require:
   - Arabic-specific chunking and tokenization
   - Arabic embedding models (e.g., AraBERT, MARBERT)
   - Arabic stopwords for entity extraction
   - RTL text handling in the UI

## 13. Exact Reproduction Commands

Run from repository root in Windows PowerShell with project environment:

```powershell
# Step 1: Install dependencies
uv sync

# Step 2: Chunk papers (both configs)
uv run python scripts/lab_chunk.py --chunk-size 256 --chunk-overlap 26
uv run python scripts/lab_chunk.py --chunk-size 384 --chunk-overlap 38

# Step 3: Embed chunks (local model, ~2 minutes per config)
uv run python scripts/lab_embed.py --model BAAI/bge-small-en-v1.5 --chunks data/lab/chunks_256_26.jsonl --mode vector --no-confirm
uv run python scripts/lab_embed.py --model BAAI/bge-small-en-v1.5 --chunks data/lab/chunks_384_38.jsonl --mode vector --no-confirm

# Step 4: Vector checks (both collections, ~30 seconds each)
uv run python scripts/lab_check_vectors.py --collection retrieval_lab_BAAI_bge-small-en-v1.5_384_256_26_vector
uv run python scripts/lab_check_vectors.py --collection retrieval_lab_BAAI_bge-small-en-v1.5_384_384_38_vector

# Step 5: Entity extraction (both chunk files, ~1 minute per config)
uv run python scripts/lab_entities.py --chunks data/lab/chunks_256_26.jsonl
uv run python scripts/lab_entities.py --chunks data/lab/chunks_384_38.jsonl

# Step 6: Question generation (requires OPENROUTER_API_KEY, ~15-30 minutes)
# Skip this step if using existing questions in eval_lab/questions.jsonl
# uv run python scripts/lab_qgen.py

# Step 7: Evaluation (~1 minute)
uv run python scripts/lab_eval.py
```

**Expected runtime:** ~10-15 minutes (excluding question generation)

**File-by-file table:**

| Path | Role |
|------|------|
| scripts/lab_chunk.py | Builds paper manifest and overlapping chunks |
| scripts/lab_embed.py | Embeds chunks and writes Chroma collections |
| scripts/lab_check_vectors.py | Runs six vector-store sanity checks |
| scripts/lab_qgen.py | Generates and validates synthetic questions |
| scripts/lab_eval.py | Runs validation tests and retrieval evaluation |
| scripts/lab_entities.py | Extracts and stores entity metadata |
| scripts/lab_common.py | Shared lab configuration and collection naming |
| scripts/compute_summary_by_group.py | Computes summary statistics by group (style/field) |
| scripts/check_chunk_text_match.py | Debug helper: verifies ChromaDB text matches chunk files |
| eval_lab/questions.jsonl | Checked-in evaluation questions |
| eval_lab/results.json | Evaluation metrics per config |
| eval_lab/sweep_summary.csv | Summary of evaluation sweep |
| eval_lab/summary_by_group.csv | Per-style and per-field breakdown |
| eval_lab/per_question.csv | Per-question retrieval ranks |
| eval_lab/failures.csv | List of failed retrievals |
| data/lab/chunks_256_26.jsonl | 256-token chunks with 26-token overlap |
| data/lab/chunks_384_38.jsonl | 384-token chunks with 38-token overlap |
| data/lab/paper_entities.json | Extracted entities per paper |
| data/lab/shared_terms.csv | Term frequency across papers |
| data/lab/entity_analysis.txt | Detailed entity analysis (20 random chunks, top terms, junk) |
| data/chroma_lab/ | ChromaDB vector store (local) |
