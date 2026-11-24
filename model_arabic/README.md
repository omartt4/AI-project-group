# Arabic retrieval mini-lab

This isolated lab evaluates retrieval over the `body` text of two Arabic books using `intfloat/multilingual-e5-small` and a whitespace-tokenized BM25 baseline.

## Run locally

Place the two parquet files in `raw/`, then run from the repository root:

```powershell
uv run python model_arabic\scripts\inspect_parquet.py
uv run python model_arabic\scripts\clean_text.py
uv run python model_arabic\scripts\chunk_text.py
uv run python model_arabic\scripts\embed_chunks.py
uv run python model_arabic\scripts\prepare_questions_ar.py
uv run python model_arabic\scripts\evaluate_ar.py
```

The raw books, cleaned texts, chunks, and Chroma database are local-only and are not committed.

## Evaluation

The supplied 22 questions use exact answer spans in the cleaned text. Results:

| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| E5 vector | 12/22 (54.5%) | 20/22 (90.9%) | 22/22 (100%) | 0.705 |
| BM25 | 10/22 (45.5%) | 19/22 (86.4%) | 20/22 (90.9%) | 0.635 |

With 22 questions, one question changes a rate by about 4.5 percentage points; differences of 1–2 questions are not meaningful. Confidence intervals are included for Hit@1 and Hit@10 in `eval/results_ar.json`.

## Limits

Cleaning preserves Arabic diacritics and letter forms, but includes only the `body` column (not page headers or footnotes). The small question set and two-book corpus are not representative of Arabic retrieval generally.
