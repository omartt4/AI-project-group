# `phase_1_graph` Branch

## Purpose

This branch contains the graph-RAG application and a separate retrieval-lab
benchmark for a small, reproducible paper-retrieval experiment. The benchmark
code is local-first and does not require Neo4j for chunking, embedding, vector
checks, or evaluation.

## Branch state

- Branch: `phase_1_graph`
- Upstream: `origin/phase_1_graph`
- PDFs, `data/`, and `.env` are intentionally ignored and must not be
  committed.
- Current state: Entity extraction refined with NLTK stopwords and document-frequency filtering (doc_freq_threshold=6, min_paper_freq=2).

## Retrieval-lab scope

The lab pipeline is:

```text
PDFs -> cleaned text -> overlapping chunks -> local embeddings
     -> Chroma vector search
     -> synthetic questions -> retrieval evaluation
```

The current experiment design uses:

- Nine papers: three NLP, three computer-vision, and three machine-learning
  papers.
- Chunk configurations: 256 tokens with 26-token overlap, and 384 tokens with
  38-token overlap.
- Embedding model: `BAAI/bge-small-en-v1.5`, 384 dimensions, normalized
  embeddings, and a 512-token model limit.
- Query variants: no prefix and
  `Represent this sentence for searching relevant passages: `.
- BM25 as a lexical validity baseline.
- YAKE with NLTK stopwords (sklearn fallback) for stored per-chunk/per-paper entities.
  Entity filtering: min_paper_freq=2 (terms must appear in >= 2 papers for paper_entities.json),
  doc_freq_threshold=6 (terms appearing in >= 6 of 9 papers are dropped from chunks).
  Entities are not used for retrieval or evaluation.
- Synthetic questions only; no manually authored questions.

## Important files

| Path | Role |
|---|---|
| `scripts/lab_chunk.py` | Builds the paper manifest and overlapping chunks. |
| `scripts/lab_embed.py` | Embeds chunks and writes Chroma collections. |
| `scripts/lab_check_vectors.py` | Runs six vector-store sanity checks. |
| `scripts/lab_qgen.py` | Generates and validates synthetic questions. |
| `scripts/lab_eval.py` | Runs validation tests and retrieval evaluation. |
| `scripts/lab_entities.py` | Extracts and stores entity metadata. |
| `scripts/lab_common.py` | Shared lab configuration and collection naming. |
| `eval_lab/questions.jsonl` | Checked-in evaluation questions currently present. |

## Runtime artifacts

The lab expects ignored files under `data/lab/`, including:

- `cleaned/<paper_id>.txt` and `cleaned/manifest.json`
- `chunks_256_26.jsonl`
- `chunks_384_38.jsonl`
- the local embedding cache
- the lab Chroma store at `data/chroma_lab`
- optional generated question data and vector-check reports

These artifacts are not part of the repository. A checkout without them cannot
honestly reproduce the completed experiment until they are restored or
regenerated from the nine PDFs.

## d2 span-mapping validation

`scripts/lab_eval.py` validates that every 256-token gold chunk center maps to
the same paper's 384-token chunks:

- one matching 384 chunk is valid;
- exactly two matches are valid when the center falls in their overlap;
- zero matches, cross-paper matches, or more than two matches fail;
- missing 384-token data fails rather than being skipped or force-passed.

This check must be run over the complete question set before evaluation.

## Safe reproduction outline

Run from the repository root in Windows PowerShell with the project
environment:

```powershell
uv sync
uv run python scripts/lab_chunk.py --chunk-size 256 --chunk-overlap 26
uv run python scripts/lab_chunk.py --chunk-size 384 --chunk-overlap 38
uv run python scripts/lab_embed.py --model BAAI/bge-small-en-v1.5 --chunks data/lab/chunks_256_26.jsonl --mode vector --no-confirm
uv run python scripts/lab_embed.py --model BAAI/bge-small-en-v1.5 --chunks data/lab/chunks_384_38.jsonl --mode vector --no-confirm
uv run python scripts/lab_check_vectors.py --collection retrieval_lab_BAAI_bge-small-en-v1.5_384_256_26_vector
uv run python scripts/lab_check_vectors.py --collection retrieval_lab_BAAI_bge-small-en-v1.5_384_384_38_vector
uv run python scripts/lab_eval.py
```

Do not run question generation or any chat endpoint as part of local
evaluation. Do not use `--force`, and do not delete existing artifacts; rename
old outputs to `.old` before producing replacements.

