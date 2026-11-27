# Habiba task implementation status

This document maps the assigned checklist to the implementation and its
verification. It deliberately separates code completion from empirical OCR
quality claims: PaddleOCR or Surya can only be declared better after running the
benchmark on human-verified samples.

| Task | Status | Implementation / evidence |
|---|---|---|
| Evaluate current OCR and select a stronger replacement | Implemented | Replaceable OCR interface in `src/graph_rag/ingestion/ocr.py`; PaddleOCR is the default, Tesseract is the fallback. |
| Benchmark PaddleOCR and Surya across Arabic, English, mixed, scans, and tables | Benchmark ready; dataset run pending | `scripts/benchmark_ocr.py` and `evaluation_datasets/ocr/` define the reproducible manifest and output reports. A real result requires verified samples. |
| CER/WER plus pages, tables, and boxes | Implemented | The benchmark reports CER, WER, latency, confidence, box count, page number, and expected/detected table count. Parser OCR metadata preserves page provenance. |
| Resolve spaCy Windows native-DLL block or verify fallback | Implemented | Native Windows keeps the explicit lexical fallback; Docker validates spaCy NER and the Stanza Arabic UD dependency parser with `scripts/check_nlp_runtime.py`. |
| Review LangChain extraction and select a free extraction model | Implemented | LangChain structured extraction remains open-schema; the default is `google/gemma-4-26b-a4b-it:free`. |
| Arabic evidence normalization while preserving original text/offsets | Implemented | Evidence normalization is used for validation while original evidence and offsets remain in extraction records. |
| Specific rejection reasons | Implemented | Entity, relation, topic, evidence, truncation, and ambiguity diagnostics are persisted with explicit reasons. Topic-cycle rejection was added. |
| Raw provider responses, finish reasons, truncation, token limits | Implemented | Extraction cache/diagnostics retain raw responses and finish metadata; truncation and retry/circuit behavior are tested. |
| Arabic name normalization plus role/relationship context in resolution | Implemented | `entity_resolution.py` normalizes Arabic spelling/diacritics; `graph_builder.py` uses normalized names, aliases, role, organization, relation type, and neighbor context. |
| Topic hierarchy and document tests | Implemented | Cycles are rejected, multi-parent topics are preserved, and focused OCR/entity/topic tests live in `scripts/test_habiba_document_pipeline.py`. |
| Docker runtime | Implemented; image build requires Docker Desktop running | `Dockerfile` and `compose.yaml` run Streamlit, Neo4j, and isolated PaddleOCR; volumes preserve data and models. |

## Verification commands

These commands do not call OpenRouter:

```powershell
.\.venv\Scripts\python.exe scripts\test_habiba_document_pipeline.py
.\.venv\Scripts\python.exe scripts\test_file_formats.py
.\.venv\Scripts\python.exe scripts\test_graph_redesign.py
docker compose config --quiet
```

When Docker Desktop is running:

```powershell
docker compose up --build
docker compose run --rm app python scripts/check_nlp_runtime.py --require-spacy --require-arabic-dependencies --require-ocr paddle
```

Do not use `--refresh-extraction` merely to test Docker; it bypasses the cache
and can create new model requests.
