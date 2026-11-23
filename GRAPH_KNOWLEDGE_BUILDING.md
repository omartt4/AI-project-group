# Automatic Document Graph Building

The ingestion pipeline treats the source as a generic document, not necessarily
an academic paper. Normal operation is: add supported files to the configured
`pdfs/` folder, run ingestion, then optionally inspect the graph and diagnostics.
Entity, relationship, type, topic, alias, and duplicate decisions are automatic.

## Ingestion and extraction

```text
Document
  -> format-aware parsing and OCR when needed
  -> language detection and source-structure filtering
  -> spaCy candidate analysis
  -> LangChain LLMGraphTransformer (open schema)
  -> exact-evidence and confidence validation
  -> type normalization and entity resolution
  -> topic hierarchy and diagnostics
  -> ChromaDB and Neo4j writes
  -> versioned extraction cache and usage/cost logs
```

spaCy supplies sentence boundaries, tokens, built-in NER, dependencies, and
optional noun-phrase candidates. These are hints only: raw spaCy output is never
written to Neo4j. The primary graph extractor is LangChain's
`LLMGraphTransformer` from `langchain-neo4j`, configured with
`allowed_nodes=[]` and `allowed_relationships=[]`. Entity and relationship
types remain open; the model can return novel labels when the source supports
them.

Documents are parsed by format-aware handlers, with OCR for scanned pages and
image inputs. Language and page/table/section provenance are retained.
Repeated headers and footers, reference sections, and boilerplate are removed
before graph extraction. The full document is sent in one extraction request
unless it exceeds the configured model context budget; only then is it split
into source regions.

Every accepted entity and relationship must include an exact source quote,
source region, and sufficient confidence. Unsupported statements, sentence-
length names, citation artifacts, unresolved pronouns, missing relationship
endpoints, and low-confidence candidates are rejected. Rejections and
ambiguous cases are recorded as diagnostics and do not require routine manual
approval.

## Identity and topic handling

Entity types are normalized against the existing graph vocabulary, with
evidence-aware common synonyms and semantic matching when an embedding model is
configured. New valid types remain accepted. Entity resolution checks names,
aliases, abbreviations, roles, organizations, graph context, and semantic
similarity. Semantic matches require a high score and a clear margin over other
candidates. If identity remains ambiguous, the candidate and dependent
relationships are omitted from the confident graph and stored in
`extraction_diagnostics`.

Topics are open-schema proposals grounded in source evidence and confidence.
The pipeline discovers new topics and parents, merges explicit aliases and
unambiguous near-duplicates, and rejects generic section headings or topic names
that duplicate extracted entities. Ambiguous topic matches are diagnostics, not
new confident duplicates. No fixed topic list or manual candidate approval is
required.

## Storage schema

```text
(Document)-[:HAS_ENTITY_EVIDENCE]->(Entity)
(Entity)-[:RELATION {
  relation_type, document_id, evidence, confidence, provenance
}]->(Entity)
(TopLevelTopic)-[:HAS_SUBTOPIC]->(Subtopic)
(Topic)-[:HAS_DOCUMENT]->(Document)
```

Neo4j stores documents, canonical entities, topics, and generic evidence-backed
relationships. It does not create `Chunk` or `Mention` nodes. Text chunks and
their embeddings, plus canonical-entity vectors for resolution, live in
ChromaDB. SQLite `graph_evidence` records preserve entity, relationship, topic,
and resolved-mention text, evidence, source document, page/section, confidence,
and provenance. Ambiguous mentions and other non-blocking cases remain
available in `unresolved_references` and `extraction_diagnostics`.

Structured rows from CSV, Excel, JSON, and Parquet are retained in
`structured_tables`. Page text, OCR text, offsets, bounding boxes, extraction
method, confidence, and language are retained in `document_pages`.

## Cache, idempotency, and rebuilds

Extraction cache signatures include document content hash, extraction model,
prompt version, pipeline version, and normalization version. Renaming unchanged
content does not trigger duplicate ingestion. New or changed documents are
processed independently. A per-document cross-process lock prevents concurrent
duplicate extraction.

```powershell
uv run python scripts\ingest.py
uv run python scripts\ingest.py --file "report.pdf" --rebuild-graph
```

`--rebuild-graph` requires a complete compatible extraction cache and does not
call the extraction model. An incomplete or stale cache fails clearly instead
of falling back to a paid extraction request. Use `--refresh-extraction` only
when a fresh model extraction is intentionally desired.

## Usage and cost accounting

Each external model request is logged with operation, model, input/output/total
tokens when provided, provider-reported cost, locally estimated cost, latency,
retry count, success/failure, cache status, and request/run identifiers. Cost
is calculated locally from the versioned rate card:

```text
cost = (input_tokens * input_USD_per_million
      + output_tokens * output_USD_per_million) / 1,000,000
```

No model is asked to calculate cost. If token usage or a rate is unavailable,
the estimated cost is explicitly unknown; it is not silently treated as zero.
Provider-reported cost is retained separately for comparison. See
`src/graph_rag/llm/pricing.py` and the rate-card references in `README.md`.
Neo4j hosting, local compute, and storage costs are outside this request-level
tracker.

## Safe validation

Tests use mocked model responses and local temporary storage. Do not run
production ingestion, paid model calls, database deletion, or graph rebuilds
without explicit approval.
