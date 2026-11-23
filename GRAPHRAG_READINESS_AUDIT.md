## Audit summary

This is a checklist-based estimate of implementation completeness, **not code coverage**. The project has a substantial automatic ingestion and graph pipeline, but it does not yet satisfy the full approved plan—most notably for Arabic evidence matching, observability, retrieval evaluation, and cross-store recovery.

| Phase | Status | Completion | Implemented evidence and tests | Main gap |
|---|---:|---:|---|---|
| 1. Repository audit | 🟡 Partial | 75% | Architecture and safe-rebuild guidance in [README.md](./README.md) and [GRAPH_KNOWLEDGE_BUILDING.md](./GRAPH_KNOWLEDGE_BUILDING.md). | The older [GRAPH_REDESIGN_PLAN.md](./GRAPH_REDESIGN_PLAN.md) conflicts with the current design; README diagrams and terminology are also partly research-paper-specific. No audit-documentation test. |
| 2. Unified document ingestion | 🟡 Partial | 85% | Format parsers, generic document IDs, language detection, and hash deduplication are present. `test_file_formats.py` covers Arabic Parquet, tables, OCR routing, and renamed-file deduplication. | Supports registered parsers, not arbitrary unknown file formats. Mixed-language processing is not comprehensively quality-tested. |
| 3. OCR, images, and tables | 🟡 Partial | 75% | Replaceable OCR adapter, configurable `eng+ara`, page/table preservation and bounding-box handling. Mocked OCR/table tests pass in `test_file_formats.py`. | No OCR quality metrics or verified real Arabic/mixed-language OCR evaluation. |
| 4. Open-schema graph extraction | 🟡 Partial | 80% | [`EntityExtractor.extract_document`](./src/graph_rag/graph/entity_extractor.py) uses `LLMGraphTransformer`, empty allowed-type lists, and evidence/confidence fields. Prompt regression tests pass. | No real Arabic or English structured-output verification in this audit; that requires model calls. |
| 5. spaCy and extraction libraries | 🔴 Blocked | 80% | Lazy loading and independent capability checks; four mocked capability cases run within `test_file_formats.py`. | spaCy’s native DLL was blocked by Windows Application Control in the reported environment. The lexical fallback allows ingestion to continue, but actual spaCy capability is not verified on this machine. |
| 6. Automatic validation | 🟡 Partial | 55% | Evidence, confidence, name-quality, and endpoint checks in [`EntityExtractor.extract_document`](./src/graph_rag/graph/entity_extractor.py); references and repeated lines are filtered in [`prepare_document_input`](./src/graph_rag/ingestion/document_structure.py). | Evidence matching is case/whitespace tolerant, **not Arabic-normalized**. Diagnostics say an item was rejected but usually not the specific reason. No entity-count cap was found; there are name-length/quality filters. |
| 7. Entity resolution | 🟡 Partial | 75% | Exact/alias matching, compatible-type and attribute checks, semantic matching at a high threshold with a margin, and ambiguity diagnostics in [`EntityResolver.resolve`](./src/graph_rag/graph/entity_resolution.py). Tests cover ambiguous matches and type normalization. | Arabic name normalization is absent. Role/relationship context is not fully integrated into resolution; a graph role lookup exists, but the resolver does not use it as a general strategy. |
| 8. Graph representation | ✅ Complete | 100% | Generic `Document`, legacy `Paper` compatibility, canonical entities/topics and generic `RELATION` edges; chunks in ChromaDB; evidence/provenance in SQLite. Graph tests cover open relationships and compact graph writes. | Complete against this phase’s checklist and mocked/local tests; this does not prove live-database consistency. |
| 9. Automatic topic hierarchy | 🟡 Partial | 75% | [`TopicTaxonomy.assign`](./src/graph_rag/graph/topic_taxonomy.py) handles grounded, high-confidence topics, parent links, aliases and diagnostics. Topic tests pass. | No comprehensive tests for rejecting datasets/people as topics across domains; no real-model topic-quality evaluation. |
| 10. Document-level extraction | 🟡 Partial | 80% | [`_extract_document_graph`](./src/graph_rag/ingestion/pipeline.py) prepares a document and splits only when the context budget requires it; repeated text/references are filtered. | Input/output token limits and provider truncation are not fully recorded. There is no reliable per-segment record of pages omitted or output truncation. |
| 11. Extraction cache | 🟡 Partial | 65% | Cache signature includes content hash, model, prompt, pipeline and normalization versions; locks prevent duplicate concurrent extraction; graph rebuild reuses ready cache. Mocked cache/rebuild tests pass. | Cache stores structured pre/post-finalization results, **not the provider’s complete raw response** or its finish reason; usage/cost is not embedded in the cache. Refresh-cost warning and configuration/language versioning are incomplete. |
| 12. Vector knowledge layer | 🟡 Partial | 70% | LangChain splitter, configurable chunking and page-aware metadata; vector replacement/deduplication tests pass. | Cross-language retrieval quality and embedding-version behavior are not validated; semantic-boundary quality is not measured. |
| 13. Transactional ingestion | 🟡 Partial | 70% | Ingestion journal, resumable stages, failure handling and mocked rollback/retry tests exist. | ChromaDB and Neo4j do not share an atomic transaction. Recovery helps, but temporary cross-store inconsistency remains possible. |
| 14. Multilingual model support | 🟡 Partial | 60% | Separate configured extraction, chat, evaluation and embedding models; OCR languages configurable; answer prompt requests the question’s language. | No explicit low-cost/high-quality profiles, no full cross-language retrieval tests, and no verified Arabic↔English end-to-end results. |
| 15. Hybrid retrieval | 🟡 Partial | 55% | [`HybridRetriever.adaptive_retrieve`](./src/graph_rag/retriever/hybrid_retriever.py) runs vector and graph retrieval in parallel, then aggregates evidence. | No dedicated hybrid retrieval metrics or learned reranker. The query path can still generate an answer with empty/weak context; insufficient evidence and zero graph results are not clearly surfaced as a distinct outcome. |
| 16. LangChain concurrency | 🟡 Partial | 55% | Runnable-based extraction/retrieval, configured `max_concurrency`, and same-document extraction locks. | No async path, sequential/concurrent benchmark, or confirmed `Retry-After` plus bounded exponential-backoff behavior. Retry-count fields exist, but actual retry accounting is not fully verified. |
| 17. Cost and usage tracking | 🟡 Partial | 65% | [`CostTracker.record`](./src/graph_rag/llm/cost_tracker.py) records model/operation, token counts, provider and estimated costs, latency, status, retry and cache fields. Cost tests pass. | No cached-token field, hard cost limit, warning threshold, or required summaries per document/question/language. Local OCR has no model-cost record. |
| 18. Evaluation | 🟡 Partial | 40% | Optional answer evaluation; setting defaults to disabled; failures are represented as unavailable. One mocked failure test passes. | No parsing/OCR/extraction/resolution/topic or retrieval gold-set metrics such as Recall@K, MRR or nDCG. |
| 19. Configuration | 🟡 Partial | 70% | Model roles, OCR languages, concurrency, thresholds, cache paths, retry limits and versions are configurable in [config.py](./src/graph_rag/config.py). `.env.example` uses placeholders. | No cost/request limits, reranker configuration, low/high profiles, or configuration for supported languages/file types. Runtime code includes development credential fallbacks; production configuration should override them. |
| 20. Streamlit UI | 🟡 Partial | 50% | UI includes graph/chat/source views and recent model usage, including cache status. | It does not expose all requested diagnostics: per-document language/OCR/spaCy status, cache hit/miss details, current-run cost by operation, and clear rate-limit status. |
| 21. Automated tests | 🟡 Partial | 60% | The two safe suites passed: `test_file_formats.py` (**36 tests, OK**) and `test_graph_redesign.py` (**passed; 23 test functions**). | Arabic evidence normalization, hybrid retrieval metrics, rate limits and cross-language retrieval lack focused coverage. `scripts/test_query_engine.py` is model-backed and does not have an explicit opt-in guard, so it was not run. |
| 22. Documentation | 🟡 Partial | 60% | Current setup/pipeline/rebuild guidance is in README and `GRAPH_KNOWLEDGE_BUILDING.md`. | The old redesign plan describes Paper/Chunk/Mention nodes and manual approval, contrary to current code. README also retains stale paper-oriented descriptions/diagrams. |

### Counts

- **Complete:** 1
- **Partial:** 20
- **Not started:** 0
- **Blocked:** 1
- **Implemented but unverified:** 0

“Blocked” refers specifically to running spaCy itself on this Windows environment; the fallback path is available and the mocked capability tests pass.

### Five highest-priority remaining tasks

1. Add Arabic-normalized evidence matching while preserving original text and accurate citation offsets.
2. Record specific rejection reasons; retain provider output/finish reason so truncation and rejected items can be diagnosed.
3. Add retrieval evaluation and an explicit insufficient-evidence response; do not present empty graph retrieval as a successful graph result.
4. Improve cross-store recovery reporting and consistency checks for ChromaDB/Neo4j.
5. Add cross-language extraction/retrieval tests and cost/retry limit observability.

**Smallest safe next phase:** implement and mock-test Arabic evidence normalization plus reason-coded rejection diagnostics. It can be verified without a paid API or database rebuild.

### Tests and verification

- **Passed:** `uv run python scripts\test_file_formats.py` — 36 tests, `OK`.
- **Passed:** `uv run python scripts\test_graph_redesign.py` — `Graph redesign tests passed`. Its logged simulated Neo4j write failure is part of a failure/recovery test, not a suite failure.
- **Failed:** None of the suites run in this audit.
- **Not run:** model-backed query/connection scripts, ingestion, database rebuilds, or real OCR/model verification.

### Documentation-only or disconnected items

- The older [GRAPH_REDESIGN_PLAN.md](./GRAPH_REDESIGN_PLAN.md) is a stale plan, not the current graph implementation.
- OCR evaluation metrics and retrieval benchmark metrics are not implemented; they should not be treated as available just because the docs discuss evaluation/observability.
- `GraphBuilder.find_person_by_role` exists, but general role-based matching is not integrated into `EntityResolver.resolve`.
- The `GRAPH_EXTRACTION_MAX_CHARS` setting appears in the legacy structured-extraction path; the main document path uses context-budget splitting instead.

### Paid verification required

A paid/configured model call would be needed to verify real Arabic and English structured extraction, model truncation behavior, and live cross-language embedding/retrieval quality. None was made for this audit.

**No files were changed. No ingestion, database rebuild, deletion, or external API call was performed.**
