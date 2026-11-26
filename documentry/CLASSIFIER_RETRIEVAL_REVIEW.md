# Classifier & Retrieval Review
**Hybrid Graph RAG — Intent Classification & Retrieval Limits Analysis**
*Reviewed: 2026-11-02 | Scope: `query_classifier.py`, `concept_discovery.py`, `hybrid_retriever.py`, `diversity_retriever.py`*

> [!IMPORTANT]
> This is a **findings-only document**. No code changes have been made. All recommendations require explicit approval before implementation.

---

## Table of Contents

1. [Methodology](#1-methodology)
2. [Part 1 — Intent Classification Findings](#2-part-1--intent-classification-findings)
3. [Part 2 — Retrieval Limit Findings](#3-part-2--retrieval-limit-findings)
4. [Summary Table](#4-summary-table)
5. [Recommended Changes (Pending Approval)](#5-recommended-changes-pending-approval)

---

## 1. Methodology

All findings are grounded in direct code inspection and log evidence:

| Source | What was read |
|---|---|
| `src/graph_rag/retriever/query_classifier.py` | All 245 lines — all 6 intent classes, all regex patterns, all parameter allocations, entity extraction logic |
| `src/graph_rag/retriever/hybrid_retriever.py` | All 170 lines — adaptive retrieve flow, classification → MMR → graph → aggregation |
| `src/graph_rag/retriever/diversity_retriever.py` | Lines 40–75 — `candidate_k` floor, ChromaDB call pattern |
| `src/graph_rag/graph/concept_discovery.py` | Lines 50–180 — keyword cap, hub limit, relationship cap |
| `data/app.log` | All 99 lines — Nov 2 session covering adaptive RAG queries and their classified intents |

Every finding below includes: the file, the specific line(s), and (where available) the log evidence that triggered it.

---

## 2. Part 1 — Intent Classification Findings

### Finding C-1 — `"what are"` missing from FACTUAL_LOOKUP patterns

**Severity: HIGH | Confirmed by log**

**File:** `query_classifier.py` — `FACTUAL_PATTERNS` list  
**Log evidence:** Line 64–70 in `app.log`: `"what are the rag types?"` → classified as `EXPLORATORY_THEMATIC`

**Root cause:**  
The `FACTUAL_PATTERNS` regex reads:
```python
r"\b(?:what\s+is|define|definition\s+of|explain\s+the\s+concept\s+of|...)\b"
```
It matches `"what is"` (singular) but **not** `"what are"` (plural). As a result, any query phrased in the plural form bypasses `FACTUAL_LOOKUP` entirely and falls through to `EXPLORATORY_THEMATIC`.

**Impact:**  
- Wrong intent → wrong parameter set → lower `K`, different graph traversal depth.
- In the Nov 2 log, the LLM produced a completely wrong answer about surgical dataset classes instead of answering the RAG-types question.

**Recommended fix:**  
Add `what\s+are` to the existing `FACTUAL_PATTERNS` alternation group alongside `what\s+is`.

---

### Finding C-2 — Word-count gate (`<= 10`) too strict on FACTUAL_LOOKUP

**Severity: MEDIUM**

**File:** `query_classifier.py` — approximately line 157  
**Condition:** `if factual_match and len(lower_query.split()) <= 10:`

**Root cause:**  
The gate rejects factual-pattern matches in queries longer than 10 words. This is intended to avoid miscategorizing long analytical questions as factual, but the threshold is too low:

| Query | Words | Actual intent | Classified as |
|---|---|---|---|
| `"what is the exact definition of retrieval-augmented generation?"` | 9 | FACTUAL ✅ | FACTUAL ✅ |
| `"what is the difference between dense and sparse retrieval augmented generation?"` | 12 | FACTUAL/COMPARATIVE | EXPLORATORY ❌ |
| `"what are the key components of a graph-based RAG pipeline?"` | 13 | FACTUAL | EXPLORATORY ❌ |

**Recommended fix:**  
Raise the word-count gate from `<= 10` to `<= 15`. This covers most genuine factual questions without opening the gate to long analytical queries.

---

### Finding C-3 — `"types of"`, `"kinds of"`, `"categories of"` not covered

**Severity: MEDIUM**

**File:** `query_classifier.py` — `FACTUAL_PATTERNS` and `EXPLORATORY_PATTERNS`

**Root cause:**  
No pattern matches these phrasing variants:
- `"what types of embeddings exist?"`
- `"list the kinds of RAG architectures"`
- `"what are the categories of retrieval methods?"`

These fall through to `EXPLORATORY_THEMATIC` only because nothing earlier matches. While `EXPLORATORY_THEMATIC` is not entirely wrong, it allocates fewer chunks (`K=5–6`) and broader graph traversal — suboptimal for a bounded enumeration question.

**Recommended fix:**  
Add `r"\b(?:types?\s+of|kinds?\s+of|categories?\s+of|forms?\s+of|variants?\s+of)\b"` to `FACTUAL_PATTERNS`. This is a high-precision signal for enumeration/definitional intent.

---

### Finding C-4 — SUMMARIZATION patterns miss common natural-language phrasings

**Severity: MEDIUM**

**File:** `query_classifier.py` — `SUMMARIZATION_PATTERNS`

**Current patterns (confirmed from code):**  
- `r"\b(?:summarize|summary\s+of|overview\s+of|...)\b"`
- `r"\b(?:corpus|knowledge\s+base|all\s+(?:my\s+)?(?:documents|papers|files))\b"`

**Missing phrasings and their expected classification:**

| Query | Should be | Currently |
|---|---|---|
| `"what does the corpus say about RAG?"` | SUMMARIZATION | EXPLORATORY (corpus word present but requires exact form) |
| `"give me a broad overview across all 100 papers"` | SUMMARIZATION | EXPLORATORY |
| `"summarize the research on retrieval methods"` | SUMMARIZATION | EXPLORATORY (no "summarize the research on" pattern) |
| `"what do all the papers say about embeddings?"` | SUMMARIZATION | EXPLORATORY |
| `"throughout the knowledge base, what are the main themes?"` | SUMMARIZATION | EXPLORATORY |

**Recommended fix:**  
Add to `SUMMARIZATION_PATTERNS`:
```python
r"\bacross\s+all\s+(?:papers?|documents?|files?|research)\b",
r"\bwhat\s+do\s+(?:all|the)\s+(?:papers?|documents?)\s+say\b",
r"\bsummarize\s+the\s+research\s+(?:on|about)\b",
r"\bthroughout\s+the\s+(?:knowledge\s+base|corpus|dataset)\b",
r"\bwhat\s+does\s+the\s+(?:corpus|knowledge\s+base)\s+say\b",
```

---

### Finding C-5 — RELATIONAL patterns miss key connection phrasings

**Severity: MEDIUM**

**File:** `query_classifier.py` — `RELATIONAL_PATTERNS`

**Current patterns catch:** `"relationship between"`, `"how X leads to Y"`, `"path from X to Y"`, `"chain of"`, `"depends on"`, `"cascades from"`.

**Missing phrasings:**

| Query | Should be | Currently |
|---|---|---|
| `"how does BERT relate to attention mechanisms?"` | MULTI_HOP_RELATIONAL | EXPLORATORY |
| `"what is the connection between RAG and LLMs?"` | MULTI_HOP_RELATIONAL | FACTUAL (if `<= 10 words`) |
| `"how are dense and sparse retrieval linked?"` | COMPARATIVE or RELATIONAL | EXPLORATORY |
| `"what is the link between graph traversal and retrieval quality?"` | MULTI_HOP_RELATIONAL | EXPLORATORY |

**Root cause:**  
`RELATIONAL_PATTERNS` does not include `"connection between"`, `"how does X relate to"`, `"linked to"`, or `"how are X and Y connected"`.

**Recommended fix:**  
Add to `RELATIONAL_PATTERNS`:
```python
r"\b(?:connection|link|linkage)\s+between\b",
r"\bhow\s+(?:does|do)\s+.+\s+relate\s+to\b",
r"\bhow\s+are\s+.+\s+(?:connected|linked|related)\b",
r"\bwhat\s+(?:is|are)\s+the\s+(?:link|connection|relationship)\s+between\b",
```

---

### Finding C-6 — CONVERSATIONAL anchor patterns fail on punctuation

**Severity: LOW**

**File:** `query_classifier.py` — `CONVERSATIONAL_PATTERNS`

**Root cause:**  
The greeting patterns use `^...$` full-match anchors. Any leading or trailing punctuation causes them to fail:

| Query | Should be | Currently |
|---|---|---|
| `"hi, how are you?"` | CONVERSATIONAL | EXPLORATORY (comma after `hi` breaks anchor) |
| `"hello, can you help me?"` | CONVERSATIONAL | EXPLORATORY |
| `"hey! what's up?"` | CONVERSATIONAL | EXPLORATORY |

**Recommended fix:**  
Change anchors to use looser boundary patterns and strip punctuation before matching, or use `re.sub(r'[^\w\s]', '', query)` before applying CONVERSATIONAL patterns.

---

### Finding C-7 — Entity extraction returns stop words in original casing

**Severity: LOW**

**File:** `query_classifier.py` — `_extract_entities()` method

**Evidence (from test output):**  
Query: `"Summarize all files in the knowledge base"` → `extracted_entities: ['Summarize', 'all', 'knowledge', 'base']`

**Root cause:**  
The stop word filter uses `.lower()` comparison, but the stop word list likely does not include `"summarize"` or `"all"` — or the comparison is against the un-lowercased token. Words like `Summarize`, `all`, `files` have no semantic value as graph entities but they pollute the Neo4j keyword search.

**Impact:**  
Graph concept discovery queries Neo4j for `"Summarize"`, `"all"`, `"knowledge"`, `"base"` — all producing 0 results while consuming query budget. This explains the `concepts=0, relations=0` pattern seen in Nov 2 logs for broad queries.

**Recommended fix:**  
Expand the stop word list to include common RAG meta-words: `{"summarize", "summary", "overview", "explain", "describe", "tell", "give", "show", "list", "all", "every", "across", "throughout", "papers", "documents", "files", "knowledge", "base", "corpus"}`. Apply comparison in lowercase consistently.

---

## 3. Part 2 — Retrieval Limit Findings

### Finding R-1 — SUMMARIZATION K=9 / pool=25 is insufficient for a 100-paper corpus

**Severity: HIGH**

**File:** `query_classifier.py` — `SUMMARIZATION` branch parameter allocation

**Current values (confirmed from code):**
```python
recommended_k = 9
candidate_pool_size = 25
graph_limit = 18
traversal_depth = 2
```

**Why this is insufficient:**  
- 9 chunks from a 100-paper corpus = **~9% coverage**.
- A query like `"Summarize the RAG types across all 100 research papers"` needs cross-corpus aggregation.
- `candidate_pool_size=25` means MMR can only select from 25 candidates before diversity filtering — likely from the same ~5–8 papers (the most semantically similar ones).
- The 91 other papers are completely invisible to the retriever.

**What the log shows:**  
Nov 2 session: SUMMARIZATION queries consistently returned `concepts=0, relations=0` from graph, so the entire answer load fell on the 9 vector chunks — which happened to be from similar papers.

**Recommended values:**

| Parameter | Current | Recommended | Justification |
|---|---|---|---|
| `recommended_k` | 9 | **15** | Cover at least 15% of a 100-paper KB |
| `candidate_pool_size` | 25 | **45** | `K * 3` ratio; gives MMR room to diversify |
| `graph_limit` | 18 | **25** | More relationships from graph to supplement |
| `traversal_depth` | 2 | **2** | Keep at 2 — depth 3 risks too much noise |

---

### Finding R-2 — Graph keyword cap `[:4]` limits concept discovery scope

**Severity: MEDIUM**

**File:** `concept_discovery.py` — approximately line 82  
**Code:** `for kw in keywords[:4]:`

**Root cause:**  
The `discover()` method receives `keywords` from entity extraction but only queries Neo4j for the **first 4**. For SUMMARIZATION queries with 6+ valid entities, the extra entities are silently dropped.

**Example:**  
Query: `"Compare BERT, GPT, T5, RoBERTa, and ALBERT for retrieval tasks"`  
Extracted entities: `['BERT', 'GPT', 'T5', 'RoBERTa', 'ALBERT', 'retrieval']`  
Queried in Neo4j: only `['BERT', 'GPT', 'T5', 'RoBERTa']` → `ALBERT` and `retrieval` never discovered.

**Recommended fix:**  
Change `[:4]` to `[:6]` uniformly, or make the cap intent-aware:
```python
kw_limit = 6 if intent in ("SUMMARIZATION", "COMPARATIVE") else 4
for kw in keywords[:kw_limit]:
```

---

### Finding R-3 — Hub limit hardcoded at 5, independent of `graph_limit`

**Severity: MEDIUM**

**File:** `concept_discovery.py` — approximately line 98  
**Code:** `hubs = self._find_thematic_hubs(session, keywords, limit=5)`

**Root cause:**  
The caller passes `graph_limit=18` for SUMMARIZATION, but `_find_thematic_hubs()` ignores this and always fetches only 5 hub concepts. For a 100-paper corpus, 5 hub concepts represent only the most globally central nodes — missing mid-tier themes.

**Recommended fix:**  
Make the hub limit scale with `graph_limit`:
```python
hub_limit = max(5, graph_limit // 3)  # e.g., 18//3 = 6 for SUMMARIZATION
hubs = self._find_thematic_hubs(session, keywords, limit=hub_limit)
```

---

### Finding R-4 — `relationships[:limit]` conflates per-keyword and total cap

**Severity: MEDIUM**

**File:** `concept_discovery.py` — approximately line 114  
**Code:**
```python
# Inside per-keyword loop:
results = session.run(cypher, keyword=kw, limit=limit)  # limit=18

# After the loop:
relationships = relationships[:limit]  # same limit=18 used as total cap
```

**Root cause:**  
The variable `limit` (which comes from `graph_limit=18`) is used both as the **per-keyword Cypher LIMIT** and as the **final total slice**. This means:
- 4 keywords × up to 18 results each = up to 72 raw relationships collected.
- Then `[:18]` discards 54 of them (the last 3 keywords' results are mostly dropped).

**Practical effect:**  
The first keyword's relationships dominate the final output. Later keywords' (potentially more relevant for diverse queries) results are truncated.

**Recommended fix:**  
Introduce a separate `per_keyword_limit` (e.g., `max(5, graph_limit // kw_count)`) and keep `graph_limit` as only the final cap:
```python
per_kw_limit = max(5, graph_limit // max(1, len(keywords[:kw_cap])))
results = session.run(cypher, keyword=kw, limit=per_kw_limit)
# ...
relationships = relationships[:graph_limit]  # final cap
```

---

### Finding R-5 — EXPLORATORY_THEMATIC K=5–6 is too low for broad thematic questions

**Severity: MEDIUM**

**File:** `query_classifier.py` — `EXPLORATORY_THEMATIC` branch  
**Code (approximately):**
```python
rec_k = 6 if len(extracted_entities) >= 2 else 5
cand_size = rec_k * 3  # → 15 or 18
```

**Root cause:**  
The K allocation is purely entity-count-driven (6 if ≥2 entities, else 5). But thematic exploration of a large corpus needs more coverage — 6 chunks from a 100-paper KB is only 6% coverage. Broad thematic questions like `"What approaches do the papers use for retrieval augmentation?"` genuinely benefit from more evidence.

**Recommended values:**

| Parameter | Current | Recommended |
|---|---|---|
| `rec_k` (base) | 5 | **8** |
| `rec_k` (entity-boosted) | 6 | **10** |
| `cand_size` | `rec_k * 3` | `rec_k * 3` (keep ratio) → **24 or 30** |

---

### Finding R-6 — COMPARATIVE K allocation not boosted for multi-entity comparisons

**Severity: LOW**

**File:** `query_classifier.py` — `COMPARATIVE` branch

**Current behavior (from code):**  
COMPARATIVE gets a fixed `recommended_k` without per-comparison-target boosting. A query comparing 2 methods needs at least 2–3 chunks per method; for 4-way comparisons (e.g., `"Compare BM25, DPR, ColBERT, and SPLADE"`) the current K may produce only 1–2 chunks per method.

**Recommended fix:**  
Count comparison targets from entity extraction and boost:
```python
comparison_count = max(2, len(extracted_entities))
recommended_k = min(16, comparison_count * 3)  # 3 chunks per comparison target, capped at 16
```

---

## 4. Summary Table

### Classification Findings

| # | Finding | Severity | File | Evidence |
|---|---|---|---|---|
| C-1 | `"what are"` missing from FACTUAL_PATTERNS | **HIGH** | `query_classifier.py` | `app.log` line 64–70 |
| C-2 | Word-count gate `<= 10` too strict | **MEDIUM** | `query_classifier.py` line ~157 | Code analysis |
| C-3 | `"types/kinds/categories of"` not covered | **MEDIUM** | `query_classifier.py` | Code analysis |
| C-4 | SUMMARIZATION missing natural-language phrasings | **MEDIUM** | `query_classifier.py` | Code analysis |
| C-5 | RELATIONAL missing `"connection between"`, `"relate to"` | **MEDIUM** | `query_classifier.py` | Code analysis |
| C-6 | CONVERSATIONAL anchors fail on punctuation | **LOW** | `query_classifier.py` | Code analysis |
| C-7 | Entity extraction returns stop words (capitalized) | **LOW** | `query_classifier.py` | Test output |

### Retrieval Limit Findings

| # | Finding | Severity | File | Evidence |
|---|---|---|---|---|
| R-1 | SUMMARIZATION K=9/pool=25 insufficient for 100 papers | **HIGH** | `query_classifier.py` | `app.log` + math |
| R-2 | Graph keyword cap `[:4]` drops entities silently | **MEDIUM** | `concept_discovery.py` line ~82 | Code analysis |
| R-3 | Hub limit hardcoded at 5, ignores `graph_limit` | **MEDIUM** | `concept_discovery.py` line ~98 | Code analysis |
| R-4 | `relationships[:limit]` conflates per-keyword and total cap | **MEDIUM** | `concept_discovery.py` line ~114 | Code analysis |
| R-5 | EXPLORATORY K=5–6 too low for broad thematic questions | **MEDIUM** | `query_classifier.py` | Code analysis |
| R-6 | COMPARATIVE K not boosted per comparison target | **LOW** | `query_classifier.py` | Code analysis |

---

## 5. Recommended Changes (Pending Approval)

> [!CAUTION]
> None of the following changes have been applied. Approval required before proceeding.

### Group A — `query_classifier.py` changes

```
A1. Add "what\s+are" to FACTUAL_PATTERNS          [C-1, HIGH]
A2. Raise word-count gate from <= 10 to <= 15      [C-2, MEDIUM]
A3. Add "types of / kinds of / categories of"     [C-3, MEDIUM]
A4. Add 5 new SUMMARIZATION_PATTERNS              [C-4, MEDIUM]
A5. Add 4 new RELATIONAL_PATTERNS                 [C-5, MEDIUM]
A6. Strip punctuation before CONVERSATIONAL match  [C-6, LOW]
A7. Expand entity extraction stop word list        [C-7, LOW]
A8. Raise SUMMARIZATION K: 9→15, pool: 25→45      [R-1, HIGH]
A9. Raise EXPLORATORY K: 5→8 / 6→10, pool scales  [R-5, MEDIUM]
A10. Add COMPARATIVE per-target K boosting         [R-6, LOW]
```

### Group B — `concept_discovery.py` changes

```
B1. Change keyword cap [:4] → [:6] (or intent-aware)         [R-2, MEDIUM]
B2. Make hub limit scale with graph_limit (÷3 formula)        [R-3, MEDIUM]
B3. Separate per-keyword Cypher limit from final total cap    [R-4, MEDIUM]
```

### Suggested approval approach

You may approve:
- **All changes at once** — I'll apply them in sequence and run the test suite.
- **Group A only** — classifier changes first, concept discovery later.
- **HIGH severity only first** (A1, A8) — minimal risk, highest gain.
- **Each finding individually** — most conservative option.

After any approval, I will:
1. Apply the specified changes.
2. Run `uv run python scripts/test_adaptive_rag.py` and `uv run python scripts/test_query_engine.py`.
3. Report results before applying the next group.

