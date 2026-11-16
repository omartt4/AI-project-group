# Graph Redesign Plan

This branch replaces the implicit, name-only graph with an evidence-first graph.
The redesign has four goals:

1. Make topics first-class nodes with an extensible taxonomy.
2. Separate real entities from the text mentions that refer to them.
3. Keep the paper, chunk, and evidence behind every extracted fact.
4. Run generation and embeddings through configurable cloud providers.

## Target schema

```text
(Topic)-[:HAS_PAPER]->(Paper)-[:HAS_CHUNK]->(Chunk)
(Chunk)-[:CONTAINS_MENTION]->(Mention)-[:REFERS_TO]->(Entity)
(Entity)-[:USES_METHOD|EVALUATES_ON|AFFILIATED_WITH|HOLDS_ROLE|...]->(Entity)
```

Semantic relationships store `paper_id`, `chunk_id`, `evidence`, `confidence`,
and `extraction_version`. A relationship is rejected if its type is not in the
application whitelist or if either endpoint cannot be resolved.

## Mentions and entity resolution

An `Entity` is the canonical graph object. A `Mention` is a phrase in a source
chunk such as a name, pronoun, or role description. Multiple mentions can refer
to one entity. The extractor receives the paper's already-known entities so it
can reuse their IDs for later references such as "the director" or "he".

Ambiguous mentions stay unresolved. The pipeline never creates a new person
solely because it encountered a role phrase.

## Topic taxonomy

Machine Learning, Computer Vision, and Natural Language Processing are seed
topics, not a closed list. Known aliases map deterministically to the seeds.
Unknown topics are stored as `candidate` topics with their source and confidence
so they can be reviewed and promoted without changing code.

Papers may belong to more than one topic.

## Model providers

Extraction, answer generation, evaluation, and embeddings use separate
OpenRouter-compatible clients. Model IDs, dimensions, timeouts, and API
credentials are configuration, not source code. A new ChromaDB collection is
used because vectors from different models or dimensions must never be mixed.

## Rollout

1. Run the redesign on three to five representative papers.
2. Inspect topic assignments, unresolved mentions, duplicate entities, and
   relationship evidence.
3. Approve the graph quality before processing the full corpus.
4. Keep the old graph untouched until the new graph passes validation.

