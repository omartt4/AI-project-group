# Graph Knowledge Building — Schema v2

The redesigned ingestion pipeline builds an evidence-first graph for the
current corpus. It does not reuse the old Neo4j graph or old embedding vectors.

## Graph hierarchy

```text
(TopLevelTopic)-[:HAS_SUBTOPIC]->(Subtopic)
(TopLevelTopic|Subtopic)-[:HAS_PAPER]->(Paper)-[:HAS_CHUNK]->(Chunk)
(Chunk)-[:CONTAINS_MENTION]->(Mention)-[:REFERS_TO]->(Entity)
```

`Paper` is the document identity. `Chunk` provides retrieval granularity.
`Mention` is the exact text phrase, while `Entity` is the canonical object that
one or more mentions refer to.

For example, "Ahmed", "he", and "the director" can be three mentions that all
refer to one `Person` entity. Ambiguous mentions remain unresolved rather than
creating a false person.

## Semantic relations

Allowed relations are controlled in application code:

- `AUTHORED`
- `AFFILIATED_WITH`
- `HOLDS_ROLE`
- `PROPOSES`
- `EVALUATES_ON`
- `USES_METHOD`
- `PART_OF_TOPIC`
- `REPORTS`
- `RELATED_TO`

Every relation stores the source `paper_id`, `chunk_id`, evidence text,
confidence, dataset ID, and extraction version. Model-generated relationship
types outside the whitelist are rejected before a Cypher query is built.

## Entity resolution

Chunks are processed in paper order. Before extracting a later chunk, the model
receives a compact registry of entities already found in that paper. It must
reuse an existing entity ID for pronouns, aliases, and role descriptions when
the context is clear.

Methods, datasets, topics, metrics, and tasks use stable normalized identities.
Potentially ambiguous people, organizations, and findings remain paper-scoped
until stronger cross-paper evidence is available. This avoids merging two
people merely because they share a name.

## Topic taxonomy

The initial active topics are:

- Machine Learning
- Computer Vision
- Natural Language Processing

Known aliases such as `ML`, `CV`, and `NLP` map deterministically to their
canonical `TopLevelTopic`. Specialist areas become `Subtopic` nodes connected
to their parent by `HAS_SUBTOPIC`. A genuinely new domain is created as a
candidate top-level topic and can later be reviewed and promoted. Papers and
subtopics may connect to multiple top-level topics.

## Cloud models and vector versioning

Generation and embeddings use configurable OpenRouter endpoints. Credentials
and exact model IDs live only in `.env`.

Changing the embedding model or its dimensions requires a new ChromaDB
collection. Schema v2 defaults to `research_papers_graph_v2`; vectors from the
previous local model are never mixed with the new cloud vectors.

## Fresh-corpus workflow

1. Configure the new Neo4j instance in `.env`.
2. Place only the new corpus in `pdfs/`.
3. Start with a fresh `metadata.csv` for that corpus.
4. Configure chat and embedding model IDs.
5. Run ingestion on three to five papers first.
6. Inspect topic assignments, unresolved mentions, duplicate entities, and
   evidence quality before ingesting the full corpus.

See `GRAPH_REDESIGN_PLAN.md` for the rollout and design decisions.
