"""
Vector Store module using ChromaDB and configurable cloud embeddings.

Persists chunks and semantic vectors into local storage.
Provides cosine similarity search and document collection management.
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from graph_rag.config import settings
from graph_rag.ingestion.chunker import TextChunk
from graph_rag.llm.model_client import embedding_model_client

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages persistent ChromaDB vector storage and semantic retrieval."""

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self.persist_dir = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": settings.EMBEDDING_MODEL or "not-configured",
                "schema_version": settings.GRAPH_SCHEMA_VERSION,
            },
        )
        self.langchain_store = Chroma(
            client=self.client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model_client.as_embeddings(),
        )
        self._entity_collection = None

    @property
    def entity_collection(self):
        """Create the canonical-entity index lazily, separate from document chunks."""
        if self._entity_collection is None:
            self._entity_collection = self.client.get_or_create_collection(
                name=f"{settings.CHROMA_COLLECTION_NAME}_entities",
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": settings.EMBEDDING_MODEL or "not-configured",
                    "schema_version": settings.GRAPH_SCHEMA_VERSION,
                    "index_type": "canonical_entities",
                },
            )
        return self._entity_collection

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """
        Compute embeddings for chunks and store them in ChromaDB.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[Document] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            # Ensure metadata values are strings, ints, floats, or bools for ChromaDB
            clean_meta = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_meta["document_name"] = chunk.document_name
            clean_meta["document_title"] = chunk.document_title
            clean_meta["page"] = chunk.page_number
            clean_meta["chunk_index"] = chunk.chunk_index
            clean_meta["chunk_id"] = chunk.chunk_id
            clean_meta["embedding_model"] = settings.EMBEDDING_MODEL
            clean_meta["schema_version"] = settings.GRAPH_SCHEMA_VERSION
            documents.append(
                Document(page_content=chunk.text, metadata=clean_meta)
            )

        self.langchain_store.add_documents(
            documents=documents,
            ids=ids,
        )
        return len(chunks)

    def replace_document_chunks(
        self,
        document_name: str,
        chunks: List[TextChunk],
    ) -> int:
        """Upsert current chunks before removing stale chunks for a document."""
        if any(chunk.document_name != document_name for chunk in chunks):
            raise ValueError("All replacement chunks must belong to the document")

        self.add_chunks(chunks)
        desired_ids = {chunk.chunk_id for chunk in chunks}
        existing = self.collection.get(
            where={"document_name": document_name},
            include=[],
        )
        stale_ids = [
            chunk_id
            for chunk_id in existing.get("ids") or []
            if chunk_id not in desired_ids
        ]
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search for a query text.
        Returns a ranked list of matched chunks with scores and metadata.
        """
        formatted_results: List[Dict[str, Any]] = []
        results = self.langchain_store.similarity_search_with_score(
            query,
            k=n_results,
            filter=filter_metadata,
        )
        for document, raw_distance in results:
            distance = float(raw_distance)
            similarity_score = max(0.0, 1.0 - distance)
            metadata = document.metadata
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                document_name = str(metadata.get("document_name", "document"))
                chunk_id = (
                    f"{Path(document_name).stem}_p{metadata.get('page', 1)}"
                    f"_c{metadata.get('chunk_index', 0)}"
                )
            formatted_results.append(
                {
                    "chunk_id": chunk_id,
                    "text": document.page_content,
                    "metadata": metadata,
                    "similarity_score": round(similarity_score, 4),
                    "distance": round(distance, 4),
                }
            )

        return formatted_results

    def get_total_chunks(self) -> int:
        """Return total number of chunks currently stored in ChromaDB."""
        return self.collection.count()

    def get_document_chunk_count(self, document_name: str) -> int:
        """Return stored chunk count for one exact document filename."""
        rows = self.collection.get(
            where={"document_name": document_name},
            include=[],
        )
        return len(rows.get("ids") or [])

    def delete_document_chunks(self, document_name: str) -> None:
        """Delete all chunks belonging to a specific document."""
        self.collection.delete(where={"document_name": document_name})

    def upsert_entity_vectors(
        self,
        entities: List[Any],
        document_id: str,
    ) -> int:
        """Index canonical entity names and aliases for automatic resolution."""
        if not entities:
            return 0
        texts: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        metadata_updates: List[tuple[str, Dict[str, Any]]] = []
        for entity in entities:
            entity_id = str(entity.entity_id or "").strip()
            if not entity_id:
                continue
            text = "\n".join(
                (
                    str(entity.name),
                    str(entity.category),
                    str(entity.description or ""),
                    " ".join(entity.aliases or []),
                    json.dumps(entity.attributes or {}, ensure_ascii=False),
                )
            )
            existing = self.entity_collection.get(
                ids=[entity_id],
                include=["metadatas", "documents"],
            )
            prior = (existing.get("metadatas") or [{}])[0] or {}
            try:
                source_ids = json.loads(prior.get("source_document_ids") or "[]")
            except json.JSONDecodeError:
                source_ids = []
            if not isinstance(source_ids, list):
                source_ids = []
            source_ids = sorted(set(source_ids + [document_id]))
            metadata = {
                "entity_id": entity_id,
                "name": str(entity.name),
                "category": str(entity.category),
                "description": str(entity.description or ""),
                "aliases_json": json.dumps(entity.aliases or [], ensure_ascii=False),
                "attributes_json": json.dumps(
                    entity.attributes or {},
                    ensure_ascii=False,
                    default=str,
                ),
                "source_document_ids": json.dumps(source_ids),
                "embedding_model": settings.EMBEDDING_MODEL,
            }
            prior_document = (existing.get("documents") or [None])[0]
            if prior and prior_document == text:
                if prior != metadata:
                    metadata_updates.append((entity_id, metadata))
                continue
            texts.append(text)
            ids.append(entity_id)
            metadatas.append(metadata)
        for entity_id, metadata in metadata_updates:
            self.entity_collection.update(ids=[entity_id], metadatas=[metadata])
        if not ids:
            return 0
        embeddings = embedding_model_client.get_embeddings_batch(texts)
        if len(embeddings) != len(ids):
            raise RuntimeError("Entity embedding response count did not match request count")
        self.entity_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(ids)

    def search_entity_vectors(
        self,
        query: str,
        *,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return semantic entity candidates with cosine similarity scores."""
        return self.search_entity_vectors_batch([query], limit=limit)[0]

    def search_entity_vectors_batch(
        self,
        queries: List[str],
        *,
        limit: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """Embed and retrieve multiple entity candidates in one tracked request."""
        if not queries or self.entity_collection.count() == 0:
            return [[] for _ in queries]
        active_queries = [
            (index, query)
            for index, query in enumerate(queries)
            if query.strip()
        ]
        if not active_queries:
            return [[] for _ in queries]
        embeddings = embedding_model_client.get_embeddings_batch(
            [query for _, query in active_queries]
        )
        if len(embeddings) != len(active_queries):
            raise RuntimeError("Entity query embedding count did not match request count")
        response = self.entity_collection.query(
            query_embeddings=embeddings,
            n_results=limit,
            include=["metadatas", "distances", "documents"],
        )
        all_ids = response.get("ids") or []
        all_metadatas = response.get("metadatas") or []
        all_distances = response.get("distances") or []
        all_documents = response.get("documents") or []
        results: List[List[Dict[str, Any]]] = [[] for _ in queries]
        for result_index, (query_index, _) in enumerate(active_queries):
            ids = all_ids[result_index] if result_index < len(all_ids) else []
            metadatas = all_metadatas[result_index] if result_index < len(all_metadatas) else []
            distances = all_distances[result_index] if result_index < len(all_distances) else []
            documents = all_documents[result_index] if result_index < len(all_documents) else []
            results[query_index] = [
                {
                    **(metadata or {}),
                    "entity_id": ids[index],
                    "document": documents[index] if index < len(documents) else "",
                    "similarity_score": max(0.0, 1.0 - float(distances[index])),
                }
                for index, metadata in enumerate(metadatas)
                if index < len(ids) and index < len(distances)
            ]
        return results

    def delete_document_entity_vectors(self, document_id: str) -> None:
        """Remove one document's ownership from entity vectors, deleting orphans."""
        rows = self.entity_collection.get(include=["metadatas"])
        remove_ids: List[str] = []
        update_rows: List[tuple[str, Dict[str, Any]]] = []
        for entity_id, metadata in zip(
            rows.get("ids") or [],
            rows.get("metadatas") or [],
        ):
            try:
                source_ids = json.loads((metadata or {}).get("source_document_ids") or "[]")
            except json.JSONDecodeError:
                source_ids = []
            if not isinstance(source_ids, list):
                source_ids = []
            remaining = [value for value in source_ids if value != document_id]
            if not remaining:
                remove_ids.append(entity_id)
            elif remaining != source_ids:
                updated = dict(metadata or {})
                updated["source_document_ids"] = json.dumps(remaining)
                update_rows.append((entity_id, updated))
        if remove_ids:
            self.entity_collection.delete(ids=remove_ids)
        for entity_id, metadata in update_rows:
            self.entity_collection.update(ids=[entity_id], metadatas=[metadata])


vector_store = VectorStore()
