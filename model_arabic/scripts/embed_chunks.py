import json
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "chunks" / "chunks.jsonl"
CHROMA_PATH = ROOT / "chroma"
MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "arabic_e5small_384_256_vector"
EMBEDDING_DIMENSION = 384
MAX_CHUNK_TOKENS = 480


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")
    with CHUNKS_PATH.open("r", encoding="utf-8") as source:
        chunks = [json.loads(line) for line in source if line.strip()]
    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}")
    return chunks


def main() -> None:
    started_at = time.perf_counter()
    chunks = load_chunks()
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Chunk IDs are not unique.")
    if any(chunk["token_count"] > MAX_CHUNK_TOKENS for chunk in chunks):
        raise ValueError(f"A chunk exceeds the {MAX_CHUNK_TOKENS}-token guard.")

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    existing_names = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME in existing_names:
        raise FileExistsError(
            f"Refusing to overwrite existing Chroma collection: {COLLECTION_NAME}"
        )

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    dimension = model.get_embedding_dimension()
    if dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION} dimensions, got {dimension}."
        )

    passages = [f"passage: {chunk['text']}" for chunk in chunks]
    tokenizer = model.tokenizer
    content_token_total = 0
    passage_token_total = 0
    for chunk, passage in zip(chunks, passages):
        content_count = len(
            tokenizer(chunk["text"], add_special_tokens=False)["input_ids"]
        )
        if content_count != chunk["token_count"]:
            raise ValueError(
                f"Stored token count mismatch for chunk {chunk['chunk_id']}: "
                f"{chunk['token_count']} != {content_count}"
            )
        content_token_total += content_count
        passage_token_total += len(
            tokenizer(passage, add_special_tokens=False)["input_ids"]
        )

    embeddings = model.encode(
        passages,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if embeddings.shape != (len(chunks), EMBEDDING_DIMENSION):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("Embedding vectors are not normalized.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "book_id": chunk["book_id"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "token_count": chunk["token_count"],
            }
            for chunk in chunks
        ],
    )

    stored_count = collection.count()
    if stored_count != len(chunks):
        raise ValueError(f"Chroma count {stored_count} != chunks count {len(chunks)}")
    stored = collection.get(ids=ids, include=["documents"])
    stored_documents = dict(zip(stored["ids"], stored["documents"]))
    expected_documents = {
        chunk["chunk_id"]: chunk["text"]
        for chunk in chunks
    }
    if stored_documents != expected_documents:
        raise ValueError("Stored Chroma documents do not match chunks.jsonl.")

    elapsed_seconds = time.perf_counter() - started_at
    print(f"collection={COLLECTION_NAME}")
    print(f"chunks={len(chunks)}")
    print(f"dimensions={dimension}")
    print(f"content_tokens={content_token_total}")
    print(f"tokens_with_passage_prefix={passage_token_total}")
    print(f"elapsed_seconds={elapsed_seconds:.1f}")
    print("cost_usd=0 (local)")
    print(f"chroma_path={CHROMA_PATH}")


if __name__ == "__main__":
    main()
