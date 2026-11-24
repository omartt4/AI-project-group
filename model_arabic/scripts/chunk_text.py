import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = ROOT / "cleaned"
CHUNKS_PATH = ROOT / "chunks" / "chunks.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 256
OVERLAP_TOKENS = 26
MAX_CHUNK_TOKENS = 480
BOOK_IDS = ("140", "1144")


def make_book_chunks(book_id: str, tokenizer) -> list[dict]:
    text_path = CLEANED_DIR / f"book_{book_id}.txt"
    if not text_path.is_file():
        raise FileNotFoundError(f"Cleaned text not found: {text_path}")
    text = text_path.read_text(encoding="utf-8")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoded["offset_mapping"]
    if any(start >= end for start, end in offsets):
        raise ValueError(f"Tokenizer returned a zero-width token offset for book {book_id}")

    chunks = []
    start_token = 0
    while start_token < len(offsets):
        end_token = min(start_token + CHUNK_SIZE, len(offsets))
        while True:
            char_start = offsets[start_token][0]
            char_end = offsets[end_token - 1][1]
            chunk_text = text[char_start:char_end]
            token_count = len(
                tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
            )
            if token_count <= CHUNK_SIZE:
                break
            end_token -= 1
            if end_token <= start_token:
                raise ValueError(f"Unable to make a bounded chunk for book {book_id}")

        assert chunk_text == text[char_start:char_end]
        assert token_count <= MAX_CHUNK_TOKENS
        chunks.append(
            {
                "chunk_id": f"{book_id}_{len(chunks):05d}",
                "book_id": book_id,
                "char_start": char_start,
                "char_end": char_end,
                "text": chunk_text,
                "token_count": token_count,
            }
        )
        if end_token == len(offsets):
            break
        start_token = max(start_token + 1, end_token - OVERLAP_TOKENS)
    return chunks


def main() -> None:
    if CHUNKS_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {CHUNKS_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("The selected tokenizer must support character offsets.")

    all_chunks = []
    for book_id in BOOK_IDS:
        book_chunks = make_book_chunks(book_id, tokenizer)
        counts = [chunk["token_count"] for chunk in book_chunks]
        print(
            f"book_id={book_id} chunks={len(book_chunks)} "
            f"mean_tokens={statistics.mean(counts):.1f} max_tokens={max(counts)}"
        )
        all_chunks.extend(book_chunks)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in all_chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"total_chunks={len(all_chunks)}")
    print(f"chunks_file={CHUNKS_PATH}")


if __name__ == "__main__":
    main()
