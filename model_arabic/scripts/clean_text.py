import random
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
CLEANED_DIR = ROOT / "cleaned"
SNIPPETS_PATH = ROOT / "eval" / "cleaned_snippets.txt"
BOOK_FILES = (
    ("book_140-train-00000-of-00001.parquet", "140"),
    ("book_1144-train-00000-of-00001.parquet", "1144"),
)
SNIPPET_LENGTH = 250


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    without_controls = "".join(
        " " if unicodedata.category(character) == "Cc" else character
        for character in normalized
    )
    return re.sub(r"\s+", " ", without_controls).strip()


def make_snippet(text: str, rng: random.Random) -> str:
    if len(text) <= SNIPPET_LENGTH:
        return text
    start = rng.randint(0, len(text) - SNIPPET_LENGTH)
    if start > 0:
        next_space = text.find(" ", start)
        if next_space != -1 and next_space < start + 80:
            start = next_space + 1
    return text[start : start + SNIPPET_LENGTH].strip()


def main() -> None:
    outputs = [
        CLEANED_DIR / f"book_{book_id}.txt"
        for _, book_id in BOOK_FILES
    ]
    if SNIPPETS_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {SNIPPETS_PATH}")
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")

    texts = []
    for (filename, book_id), output in zip(BOOK_FILES, outputs):
        source = RAW_DIR / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required parquet file not found: {source}")
        frame = pd.read_parquet(source, columns=["body"])
        pages = [value if isinstance(value, str) else "" for value in frame["body"]]
        text = clean_text("\n\n".join(pages))
        output.write_text(text, encoding="utf-8")
        texts.append((book_id, text))
        print(f"book_id={book_id} characters={len(text)}")

    rng = random.Random(2026)
    snippet_sources = [texts[0], texts[1], rng.choice(texts)]
    snippets = [
        (book_id, make_snippet(text, rng))
        for book_id, text in snippet_sources
    ]
    snippet_lines = [
        f"Snippet {index} (book_id={book_id}):\n{text}"
        for index, (book_id, text) in enumerate(snippets, start=1)
    ]
    SNIPPETS_PATH.write_text("\n\n".join(snippet_lines) + "\n", encoding="utf-8")
    print(f"snippets_file={SNIPPETS_PATH}")


if __name__ == "__main__":
    main()
