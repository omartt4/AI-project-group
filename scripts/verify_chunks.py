import argparse
import hashlib
import json
from pathlib import Path

from lab_chunk import PDFParser, clean_text, detect_and_remove_headers_footers


def save_cleaned(paper_ids):
    cleaned_dir = Path("data/lab/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    parser = PDFParser("pymupdf")
    manifest = {}

    for paper_id in paper_ids:
        matches = list(Path("pdfs").glob(f"{paper_id}_*.pdf"))
        if len(matches) != 1:
            raise ValueError(f"Expected one PDF for {paper_id}, found {len(matches)}")
        pages = parser.extract_pages(matches[0])
        pages = detect_and_remove_headers_footers(pages)
        full_text = ""
        for page in pages:
            full_text += clean_text(page["text"]) + "\n\n"
        cleaned_path = cleaned_dir / f"{matches[0].stem}.txt"
        if not cleaned_path.exists():
            cleaned_path.write_text(full_text, encoding="utf-8")
        text = cleaned_path.read_text(encoding="utf-8")
        manifest[matches[0].stem] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "characters": len(text)
        }

    manifest_path = cleaned_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def verify_chunks(chunks_path, paper_ids):
    cleaned_dir = Path("data/lab/cleaned")
    rows = [json.loads(line) for line in Path(chunks_path).open("r", encoding="utf-8")]
    valid = 0
    for row in rows:
        cleaned_path = cleaned_dir / f"{row['paper_id']}.txt"
        if not cleaned_path.exists():
            raise ValueError(f"Missing cleaned text: {cleaned_path}")
        cleaned = cleaned_path.read_text(encoding="utf-8")
        if cleaned[row["char_start"]:row["char_end"]] == row["text"]:
            valid += 1
    print(f"{Path(chunks_path).name}: {valid}/{len(rows)}")
    return valid == len(rows)


def repair_spans(chunks_path):
    cleaned_dir = Path("data/lab/cleaned")
    path = Path(chunks_path)
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8")]
    for paper_id in sorted(set(row["paper_id"] for row in rows)):
        cleaned = (cleaned_dir / f"{paper_id}.txt").read_text(encoding="utf-8")
        paper_rows = [row for row in rows if row["paper_id"] == paper_id]
        search_start = 0
        for row in paper_rows:
            char_start = cleaned.find(row["text"], search_start)
            if char_start < 0:
                char_start = cleaned.find(row["text"])
            if char_start < 0:
                raise ValueError(f"Chunk text not found: {row['chunk_id']}")
            row["char_start"] = char_start
            row["char_end"] = char_start + len(row["text"])
            search_start = char_start + 1
    temp_path = path.with_suffix(".repair.jsonl")
    with temp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return temp_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-cleaned", action="store_true")
    parser.add_argument("--chunks", action="append")
    parser.add_argument("--repair-spans", action="store_true")
    args = parser.parse_args()
    paper_ids = ["001", "002", "003", "035", "041", "045", "072", "074", "090"]
    if args.save_cleaned:
        save_cleaned(paper_ids)
    if not args.chunks:
        args.chunks = [
            "data/lab/chunks_256_26.jsonl",
            "data/lab/chunks_384_38.jsonl",
        ]
    if args.repair_spans:
        for path in args.chunks:
            repaired = repair_spans(path)
            print(f"Repaired temporary file: {repaired}")
    passed = all(verify_chunks(path, paper_ids) for path in args.chunks)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
