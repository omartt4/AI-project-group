import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
SNIPPETS_SOURCE = EVAL_DIR / "cleaned_snippets.txt"
SNIPPETS_BACKUP = EVAL_DIR / "cleaned_snippets_legacy.old"
SNIPPETS_JSONL = EVAL_DIR / "cleaned_snippets.jsonl"
REPORT_SOURCE = EVAL_DIR / "inspect_report.txt.old"
REPORT_BACKUP = EVAL_DIR / "inspect_report_backup_oversized.old"


def main() -> None:
    if not SNIPPETS_SOURCE.is_file():
        raise FileNotFoundError(f"Snippet source not found: {SNIPPETS_SOURCE}")
    for path in (SNIPPETS_BACKUP, SNIPPETS_JSONL):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    if REPORT_SOURCE.exists() and REPORT_BACKUP.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {REPORT_BACKUP}")

    content = SNIPPETS_SOURCE.read_text(encoding="utf-8")
    matches = re.findall(
        r"Snippet (\d+) \(book_id=(\d+)\):\n(.*?)(?=\n\nSnippet \d+ "
        r"\(book_id=\d+\):\n|\Z)",
        content,
        flags=re.DOTALL,
    )
    if len(matches) != 3:
        raise ValueError(f"Expected 3 snippets, found {len(matches)}.")

    records = [
        {
            "snippet_id": f"snippet_{index}",
            "book_id": book_id,
            "text": text.strip(),
        }
        for index, book_id, text in matches
    ]
    SNIPPETS_SOURCE.rename(SNIPPETS_BACKUP)
    if REPORT_SOURCE.exists():
        REPORT_SOURCE.rename(REPORT_BACKUP)
    with SNIPPETS_JSONL.open("x", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"converted_snippets={len(records)}")
    print(f"snippets_jsonl={SNIPPETS_JSONL}")
    if REPORT_BACKUP.exists():
        print(f"preserved_report_backup={REPORT_BACKUP}")


if __name__ == "__main__":
    main()
