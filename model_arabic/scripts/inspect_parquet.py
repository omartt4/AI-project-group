import json
import statistics
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
REPORT_PATH = ROOT / "eval" / "inspect_report.txt"
BOOK_FILES = (
    "book_140-train-00000-of-00001.parquet",
    "book_1144-train-00000-of-00001.parquet",
)


def safe_sample_value(value):
    if isinstance(value, bytes):
        return f"<binary data: {len(value)} bytes>"
    if isinstance(value, str):
        return value[:1000] + "...[truncated]" if len(value) > 1000 else value
    if isinstance(value, dict):
        return {str(key): safe_sample_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_sample_value(item) for item in value[:20]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}: {str(value)[:300]}>"


def describe_book(path: Path) -> list[str]:
    frame = pd.read_parquet(path)
    lines = [
        f"FILE: {path.name}",
        f"File size: {path.stat().st_size} bytes",
        f"Rows: {len(frame)}",
        f"Columns: {len(frame.columns)}",
        "Column names and dtypes:",
    ]
    for column, dtype in frame.dtypes.items():
        lines.append(f"  {column}: {dtype}")

    text_stats = []
    for column in frame.columns:
        values = frame[column].dropna()
        string_values = values[values.map(lambda value: isinstance(value, str))]
        if string_values.empty:
            continue
        lengths = string_values.map(len).tolist()
        text_stats.append((column, lengths))

    lines.append("String-column length statistics (characters per non-null string):")
    for column, lengths in text_stats:
        lines.append(
            f"  {column}: strings={len(lengths)}, min={min(lengths)}, "
            f"median={statistics.median(lengths)}, mean={statistics.mean(lengths):.1f}, "
            f"max={max(lengths)}"
        )
    if text_stats:
        likely_column, _ = max(
            text_stats,
            key=lambda item: statistics.median(item[1]),
        )
        lines.append(
            f"Provisional longest-text candidate (verify from samples): {likely_column}"
        )

    lines.append("First 3 rows (string values truncated to 1000 characters):")
    sample = [
        {
            column: safe_sample_value(value)
            for column, value in row.items()
        }
        for row in frame.head(3).to_dict(orient="records")
    ]
    lines.append(
        json.dumps(
            sample,
            ensure_ascii=False,
            default=str,
            indent=2,
        )
    )
    return lines


def main() -> None:
    report_lines = ["Arabic parquet inspection report", ""]
    for filename in BOOK_FILES:
        path = RAW_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required parquet file not found: {path}")
        report_lines.extend(describe_book(path))
        report_lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        raise FileExistsError(
            f"Report already exists; preserve or rename it before rerunning: {REPORT_PATH}"
        )
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
