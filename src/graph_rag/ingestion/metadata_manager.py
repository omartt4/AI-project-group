"""Metadata manager for the document registry."""

import threading
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from graph_rag.config import settings

# Re-entrant lock because update methods call the locked read helpers.
_csv_lock = threading.RLock()

EXPECTED_COLUMNS = [
    "topic",
    "title",
    "authors",
    "abstract",
    "categories",
    "local_pdf",
    "source",
    "file_hash",
    "ingested_at",
]


def _topic_scores(title: str = "", text: str = "") -> Dict[str, int]:
    """Score broad research areas using document title and text."""
    content = f"{title} {text}".lower()
    return {
        "NLP": sum(
            len(re.findall(rf"\b{re.escape(keyword)}\b", content))
            for keyword in (
                "natural language",
                "nlp",
                "language model",
                "large language model",
                "llm",
                "text",
                "speech",
                "dialogue",
                "question answering",
                "relation extraction",
                "named entity",
                "agent",
            )
        ),
        "Computer_Vision": sum(
            len(re.findall(rf"\b{re.escape(keyword)}\b", content))
            for keyword in (
                "computer vision",
                "image",
                "visual",
                "video",
                "object detection",
                "segmentation",
                "3d reconstruction",
                "scene",
            )
        ),
        "Machine_Learning": sum(
            len(re.findall(rf"\b{re.escape(keyword)}\b", content))
            for keyword in (
                "machine learning",
                "deep learning",
                "neural network",
                "optimization",
                "inference",
                "training",
                "classification",
                "regression",
                "reinforcement learning",
                "chiplet",
                "hardware",
            )
        ),
    }


def infer_topics(title: str = "", text: str = "") -> str:
    """Infer one or more broad research areas from a paper."""
    scores = _topic_scores(title, text)
    highest = max(scores.values())
    if not highest:
        return "Machine_Learning"
    selected = [topic for topic, score in scores.items() if score >= highest * 0.5]
    return "; ".join(selected)


def infer_topic(title: str = "", text: str = "") -> str:
    """Infer the primary broad research area from a paper."""
    return infer_topics(title, text).split("; ", 1)[0]


class MetadataManager:
    """Manages reading, querying, and appending to pdfs/metadata.csv."""

    def __init__(self, metadata_path: Optional[Path] = None) -> None:
        self.metadata_path = metadata_path or settings.METADATA_FILE
        self._ensure_file_and_columns()

    def _ensure_file_and_columns(self) -> None:
        """Ensure metadata.csv exists and has all required schema columns."""
        with _csv_lock:
            if not self.metadata_path.exists():
                self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(columns=EXPECTED_COLUMNS)
                df.to_csv(self.metadata_path, index=False)
                return

            try:
                df = pd.read_csv(self.metadata_path)
            except Exception:
                df = pd.DataFrame(columns=EXPECTED_COLUMNS)

            # Ensure tracking columns exist
            modified = False
            for col in EXPECTED_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
                    modified = True

            # Populate default source for initial documents if empty
            if "source" in df.columns:
                mask = df["source"].isna() | (df["source"] == "")
                if mask.any():
                    upload_mask = df["topic"].astype(str).str.strip().str.lower().isin(
                        {"user upload", "user-upload"}
                    )
                    df.loc[mask & upload_mask, "source"] = "upload"
                    df.loc[mask & ~upload_mask, "source"] = "initial"
                    modified = True

            modified |= self._repair_upload_rows(df)
            if list(df.columns) != EXPECTED_COLUMNS:
                df = df[EXPECTED_COLUMNS]
                modified = True

            if modified:
                df.to_csv(self.metadata_path, index=False)

    @staticmethod
    def _repair_upload_rows(df: pd.DataFrame) -> bool:
        """Repair legacy upload rows whose values were written in shifted columns."""
        modified = False
        upload_mask = (
            df["topic"].astype(str).str.strip().str.lower().isin({"user upload", "user-upload"})
            | df["source"].astype(str).str.strip().str.lower().eq("upload")
        )

        for index in df.index[upload_mask]:
            # A legacy row omitted the trailing schema columns and shifted the
            # filename, source, hash, and timestamp into earlier columns.
            if (
                "published" in df.columns
                and "arxiv_id" in df.columns
                and str(df.at[index, "authors"]).strip().lower() == "upload"
                and re.fullmatch(r"[0-9a-f]{64}", str(df.at[index, "published"]).strip(), re.I)
                and str(df.at[index, "arxiv_id"]).strip()
            ):
                filename = str(df.at[index, "title"]).strip()
                file_hash = str(df.at[index, "published"]).strip().lower()
                ingested_at = str(df.at[index, "arxiv_id"]).strip()
                df.at[index, "local_pdf"] = filename
                df.at[index, "source"] = "upload"
                df.at[index, "file_hash"] = file_hash
                df.at[index, "ingested_at"] = ingested_at
                df.at[index, "authors"] = ""
                df.at[index, "abstract"] = ""
                modified = True

            title = str(df.at[index, "title"]).strip()
            abstract = str(df.at[index, "abstract"]).strip()
            inferred_topic = infer_topics(title, abstract)
            if str(df.at[index, "topic"]).strip().lower() in {"user upload", "user-upload"}:
                df.at[index, "topic"] = inferred_topic
                modified = True
            if not str(df.at[index, "categories"]).strip():
                df.at[index, "categories"] = inferred_topic
                modified = True
            if not str(df.at[index, "source"]).strip():
                df.at[index, "source"] = "upload"
                modified = True

        return modified

    def load_metadata(self) -> pd.DataFrame:
        """Return the current metadata dataframe."""
        with _csv_lock:
            if not self.metadata_path.exists():
                return pd.DataFrame(columns=EXPECTED_COLUMNS)
            df = pd.read_csv(self.metadata_path)
            # Fill NaNs with empty string for string columns
            return df.fillna("")

    def get_known_hashes(self) -> Set[str]:
        """Return set of all non-empty SHA-256 hashes present in registry."""
        df = self.load_metadata()
        if "file_hash" not in df.columns:
            return set()
        hashes = set(df["file_hash"].dropna().astype(str).str.strip().str.lower())
        hashes.discard("")
        return hashes

    def get_known_filenames(self) -> Set[str]:
        """Return set of all registered local filenames (case-insensitive)."""
        df = self.load_metadata()
        if "local_pdf" not in df.columns:
            return set()
        names = set(df["local_pdf"].dropna().astype(str).str.strip().str.lower())
        names.discard("")
        return names

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Return list of all registered document records as dictionaries."""
        df = self.load_metadata()
        return df.to_dict(orient="records")

    def find_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Find a document by exact SHA-256 hash."""
        clean_hash = file_hash.strip().lower()
        if not clean_hash:
            return None
        df = self.load_metadata()
        match = df[df["file_hash"].astype(str).str.strip().str.lower() == clean_hash]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None

    def find_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Find a document by filename (local_pdf)."""
        clean_name = Path(filename).name.strip().lower()
        if not clean_name:
            return None
        df = self.load_metadata()
        match = df[df["local_pdf"].astype(str).str.strip().str.lower() == clean_name]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None

    def append_document(self, doc_data: Dict[str, Any]) -> None:
        """Safely append a newly ingested document to metadata.csv."""
        with _csv_lock:
            df = self.load_metadata()
            new_row = {col: doc_data.get(col, "") for col in EXPECTED_COLUMNS}
            new_row["topic"] = new_row["topic"] or infer_topics(
                str(new_row["title"]), str(new_row["abstract"])
            )
            if not new_row.get("source"):
                new_row["source"] = "upload"
            if not new_row.get("ingested_at"):
                new_row["ingested_at"] = datetime.now(timezone.utc).isoformat()

            new_df = pd.DataFrame([new_row])
            updated_df = pd.concat([df, new_df], ignore_index=True)
            updated_df.to_csv(self.metadata_path, index=False)

    def mark_as_ingested(self, local_pdf: str, file_hash: str) -> None:
        """Update existing row with hash and ingested_at timestamp."""
        with _csv_lock:
            df = self.load_metadata()
            target_name = Path(local_pdf).name.strip().lower()
            mask = df["local_pdf"].astype(str).str.strip().str.lower() == target_name
            if mask.any():
                df.loc[mask, "file_hash"] = file_hash.strip().lower()
                df.loc[mask, "ingested_at"] = datetime.now(timezone.utc).isoformat()
                df.to_csv(self.metadata_path, index=False)

    def get_document_count(self) -> int:
        """Return total count of documents registered."""
        df = self.load_metadata()
        return len(df)


# Global singleton instance
metadata_manager = MetadataManager()
