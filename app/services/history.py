"""Prediction log stored in ``results/prediction_history.csv``.

The log is intentionally simple -- append-only CSV with a fixed schema and a
row cap -- so it survives a server restart, stays diff-friendly and can be
downloaded straight from the UI.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pandas as pd

from src.config import PREDICTION_HISTORY

COLUMNS = [
    "Timestamp",
    "Source",
    "Title",
    "Prediction",
    "Confidence",
    "Real score",
    "Fake score",
    "Words",
    "Model",
]

#: Keep the file bounded; older rows are dropped on write.
MAX_ROWS = 5000


def _ensure_parent() -> None:
    PREDICTION_HISTORY.parent.mkdir(parents=True, exist_ok=True)


def load_history() -> pd.DataFrame:
    """Return the log, or an empty frame with the correct schema."""
    if not PREDICTION_HISTORY.exists():
        return pd.DataFrame(columns=COLUMNS)

    try:
        frame = pd.read_csv(PREDICTION_HISTORY)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=COLUMNS)

    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)

    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    return frame[COLUMNS]


def _atomic_write(frame: pd.DataFrame) -> None:
    """Replace the log in one step so a crash cannot truncate it."""
    _ensure_parent()
    handle = tempfile.NamedTemporaryFile(
        "w", delete=False, dir=PREDICTION_HISTORY.parent, suffix=".tmp", encoding="utf-8", newline=""
    )
    try:
        with handle:
            frame.to_csv(handle, index=False)
        os.replace(handle.name, PREDICTION_HISTORY)
    except Exception:
        os.unlink(handle.name)
        raise


def append_record(record: dict) -> None:
    """Append one prediction, trimming the log to :data:`MAX_ROWS`."""
    row = {column: record.get(column, "") for column in COLUMNS}
    row["Timestamp"] = row["Timestamp"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing = load_history()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated = updated.tail(MAX_ROWS)

    _atomic_write(updated)


def remove_entry(label) -> bool:
    """Drop a single row by its label in the stored log."""
    frame = load_history()
    if frame.empty or label not in frame.index:
        return False

    _atomic_write(frame.drop(index=label).reset_index(drop=True))
    return True


def clear_history() -> bool:
    """Delete the log. Returns ``True`` when a file was removed."""
    if PREDICTION_HISTORY.exists():
        PREDICTION_HISTORY.unlink()
        return True
    return False


def summarize(history: pd.DataFrame) -> dict:
    """Aggregate counts used by the KPI row on the history page."""
    if history.empty:
        return {
            "total": 0,
            "real": 0,
            "fake": 0,
            "fake_share": 0.0,
            "mean_confidence": 0.0,
            "low_confidence": 0,
        }

    numeric = pd.to_numeric(history["Confidence"], errors="coerce")
    fake = int((history["Prediction"] == "Fake").sum())

    return {
        "total": int(len(history)),
        "real": int((history["Prediction"] == "Real").sum()),
        "fake": fake,
        "fake_share": round(fake / len(history) * 100, 1),
        "mean_confidence": round(float(numeric.mean()), 2) if numeric.notna().any() else 0.0,
        "low_confidence": int((numeric < 80).sum()),
    }


def filter_history(
    history: pd.DataFrame,
    *,
    query: str = "",
    verdicts: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the free-text search and the verdict filter.

    The original index labels are preserved so a filtered view can still be used
    to address rows in the stored log.
    """
    frame = history

    if verdicts:
        frame = frame[frame["Prediction"].isin(verdicts)]

    needle = query.strip()
    if needle:
        haystack = (
            frame["Title"].fillna("").astype(str)
            + " "
            + frame["Source"].fillna("").astype(str)
            + " "
            + frame["Prediction"].fillna("").astype(str)
        )
        frame = frame[haystack.str.contains(needle, case=False, regex=False)]

    return frame


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialise a log slice for :func:`st.download_button`."""
    return frame.to_csv(index=False).encode("utf-8")


def build_record(
    *,
    source: str,
    title: str,
    prediction: str,
    confidence: float,
    real_score: float,
    fake_score: float,
    words: int,
    model: str,
) -> dict:
    """Assemble a log row from an :class:`~app.services.inference.Verdict`."""
    return {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source,
        "Title": (title or "Untitled").strip()[:180],
        "Prediction": prediction,
        "Confidence": round(float(confidence) * 100, 2),
        "Real score": round(float(real_score) * 100, 2),
        "Fake score": round(float(fake_score) * 100, 2),
        "Words": int(words),
        "Model": model,
    }
