"""Dataset loading and profiling for the news-authenticity pipeline.

The loader resolves the training corpus in priority order:

1. ``data/Fake.csv`` + ``data/True.csv`` -- the full Kaggle sources, when present.
2. ``data/sample_dataset.csv``            -- the bundled, versioned corpus.
3. ``data/cleaned_news.csv``              -- the derived, NLP-cleaned frame.

:func:`load_corpus` is the single entry point used by the scripts and the app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import (
    CLEAN_DATA,
    DATASET_PROFILE,
    FAKE_RAW,
    LABEL_NAMES,
    RAW_DATA,
    SAMPLE_DATA,
    TRUE_RAW,
)
from src.utils import build_full_text

#: Columns every corpus is guaranteed to expose.
CORE_COLUMNS = ["title", "text", "label"]


def _has_full_sources() -> bool:
    return FAKE_RAW.exists() and TRUE_RAW.exists()


def _read_full_sources() -> pd.DataFrame:
    """Concatenate the original Kaggle fake/real files into one labelled frame."""
    fake = pd.read_csv(FAKE_RAW)
    real = pd.read_csv(TRUE_RAW)

    fake["label"] = 1
    real["label"] = 0

    frame = pd.concat([fake, real], ignore_index=True)
    frame["subject"] = frame.get("subject", pd.Series(dtype="object")).fillna("unknown")
    return frame


def resolve_corpus_path() -> Path:
    """Return the CSV that should feed the pipeline, preferring real sources."""
    if _has_full_sources():
        return FAKE_RAW  # Sentinel: the caller handles the two-file case.
    if SAMPLE_DATA.exists():
        return SAMPLE_DATA
    if RAW_DATA.exists():
        return RAW_DATA
    raise FileNotFoundError(
        "No corpus found. Expected data/Fake.csv + data/True.csv, "
        "data/sample_dataset.csv or data/GlobalFakeNews_Research2026_v1.csv."
    )


def load_corpus() -> pd.DataFrame:
    """Load the source corpus with ``title``/``text``/``label`` guaranteed."""
    if _has_full_sources():
        frame = _read_full_sources()
    else:
        path = resolve_corpus_path()
        frame = pd.read_csv(path)

    missing = [column for column in CORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Corpus is missing required columns: {missing}")

    frame = frame[frame["text"].notna() | frame["title"].notna()].copy()
    frame["label"] = frame["label"].astype(int)
    frame["title"] = frame["title"].fillna("")
    frame["text"] = frame["text"].fillna("")
    return frame.reset_index(drop=True)


def load_training_frame() -> pd.DataFrame:
    """Load the NLP-cleaned frame the model was fitted on."""
    if not CLEAN_DATA.exists():
        raise FileNotFoundError(
            "data/cleaned_news.csv not found. Run `python -m src.preprocessing` first."
        )
    return pd.read_csv(CLEAN_DATA)


def load_prepared_frame() -> pd.DataFrame:
    """Load the merged + engineered corpus produced by ``src.prepare_dataset``.

    Falls back to :func:`load_corpus` when stage 1 has not been run yet, so the
    downstream stages work on a fresh clone either way. Using this (rather than
    :func:`load_corpus`) guarantees that ``full_text`` and the engineered signals
    are the values the preparation stage actually computed, instead of any
    placeholder columns that happen to ship inside a source file.
    """
    if not RAW_DATA.exists():
        return load_corpus()

    frame = pd.read_csv(RAW_DATA)
    missing = [column for column in CORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{RAW_DATA.name} is missing required columns: {missing}. "
            "Re-run `python -m src.prepare_dataset`."
        )
    if "full_text" not in frame.columns:
        frame["full_text"] = [
            build_full_text(title, body) for title, body in zip(frame["title"], frame["text"])
        ]
    return frame


def label_summary(frame: pd.DataFrame) -> dict:
    counts = frame["label"].value_counts().to_dict()
    total = int(len(frame))
    return {
        "total": total,
        "real": int(counts.get(0, 0)),
        "fake": int(counts.get(1, 0)),
        "real_share": round(counts.get(0, 0) / total, 4) if total else 0.0,
        "fake_share": round(counts.get(1, 0) / total, 4) if total else 0.0,
    }


def profile_corpus(frame: pd.DataFrame) -> dict:
    """Build a JSON-serialisable statistical profile of the corpus."""
    profile = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_values": {c: int(v) for c, v in frame.isna().sum().items() if v},
        "labels": label_summary(frame),
        "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
    }

    if "subject" in frame.columns:
        profile["subjects"] = {
            str(k): int(v) for k, v in frame["subject"].value_counts().items()
        }

    lengths = frame["text"].astype(str).str.len()
    profile["characters"] = {
        "mean": round(float(lengths.mean()), 2),
        "median": round(float(lengths.median()), 2),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
    }

    words = frame["text"].astype(str).str.split().str.len()
    profile["words"] = {
        "mean": round(float(words.mean()), 2),
        "median": round(float(words.median()), 2),
        "min": int(words.min()),
        "max": int(words.max()),
    }

    return profile


def save_profile(profile: dict, path: Path = DATASET_PROFILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
    return path


def load_profile() -> dict:
    """Read the cached profile, returning an empty dict when it is absent."""
    if not DATASET_PROFILE.exists():
        return {}
    with open(DATASET_PROFILE, encoding="utf-8") as handle:
        return json.load(handle)
