"""Exploratory data analysis for the training corpus.

Every number printed here is computed from the dataset on disk -- nothing is
read from a cached result or written by hand. The console report covers the
classic EDA checklist:

* dataset size, shape and column types
* missing values and the rows they affect
* duplicate / degenerate rows
* class balance, overall and per news desk
* text-length statistics, overall and split by class
* vocabulary size, hapax legomena and the terms that separate the classes
* engineered-feature profiles per class
* how much the cleaning chain removes

A condensed machine-readable profile is also cached to
``results/dataset_profile.json`` for the Streamlit *Dataset explorer* page.

Run with::

    python -m src.eda
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import (
    DATASET_PROFILE,
    ENGINEERED_FEATURES,
    LABEL_NAMES,
    RAW_DATA,
    RESULT_DIR,
    ensure_dirs,
)
from src.data_loader import (
    load_prepared_frame,
    load_training_frame,
    profile_corpus,
    save_profile,
)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 30)

FEATURE_LABELS = {
    "sentiment_score": "Sentiment (-1..1)",
    "clickbait_score": "Clickbait (0..1)",
    "emotional_word_ratio": "Emotional ratio (0..1)",
    "readability_score": "Readability (0..100)",
    "source_trust_score": "Source trust (0..1)",
}

#: How many terms to print per class in the vocabulary section.
TOP_TERMS = 15


def _rule(title: str) -> None:
    print("\n" + "-" * 66)
    print(title)
    print("-" * 66)


# --------------------------------------------------------------------------
# Structure & data quality
# --------------------------------------------------------------------------

def report_structure(frame: pd.DataFrame) -> None:
    _rule("DATASET OVERVIEW")
    print(f"Rows            : {frame.shape[0]:,}")
    print(f"Columns         : {frame.shape[1]}")
    print(f"Memory (MB)     : {frame.memory_usage(deep=True).sum() / 1e6:.2f}")

    _rule("COLUMN TYPES")
    print(frame.dtypes.to_string())

    _rule("MISSING VALUES")
    missing = frame.isna().sum()
    if (missing > 0).any():
        print(missing[missing > 0].to_string())
        print(
            f"\n{len(missing[missing > 0])} column(s) contain nulls. "
            "`src/data_loader.py` fills text and title nulls with an empty string "
            "before any feature extraction runs, so a null body can never reach the "
            "vectoriser."
        )
    else:
        print("None")


def report_quality(frame: pd.DataFrame) -> None:
    """Duplicates, empty documents and label validity."""
    _rule("DATA QUALITY")

    duplicated = int(frame.duplicated(subset=["title", "text"]).sum())
    empty_text = int((frame["text"].astype(str).str.strip() == "").sum())
    empty_title = int((frame["title"].astype(str).str.strip() == "").sum())

    labels = sorted(frame["label"].unique().tolist())
    unknown = [value for value in labels if value not in LABEL_NAMES]

    print(f"Duplicate title+text rows : {duplicated:,}")
    print(f"Rows with empty body      : {empty_text:,}")
    print(f"Rows with empty headline  : {empty_title:,}")
    print(f"Label values present      : {labels}")
    print(f"Unrecognised labels       : {unknown if unknown else 'none'}")
    print(
        "Duplicates are removed in `src/prepare_dataset.py`; rows that survive "
        "cleaning with no tokens at all are dropped when the split is drawn in "
        "`src/feature_engineering.py`."
    )


# --------------------------------------------------------------------------
# Class balance
# --------------------------------------------------------------------------

def report_classes(frame: pd.DataFrame) -> None:
    _rule("CLASS BALANCE")
    counts = frame["label"].value_counts().reindex([0, 1], fill_value=0)
    share = (counts / max(counts.sum(), 1) * 100).round(2)
    print(
        pd.DataFrame(
            {
                "label": [LABEL_NAMES[0], LABEL_NAMES[1]],
                "articles": counts.values,
                "share_%": share.values,
            }
        ).to_string(index=False)
    )
    balance = counts.min() / max(counts.max(), 1)
    print(f"\nMajority/minority ratio  : {balance:.2f}  "
          f"(below 1.0 means the corpus is majority-{LABEL_NAMES[int(counts.idxmax())]})")

    if "subject" in frame.columns:
        _rule("NEWS DESKS")
        desks = frame["subject"].value_counts()
        print(desks.to_string())

        _rule("DESK x LABEL CROSSTAB")
        crosstab = pd.crosstab(frame["subject"], frame["label"])
        crosstab.columns = [LABEL_NAMES[int(column)] for column in crosstab.columns]
        crosstab["fake_share_%"] = (
            crosstab[LABEL_NAMES[1]] / crosstab.sum(axis=1) * 100
        ).round(1)
        print(crosstab.sort_values("fake_share_%", ascending=False).to_string())


# --------------------------------------------------------------------------
# Text statistics
# --------------------------------------------------------------------------

def report_length(frame: pd.DataFrame) -> None:
    _rule("LENGTH PROFILE (all articles)")
    lengths = frame["full_text"].astype(str)
    stats = pd.DataFrame(
        {
            "metric": ["characters", "words", "sentences"],
            "mean": [
                round(lengths.str.len().mean(), 1),
                round(lengths.str.split().str.len().mean(), 1),
                round(lengths.str.count(r"[.!?]+").mean(), 1),
            ],
            "median": [
                lengths.str.len().median(),
                lengths.str.split().str.len().median(),
                lengths.str.count(r"[.!?]+").median(),
            ],
            "std": [
                round(lengths.str.len().std(), 1),
                round(lengths.str.split().str.len().std(), 1),
                round(lengths.str.count(r"[.!?]+").std(), 1),
            ],
            "min": [
                lengths.str.len().min(),
                lengths.str.split().str.len().min(),
                lengths.str.count(r"[.!?]+").min(),
            ],
            "max": [
                lengths.str.len().max(),
                lengths.str.split().str.len().max(),
                lengths.str.count(r"[.!?]+").max(),
            ],
        }
    )
    print(stats.to_string(index=False))

    _rule("LENGTH PROFILE BY CLASS")
    working = frame.assign(
        words=frame["full_text"].astype(str).str.split().str.len(),
        characters=frame["full_text"].astype(str).str.len(),
    )
    grouped = (
        working.groupby("label")[["characters", "words"]]
        .agg(["mean", "median"])
        .round(1)
        .reset_index()
    )
    grouped["label"] = grouped["label"].map(LABEL_NAMES)
    print(grouped.to_string(index=False))

    fake_mean = working.loc[working["label"] == 1, "words"].mean()
    real_mean = working.loc[working["label"] == 0, "words"].mean()
    print(
        f"\nMean length gap: {LABEL_NAMES[1]} articles are "
        f"{abs(fake_mean - real_mean):.0f} words "
        f"{'longer' if fake_mean > real_mean else 'shorter'} than {LABEL_NAMES[0]} ones. "
        "Length alone is therefore a weak signal, which is why the model reads the words "
        "rather than the word count."
    )


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

def _document_frequency(series: pd.Series) -> pd.Series:
    """How many documents each token appears in, computed class by class."""
    counts: dict[str, int] = {}
    for document in series.fillna("").astype(str):
        for token in set(document.split()):
            counts[token] = counts.get(token, 0) + 1
    return pd.Series(counts, dtype="int64").sort_values(ascending=False)


def report_vocabulary(frame: pd.DataFrame) -> None:
    _rule("VOCABULARY")
    column = "clean_text" if "clean_text" in frame.columns else "full_text"

    real = _document_frequency(frame.loc[frame["label"] == 0, column])
    fake = _document_frequency(frame.loc[frame["label"] == 1, column])
    combined = _document_frequency(frame[column])

    hapax = int((combined == 1).sum())
    print(f"Distinct tokens (all)     : {len(combined):,}")
    print(f"Appearing in one document : {hapax:,} ({hapax / max(len(combined), 1) * 100:.1f}% hapax legomena)")
    print(f"Appearing in >= 2 docs    : {int((combined >= 2).sum()):,}")

    _rule("MOST FREQUENT TERMS PER CLASS")
    print(f"{'term':<28}{LABEL_NAMES[0]:>10}{LABEL_NAMES[1]:>10}")
    terms = pd.concat([real.head(TOP_TERMS), fake.head(TOP_TERMS)], axis=1)
    terms.columns = [LABEL_NAMES[0], LABEL_NAMES[1]]
    terms = terms.fillna(0).astype(int).sort_values(LABEL_NAMES[1], ascending=False)
    print(terms.head(TOP_TERMS).to_string())

    _rule("CLASS-SEPARATING TERMS (log-odds ratio)")
    real_rate = (real / max(len(frame[frame['label'] == 0]), 1)).clip(lower=1e-4)
    fake_rate = (fake / max(len(frame[frame['label'] == 1]), 1)).clip(lower=1e-4)
    shared = real_rate.index.intersection(fake_rate.index)
    # Positive log-odds means the term is more frequent in FAKE documents.
    log_odds = (np_log(fake_rate.reindex(shared)) - np_log(real_rate.reindex(shared))).sort_values()

    toward_fake = log_odds.tail(8).round(3).sort_values(ascending=False)
    toward_real = log_odds.head(8).round(3).sort_values(ascending=False)
    print(f"-- most {LABEL_NAMES[1]}-indicative (log-odds > 0) --")
    print(toward_fake.to_string())
    print(f"\n-- most {LABEL_NAMES[0]}-indicative (log-odds < 0) --")
    print(toward_real.to_string())


def np_log(series: pd.Series) -> pd.Series:
    """Natural log, imported lazily so the module has no hard numpy dependency
    at import time in the app's chart path."""
    import numpy as np

    return np.log(series)


# --------------------------------------------------------------------------
# Engineered features & cleaning yield
# --------------------------------------------------------------------------

def report_engineered(frame: pd.DataFrame) -> None:
    available = [column for column in ENGINEERED_FEATURES if column in frame.columns]
    if not available:
        return

    _rule("ENGINEERED FEATURES BY CLASS")
    grouped = frame.groupby("label")[available].mean().round(3)
    grouped.index = [LABEL_NAMES[int(index)] for index in grouped.index]
    grouped.columns = [FEATURE_LABELS.get(column, column) for column in grouped.columns]
    print(grouped.to_string())

    _rule("ENGINEERED FEATURES: REAL vs FAKE DELTA")
    delta = (grouped.loc[LABEL_NAMES[1]] - grouped.loc[LABEL_NAMES[0]]).round(3)
    print(delta.to_string())
    print(
        "\nThese five signals are reported for auditability only -- they are never "
        "fed to the classifier, so the benchmark measures the text model rather "
        "than hand-written rules."
    )


def report_cleaning(frame: pd.DataFrame) -> None:
    if "clean_text" not in frame.columns:
        return

    _rule("CLEANING YIELD")
    before_tokens = frame["full_text"].astype(str).str.split().str.len()
    after_tokens = frame["clean_text"].fillna("").str.split().str.len()
    before = int(before_tokens.sum())
    after = int(after_tokens.sum())
    removed = before - after
    print(f"Tokens before cleaning : {before:,}")
    print(f"Tokens after cleaning  : {after:,}")
    print(f"Tokens removed         : {removed:,} ({removed / max(before, 1) * 100:.1f}%)")
    print(f"Rows with empty output : {int((frame['clean_text'].fillna('').str.len() == 0).sum())}")
    print(
        "\nThe removed tokens are stopwords, punctuation, digits, URLs and one- and "
        "two-character fragments, plus the morphological variation that lemmatisation "
        "collapses."
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def report_dataset(frame: pd.DataFrame) -> None:
    report_structure(frame)
    report_quality(frame)
    report_classes(frame)
    report_length(frame)
    report_vocabulary(frame)
    report_engineered(frame)
    report_cleaning(frame)


def main() -> None:
    ensure_dirs()

    print("=" * 66)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 66)

    # Prefer the cleaned frame: it carries both the engineered features and the
    # post-cleaning text, so vocabulary and cleaning yield can be measured.
    try:
        dataset = load_training_frame()
        source = "data/cleaned_news.csv (post-cleaning)"
    except FileNotFoundError:
        dataset = load_prepared_frame()
        source = str(RAW_DATA)

    print(f"Source: {source}")
    report_dataset(dataset)

    cleaned = dataset["clean_text"] if "clean_text" in dataset.columns else None
    profile = profile_corpus(dataset)
    profile["eda"] = {
        "source": source,
        "duplicate_rows": int(dataset.duplicated(subset=["title", "text"]).sum()),
        "empty_body_rows": int((dataset["text"].astype(str).str.strip() == "").sum()),
        "empty_cleaned_rows": int((cleaned.fillna("").str.strip() == "").sum()) if cleaned is not None else 0,
    }
    if cleaned is not None:
        profile["vocabulary"] = {
            "distinct_tokens": int(len(_document_frequency(cleaned))),
        }

    path = save_profile(profile)
    print(f"\nCached dataset profile written to {path}.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dashboard profile  : {DATASET_PROFILE.name}")
    print("=" * 66)


if __name__ == "__main__":
    main()
