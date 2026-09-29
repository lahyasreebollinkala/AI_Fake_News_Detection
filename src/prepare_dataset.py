"""Stage 1 -- build the merged, feature-enriched training corpus.

Reads the source corpus (full Kaggle files when available, otherwise the bundled
sample) and writes ``data/GlobalFakeNews_Research2026_v1.csv`` containing:

``title``, ``text``, ``subject``, ``label``, ``full_text`` plus the five
engineered editorial features.

Run with::

    python -m src.prepare_dataset
"""

from __future__ import annotations

import pandas as pd

from src.config import ENGINEERED_FEATURES, RAW_DATA, ensure_dirs
from src.data_loader import load_corpus, profile_corpus, save_profile
from src.utils import build_full_text, extract_article_metadata

OUTPUT_COLUMNS = ["title", "text", "subject", "label", "full_text", *ENGINEERED_FEATURES]


def build_dataset() -> pd.DataFrame:
    """Return the merged, de-duplicated and engineered corpus."""
    corpus = load_corpus()

    frame = pd.DataFrame(
        {
            "title": corpus["title"].astype(str).str.strip(),
            "text": corpus["text"].astype(str).str.strip(),
            "subject": corpus.get("subject", pd.Series("unknown", index=corpus.index))
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown"),
            "label": corpus["label"].astype(int),
        }
    )

    # Identical headline+body pairs would let the same article appear on both
    # sides of the split, which inflates every score. Remove them here, before
    # anything is fitted.
    duplicates = int(frame.duplicated(subset=["title", "text"]).sum())
    if duplicates:
        frame = frame.drop_duplicates(subset=["title", "text"], keep="first")

    frame["full_text"] = [
        build_full_text(title, body) for title, body in zip(frame["title"], frame["text"])
    ]

    metadata = [
        extract_article_metadata(full_text, title)
        for full_text, title in zip(frame["full_text"], frame["title"])
    ]
    for column in ENGINEERED_FEATURES:
        frame[column] = [entry[column] for entry in metadata]

    frame = frame[OUTPUT_COLUMNS]
    frame = frame[frame["full_text"].str.len() > 0].reset_index(drop=True)
    frame.attrs["duplicates_removed"] = duplicates
    return frame


def main() -> None:
    ensure_dirs()

    print("=" * 66)
    print("STAGE 1  |  DATASET PREPARATION")
    print("=" * 66)

    frame = build_dataset()
    frame.to_csv(RAW_DATA, index=False)

    profile = save_profile(profile_corpus(frame))

    print(f"Rows written       : {len(frame):,}")
    print(f"Duplicates removed : {frame.attrs.get('duplicates_removed', 0):,}")
    print(f"Output             : {RAW_DATA}")
    print(f"Profile            : {profile}")
    print(f"Real / fake        : {(frame['label'] == 0).sum():,} / {(frame['label'] == 1).sum():,}")
    print("=" * 66)


if __name__ == "__main__":
    main()
