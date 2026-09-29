"""Stage 2 -- NLP preprocessing.

Applies the cleaning pipeline in :func:`src.utils.clean_text` to every
``full_text`` value and writes ``data/cleaned_news.csv``. That file is the only
input :mod:`src.feature_engineering` reads.

Run with::

    python -m src.preprocessing
"""

from __future__ import annotations

import pandas as pd

from src.config import CLEAN_DATA, ensure_dirs
from src.utils import clean_series


def preprocess_frame(frame: pd.DataFrame, text_column: str = "full_text") -> pd.DataFrame:
    """Return ``frame`` with an added ``clean_text`` column."""
    result = frame.copy()
    result["clean_text"] = clean_series(result[text_column])
    return result


def main() -> None:
    from src.data_loader import load_prepared_frame

    ensure_dirs()

    print("=" * 66)
    print("STAGE 2  |  NLP PREPROCESSING")
    print("=" * 66)

    corpus = load_prepared_frame()
    cleaned = preprocess_frame(corpus)

    cleaned.to_csv(CLEAN_DATA, index=False)

    empty = int((cleaned["clean_text"].str.len() == 0).sum())

    print(f"Rows processed        : {len(cleaned)}")
    print(f"Rows with empty text  : {empty}")
    print(f"Output                : {CLEAN_DATA}")
    print("-" * 66)
    print("Original :", str(cleaned["full_text"].iloc[0])[:160])
    print("Cleaned  :", str(cleaned["clean_text"].iloc[0])[:160])
    print("=" * 66)


if __name__ == "__main__":
    main()
