"""Cached corpus-level reference statistics used for comparison panels.

Every value is measured from the built corpus. There is no hard-coded fallback:
if the dataset has not been prepared, the function reports that rather than
inventing numbers, so a comparison panel can never silently show a statistic
that does not match the data on disk.
"""

from __future__ import annotations

import streamlit as st

from src.config import ENGINEERED_FEATURES


@st.cache_data(show_spinner=False, ttl="1h")
def corpus_signal_means() -> dict[str, float] | None:
    """Mean engineered-feature values across the whole corpus.

    Returns ``None`` when the engineered dataset is unavailable, so callers can
    render an explicit empty state instead of a fabricated comparison.
    """
    from app.services.artifacts import load_dataset

    try:
        frame = load_dataset()
    except FileNotFoundError:
        return None

    if frame.empty:
        return None

    available = [column for column in ENGINEERED_FEATURES if column in frame.columns]
    if not available:
        return None

    means = frame[available].mean().round(4).to_dict()
    return {key: float(value) for key, value in means.items()}
