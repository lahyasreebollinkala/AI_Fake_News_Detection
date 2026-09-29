"""Figure builders for the exploratory data analysis stage.

Every function returns a ``matplotlib`` figure so the charts can be reused by
the scripts or embedded elsewhere. ``main()`` renders the full set to
``images/`` and is the module's command-line entry point.

Run with::

    python -m src.eda_visualization
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import IMAGE_DIR, ensure_dirs  # noqa: E402
from src.evaluate_models import (  # noqa: E402
    FAKE_COLOR,
    INK,
    MUTED,
    PRIMARY,
    REAL_COLOR,
    style_axis,
)

LABEL_COLORS = {0: REAL_COLOR, 1: FAKE_COLOR}
LABEL_TICKS = {0: "Real", 1: "Fake"}

FEATURE_LABELS = {
    "sentiment_score": "Sentiment",
    "clickbait_score": "Clickbait",
    "emotional_word_ratio": "Emotional ratio",
    "readability_score": "Readability",
    "source_trust_score": "Source trust",
}


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def label_distribution(frame) -> "plt.Figure":
    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    style_axis(axis, xlabel="Label", ylabel="Articles")

    counts = frame["label"].value_counts().reindex([0, 1], fill_value=0)
    bars = axis.bar(
        [LABEL_TICKS[0], LABEL_TICKS[1]],
        counts.values,
        color=[LABEL_COLORS[0], LABEL_COLORS[1]],
        width=0.55,
    )
    axis.bar_label(bars, padding=3, fontsize=10, color=INK, fontweight="bold")
    axis.set_title("Real vs fake", color=INK, fontsize=13, fontweight="bold", loc="left", pad=14)
    return figure


def length_distribution(frame, length_column: str = "char_count") -> "plt.Figure":
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    style_axis(axis, xlabel="Characters per article", ylabel="Articles")

    for label, subset in frame.groupby("label"):
        axis.hist(
            subset[length_column].clip(upper=int(frame[length_column].quantile(0.99))),
            bins=45,
            alpha=0.62,
            label=LABEL_TICKS.get(int(label), str(label)),
            color=LABEL_COLORS.get(int(label), MUTED),
            edgecolor="white",
            linewidth=0.5,
        )

    axis.legend(frameon=False, fontsize=9)
    axis.set_title(
        "Article length by class", color=INK, fontsize=13, fontweight="bold", loc="left", pad=14
    )
    return figure


def average_length(frame, length_column: str = "char_count") -> "plt.Figure":
    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    style_axis(axis, xlabel="Label", ylabel="Mean characters")

    means = frame.groupby("label")[length_column].mean().reindex([0, 1], fill_value=0)
    bars = axis.bar(
        [LABEL_TICKS[0], LABEL_TICKS[1]],
        means.values,
        color=[LABEL_COLORS[0], LABEL_COLORS[1]],
        width=0.55,
    )
    axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=10, color=INK, fontweight="bold")
    axis.set_title(
        "Average article length", color=INK, fontsize=13, fontweight="bold", loc="left", pad=14
    )
    return figure


def subject_distribution(frame, top: int = 10) -> "plt.Figure":
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    style_axis(axis, xlabel="Articles", grid_axis="x")
    axis.grid(axis="y", visible=False)

    counts = frame["subject"].value_counts().head(top).sort_values()
    axis.barh(counts.index, counts.values, color=PRIMARY, height=0.66)
    axis.tick_params(axis="y", labelsize=9)
    axis.set_title(
        "News desks represented", color=INK, fontsize=13, fontweight="bold", loc="left", pad=14
    )
    return figure


def word_count_box(frame) -> "plt.Figure":
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    style_axis(axis, ylabel="Words per article")

    data = [
        frame.loc[frame["label"] == label, "word_count"].clip(upper=800)
        for label in (0, 1)
    ]
    boxes = axis.boxplot(data, patch_artist=True, widths=0.5, showfliers=False)

    for patch, color in zip(boxes["boxes"], (REAL_COLOR, FAKE_COLOR)):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    for element in ("whiskers", "caps", "medians"):
        for artist in boxes[element]:
            artist.set_color(MUTED)

    axis.set_xticks([1, 2])
    axis.set_xticklabels([LABEL_TICKS[0], LABEL_TICKS[1]], fontsize=10, color=INK)
    axis.set_title(
        "Word count spread", color=INK, fontsize=13, fontweight="bold", loc="left", pad=14
    )
    return figure


def engineered_feature_profiles(frame, features: list[str]) -> "plt.Figure":
    """Small-multiple means of the engineered editorial features by class."""
    available = [f for f in features if f in frame.columns]
    if not available:
        raise ValueError("No engineered feature columns found in the frame.")

    figure, axes = plt.subplots(1, len(available), figsize=(3.0 * len(available), 3.4))
    if len(available) == 1:
        axes = [axes]

    for axis, feature in zip(np.atleast_1d(axes), available):
        style_axis(axis, ylabel="Mean value")
        means = frame.groupby("label")[feature].mean().reindex([0, 1], fill_value=0)
        bars = axis.bar(
            [0, 1], means.values, color=[REAL_COLOR, FAKE_COLOR], width=0.5
        )
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=MUTED)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Real", "Fake"], fontsize=9, color=INK)
        axis.set_title(FEATURE_LABELS.get(feature, feature), fontsize=10.5, color=INK, pad=10)

    figure.suptitle(
        "Engineered editorial signals by class",
        color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left", y=1.04,
    )
    return figure


def feature_correlation(frame, features: list[str]) -> "plt.Figure":
    available = [f for f in features if f in frame.columns]
    if len(available) < 2:
        raise ValueError("Need at least two engineered features for a correlation matrix.")

    matrix = frame[available].corr()

    figure, axis = plt.subplots(figsize=(6.0, 5.0))
    image = axis.imshow(matrix, cmap="Purples", vmin=-1, vmax=1)
    axis.set_xticks(range(len(available)))
    axis.set_xticklabels([FEATURE_LABELS.get(f, f) for f in available], rotation=35, ha="right",
                         fontsize=9, color=INK)
    axis.set_yticks(range(len(available)))
    axis.set_yticklabels([FEATURE_LABELS.get(f, f) for f in available], fontsize=9, color=INK)
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)

    for row in range(len(available)):
        for column in range(len(available)):
            value = matrix.iloc[row, column]
            axis.text(
                column, row, f"{value:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if abs(value) > 0.55 else INK,
            )

    figure.colorbar(image, ax=axis, shrink=0.8, label="Pearson r")
    axis.set_title(
        "Feature correlation", color=INK, fontsize=13, fontweight="bold", loc="left", pad=14
    )
    return figure


def term_separation(frame, text_column: str = "clean_text", top: int = 12) -> "plt.Figure":
    """Log-odds ratio of the terms that most separate the two classes.

    A term that appears in 40% of fabricated articles and 0.5% of authentic ones
    has a strongly positive log-odds ratio, and vice versa. The chart is the
    textual counterpart of the confusion matrix: it shows *what* the model is
    keying on when it separates the classes.
    """
    if text_column not in frame.columns:
        raise ValueError(f"Column '{text_column}' not present; run the preprocessing stage first.")

    def document_frequency(mask: pd.Series) -> pd.Series:
        counts: dict[str, int] = {}
        for document in frame.loc[mask, text_column].fillna("").astype(str):
            for token in set(document.split()):
                counts[token] = counts.get(token, 0) + 1
        return pd.Series(counts, dtype="int64")

    real = document_frequency(frame["label"] == 0)
    fake = document_frequency(frame["label"] == 1)

    if real.empty or fake.empty:
        raise ValueError("Both classes must be present to compare their vocabularies.")

    real_rate = (real / max(int((frame["label"] == 0).sum()), 1)).clip(lower=1e-3)
    fake_rate = (fake / max(int((frame["label"] == 1).sum()), 1)).clip(lower=1e-3)

    shared = real_rate.index.intersection(fake_rate.index)
    log_odds = np.log(fake_rate.reindex(shared)) - np.log(real_rate.reindex(shared))

    toward_fake = log_odds.nlargest(top).sort_values()
    toward_real = log_odds.nsmallest(top).sort_values()
    selected = pd.concat([toward_real, toward_fake])
    selected = selected[selected.abs() > 0]

    figure, axis = plt.subplots(figsize=(7.6, 0.38 * len(selected) + 1.8))
    style_axis(axis, xlabel="Log-odds ratio  (left = Real, right = Fake)", grid_axis="x")

    axis.barh(
        [str(term) for term in selected.index],
        selected.values,
        color=[FAKE_COLOR if value > 0 else REAL_COLOR for value in selected.values],
        height=0.7,
    )
    axis.axvline(0, color=INK, linewidth=1)
    axis.tick_params(axis="y", labelsize=8.5)
    axis.set_title(
        "Terms that separate the classes",
        color=INK, fontsize=13, fontweight="bold", loc="left", pad=14,
    )
    return figure



# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def build_frames() -> list[tuple[str, "object"]]:
    """Assemble the (filename, figure-builder) pairs rendered by :func:`main`."""
    from src.config import ENGINEERED_FEATURES
    from src.data_loader import load_prepared_frame, load_training_frame

    try:
        frame = load_training_frame()
    except FileNotFoundError:
        frame = load_prepared_frame()

    frame = frame.copy()
    frame["char_count"] = frame["full_text"].astype(str).str.len()
    frame["word_count"] = frame["full_text"].astype(str).str.split().str.len()

    charts = [
        ("label_distribution.png", lambda: label_distribution(frame)),
        ("length_distribution.png", lambda: length_distribution(frame)),
        ("average_length.png", lambda: average_length(frame)),
        ("subject_distribution.png", lambda: subject_distribution(frame)),
        ("word_count_box.png", lambda: word_count_box(frame)),
        (
            "engineered_features.png",
            lambda: engineered_feature_profiles(frame, ENGINEERED_FEATURES),
        ),
        ("feature_correlation.png", lambda: feature_correlation(frame, ENGINEERED_FEATURES)),
    ]

    if "clean_text" in frame.columns:
        charts.append(("term_separation.png", lambda: term_separation(frame)))

    return charts


def main() -> None:
    ensure_dirs()

    print("=" * 66)
    print("EDA FIGURES")
    print("=" * 66)

    for name, builder in build_frames():
        figure = builder()
        figure.tight_layout()
        figure.savefig(IMAGE_DIR / name, dpi=200, facecolor="white")
        plt.close(figure)
        print(f"  wrote images/{name}")

    print("=" * 66)


if __name__ == "__main__":
    main()
