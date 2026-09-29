"""Stage 5 -- evaluation reporting.

Reads the benchmark written by :mod:`src.train_models`, re-scores every
candidate on the held-out split, and renders the publication-quality figures the
*Model lab* page displays. Every number in every figure is computed here from
real predictions -- nothing is read from a previous run's cache or hard-coded.

Generated figures:

* ``images/confusion_matrix.png``      -- the selected model
* ``images/confusion_matrices.png``    -- all four models, one grid
* ``images/roc_curve.png``             -- the selected model
* ``images/roc_comparison.png``        -- every candidate on one chart
* ``images/metric_comparison.png``     -- accuracy / precision / recall / F1
* ``images/feature_importance.png``    -- the terms the model leans on
* ``images/representation_comparison.png`` -- the 3 x 4 representation grid

Run with::

    python -m src.evaluate_models
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_curve as roc_curve_points,
)

from src.config import (  # noqa: E402
    BEST_MODEL,
    FEATURE_NAMES,
    IMAGE_DIR,
    METRICS_FILE,
    MODEL_DIR,
    XTRAIN,
    XTEST,
    YTEST,
    YTRAIN,
    ensure_dirs,
)
from src.feature_extraction import build_representations  # noqa: E402
from src.train_models import build_models  # noqa: E402

#: Brand-consistent palette shared with the Streamlit theme.
INK = "#1F2333"
MUTED = "#7C8199"
FAKE_COLOR = "#E11D48"
REAL_COLOR = "#0E9F6E"
PRIMARY = "#4F46E5"
GRID = "#E4E4EC"

MODEL_COLORS = {
    "Logistic Regression": PRIMARY,
    "Random Forest": REAL_COLOR,
    "Neural Network": "#0EA5E9",
    "K-Nearest Neighbors": "#D97706",
}


# --------------------------------------------------------------------------
# Figure helpers
# --------------------------------------------------------------------------

def style_axis(axis, *, xlabel: str = "", ylabel: str = "", grid_axis: str = "y") -> None:
    axis.set_facecolor("white")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9, length=0)
    if xlabel:
        axis.set_xlabel(xlabel, color=INK, fontsize=10, fontweight="semibold")
    if ylabel:
        axis.set_ylabel(ylabel, color=INK, fontsize=10, fontweight="semibold")
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)


def _title(axis, text: str) -> None:
    axis.set_title(text, color=INK, fontsize=13, fontweight="bold", loc="left", pad=14)


def _save(figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(IMAGE_DIR / name, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"  wrote images/{name}")


def _legend(axis) -> None:
    handle = axis.legend(loc="lower right", frameon=False, fontsize=9)
    for text in handle.get_texts():
        text.set_color(INK)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def plot_confusion_matrix(model, X_test, y_test) -> None:
    display = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Real", "Fake"],
        cmap="Purples",
        colorbar=False,
        values_format="d",
    )
    axis = display.ax_
    style_axis(axis, xlabel="Predicted label", ylabel="Actual label")
    axis.grid(False)
    _title(axis, "Confusion matrix")
    for text in axis.texts:
        text.set_fontsize(12)
    _save(display.figure_, "confusion_matrix.png")


def plot_confusion_matrix_grid(matrices: dict, title_suffix: str = "") -> None:
    """One confusion matrix per algorithm, side by side, for direct comparison."""
    names = list(matrices)
    if not names:
        return

    columns = 2
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, name in zip(axes, names):
        block = matrices[name]
        values = np.asarray(block["matrix"], dtype=int)
        display = ConfusionMatrixDisplay(confusion_matrix=values, display_labels=["Real", "Fake"])
        display.plot(ax=axis, cmap="Purples", colorbar=False, values_format="d")

        style_axis(axis, xlabel="Predicted", ylabel="Actual")
        axis.grid(False)
        total = values.sum()
        correct = np.trace(values)
        axis.set_title(
            f"{name}\naccuracy {correct / total:.4f}" if total else name,
            color=INK, fontsize=10.5, fontweight="bold", loc="left", pad=10,
        )
        for text in axis.texts:
            text.set_fontsize(11)

    for axis in axes[len(names):]:
        axis.set_visible(False)

    figure.suptitle(
        f"Confusion matrices on the held-out split{title_suffix}",
        color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left", y=1.02,
    )
    _save(figure, "confusion_matrices.png")


def plot_roc_curve(model, X_test, y_test) -> None:
    display = RocCurveDisplay.from_estimator(
        model, X_test, y_test, curve_kwargs={"color": PRIMARY, "linewidth": 2.2}
    )
    display.ax_.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.2, label="Chance level")
    style_axis(display.ax_, xlabel="False positive rate", ylabel="True positive rate", grid_axis="both")
    _legend(display.ax_)
    _title(display.ax_, "ROC curve for the selected model")
    _save(display.figure_, "roc_curve.png")


def plot_roc_comparison(metrics: dict, test_scores: dict[str, np.ndarray], y_test) -> None:
    """One ROC line per candidate, each quoted at its strongest representation."""
    figure, axis = plt.subplots(figsize=(6.6, 4.4))
    style_axis(axis, xlabel="False positive rate", ylabel="True positive rate", grid_axis="both")

    best = metrics["best_model_name"]

    for name, scores in test_scores.items():
        fpr, tpr, _ = roc_curve_points(y_test, scores)
        auc = metrics["models"].get(name, {}).get("roc_auc")
        if auc is None:
            continue
        axis.plot(
            fpr,
            tpr,
            linewidth=2.6 if name == best else 1.6,
            color=MODEL_COLORS.get(name, MUTED) if name == best else MUTED,
            alpha=1.0 if name == best else 0.85,
            label=f"{name} - AUC {auc:.3f}",
        )

    axis.plot([0, 1], [0, 1], "--", color=GRID, linewidth=1.2)
    _legend(axis)
    _title(axis, "ROC comparison across candidates")
    _save(figure, "roc_comparison.png")


def plot_metric_comparison(metrics: dict) -> None:
    names = list(metrics["models"])
    keys = ("accuracy", "precision", "recall", "f1_score")
    labels = ("Accuracy", "Precision", "Recall", "F1")
    colors = (PRIMARY, "#8B5CF6", "#0EA5E9", FAKE_COLOR)

    figure, axis = plt.subplots(figsize=(9.2, 4.4))
    style_axis(axis, ylabel="Score")
    axis.set_ylim(0.84, 1.008)
    axis.set_yticks([0.85, 0.9, 0.95, 1.0])

    width = 0.2
    positions = np.arange(len(names))

    for offset, (key, label, color) in enumerate(zip(keys, labels, colors)):
        values = [metrics["models"][name][key] for name in names]
        bars = axis.bar(
            positions + (offset - 1.5) * width,
            values,
            width * 0.92,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=7.5, color=MUTED, padding=2)

    axis.set_xticks(positions)
    axis.set_xticklabels([name.replace(" ", "\n") for name in names], fontsize=9, color=INK)

    handle = axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=9
    )
    for text in handle.get_texts():
        text.set_color(INK)

    _title(axis, "Benchmark scores on the held-out split")
    _save(figure, "metric_comparison.png")


def plot_representation_comparison(metrics: dict) -> None:
    """The 3 x 4 grid: every representation against every algorithm."""
    representations = metrics.get("representations")
    if not representations:
        return

    representation_names = list(representations)
    model_names = list(metrics["models"])

    figure, axis = plt.subplots(figsize=(9.6, 4.6))
    style_axis(axis, ylabel="F1 score on the hold-out split")

    width = 0.8 / len(model_names)
    positions = np.arange(len(representation_names))

    for index, model_name in enumerate(model_names):
        values = [
            representations[rep]["models"][model_name]["f1_score"] for rep in representation_names
        ]
        bars = axis.bar(
            positions + (index - (len(model_names) - 1) / 2) * width,
            values,
            width * 0.92,
            label=model_name,
            color=MODEL_COLORS.get(model_name, MUTED),
            edgecolor="white",
            linewidth=0.8,
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=7, color=MUTED, padding=2)

    axis.set_ylim(0.7, 1.005)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{rep}\n{representations[rep].get('dimensions', 0):,} dims"
            for rep in representation_names
        ],
        fontsize=9,
        color=INK,
    )

    handle = axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=9
    )
    for text in handle.get_texts():
        text.set_color(INK)

    _title(axis, "Representation x algorithm benchmark")
    _save(figure, "representation_comparison.png")


def plot_feature_importance(top_features: list[dict]) -> None:
    rows = list(reversed(top_features[:15]))
    if not rows:
        return

    figure, axis = plt.subplots(figsize=(7.8, 5.0))
    style_axis(axis, xlabel="Weight magnitude", grid_axis="x")

    axis.barh(
        [row["feature"] for row in rows],
        [abs(row["weight"]) for row in rows],
        color=[FAKE_COLOR if row["direction"] == "FAKE" else REAL_COLOR for row in rows],
        height=0.68,
    )
    axis.tick_params(axis="y", labelsize=8.5)
    _title(axis, "Terms that drive the decision")
    _save(figure, "feature_importance.png")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def candidate_test_scores(y_test) -> dict[str, np.ndarray]:
    """Positive-class scores on the hold-out split for every candidate model.

    Each algorithm is refitted on its own strongest representation -- the same
    choice recorded in ``results/model_metrics.json`` -- so the ROC comparison
    matches the benchmark table.
    """
    headline = load_metrics().get("models", {})

    train_documents = joblib.load(MODEL_DIR / "train_documents.pkl")
    test_documents = joblib.load(MODEL_DIR / "test_documents.pkl")
    y_train = joblib.load(YTRAIN)

    chosen: dict[str, str] = {}
    for name, scores in headline.items():
        chosen[name] = scores.get("representation", "TF-IDF")

    needed = sorted(set(chosen.values()))
    fitted = {}
    for representation_name in needed:
        representation = build_representations()[representation_name]
        representation.fit(train_documents)
        fitted[representation_name] = (
            representation.transform(train_documents),
            representation.transform(test_documents),
        )

    scores: dict[str, np.ndarray] = {}
    for name, representation_name in chosen.items():
        model = build_models()[name]
        X_train, X_test = fitted[representation_name]
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            scores[name] = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores[name] = model.decision_function(X_test)

    return scores


def load_metrics() -> dict:
    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"{METRICS_FILE} not found. Run `python -m src.train_models` first."
        )
    with open(METRICS_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    ensure_dirs()

    print("=" * 66)
    print("STAGE 5  |  EVALUATION REPORT")
    print("=" * 66)

    metrics = load_metrics()
    model = joblib.load(BEST_MODEL)
    X_test = joblib.load(XTEST)
    y_test = joblib.load(YTEST)
    feature_names = joblib.load(FEATURE_NAMES)

    predictions = model.predict(X_test)

    print(f"\nSelected model : {metrics['best_model_name']} "
          f"on {metrics['best_representation']} features")
    print("\nClassification report")
    print(classification_report(y_test, predictions, target_names=["Real", "Fake"], digits=4))

    print("\nConfusion matrix per algorithm")
    for name, block in metrics.get("confusion_matrices", {}).items():
        values = np.asarray(block["matrix"], dtype=float)
        total = values.sum()
        accuracy = float(np.trace(values) / total) if total else 0.0
        print(f"  {name:<20} {block['matrix']}  accuracy={accuracy:.4f}")

    print("\nFigures")
    plot_confusion_matrix(model, X_test, y_test)
    plot_confusion_matrix_grid(metrics.get("confusion_matrices", {}))
    plot_roc_curve(model, X_test, y_test)
    plot_metric_comparison(metrics)
    plot_representation_comparison(metrics)
    plot_feature_importance(metrics["best_model"]["top_features"])

    try:
        plot_roc_comparison(metrics, candidate_test_scores(y_test), y_test)
    except Exception as error:  # pragma: no cover - chart is non-critical
        print(f"  skipped roc_comparison.png ({error})")

    print(f"representation  : {metrics['features'].get('representation', 'n/a')}")
    print(f"columns         : {len(feature_names)}")
    print("=" * 66)


if __name__ == "__main__":
    main()
