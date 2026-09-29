"""Loading and health-checking of the trained artifacts.

The app degrades gracefully: :func:`artifact_status` reports exactly which
files are missing so the UI can show an actionable message instead of an
opaque traceback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import streamlit as st

from src.config import (
    BEST_MODEL,
    CLEAN_DATA,
    FEATURE_NAMES,
    IMAGE_DIR,
    METRICS_FILE,
    RAW_DATA,
    VECTORIZER,
)


@dataclass(frozen=True)
class ModelBundle:
    """Everything needed to score a document, loaded once per server."""

    model: object
    vectorizer: object
    feature_names: list[str]
    metrics: dict
    best_model_name: str
    generated_at: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def scores(self) -> dict:
        return self.metrics.get("models", {}).get(self.best_model_name, {})

    @property
    def representation(self) -> str:
        """Name of the feature representation the served model was fitted on."""
        return self.metrics.get("best_representation", "unknown")

    @property
    def representations(self) -> dict:
        return self.metrics.get("representations", {})

    @property
    def confusion_matrices(self) -> dict:
        return self.metrics.get("confusion_matrices", {})

    @property
    def per_class(self) -> dict:
        return self.metrics.get("per_class_metrics", {})

    @property
    def selection(self) -> dict:
        return self.metrics.get("selection", {})

    @property
    def vocabulary_size(self) -> int:
        return int(self.metrics.get("features", {}).get("vocabulary_size", len(self.feature_names)))

    @property
    def confusion_matrix(self) -> dict:
        return self.metrics.get("best_model", {}).get("confusion_matrix", {})

    @property
    def roc_curve(self) -> dict:
        return self.metrics.get("best_model", {}).get("roc_curve", {})

    @property
    def top_features(self) -> list[dict]:
        return self.metrics.get("best_model", {}).get("top_features", [])


REQUIRED_ARTIFACTS = {
    "Model weights": BEST_MODEL,
    "Feature representation": VECTORIZER,
    "Feature names": FEATURE_NAMES,
    "Benchmark metrics": METRICS_FILE,
}


def artifact_status() -> list[dict]:
    """Report on every artifact the app reads."""
    rows = [
        {"Artifact": label, "Path": str(path), "Present": path.exists()}
        for label, path in REQUIRED_ARTIFACTS.items()
    ]
    rows.append(
        {
            "Artifact": "Training corpus",
            "Path": str(CLEAN_DATA),
            "Present": CLEAN_DATA.exists(),
        }
    )
    return rows


def model_ready() -> bool:
    """True when every artifact required for inference is on disk."""
    return all(path.exists() for path in REQUIRED_ARTIFACTS.values())


@st.cache_resource(show_spinner="Loading the trained model...")
def load_bundle() -> ModelBundle:
    """Load (and memoise) the model, vectoriser and benchmark metadata."""
    import joblib

    metrics: dict = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE, encoding="utf-8") as handle:
            metrics = json.load(handle)

    return ModelBundle(
        model=joblib.load(BEST_MODEL),
        vectorizer=joblib.load(VECTORIZER),
        feature_names=list(joblib.load(FEATURE_NAMES)),
        metrics=metrics,
        best_model_name=metrics.get("best_model_name", "Unknown"),
        generated_at=metrics.get("generated_at"),
        extra={
            "features": metrics.get("features", {}),
            "split": metrics.get("split", {}),
            "dataset": metrics.get("dataset", {}),
            "representation": metrics.get("best_representation"),
        },
    )


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    """Read the benchmark file. Returns ``{}`` when it is absent."""
    if not METRICS_FILE.exists():
        return {}
    with open(METRICS_FILE, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the derived corpus that backs the dataset explorer.

    Falls back to the cleaned training frame when the engineered dataset has
    not been generated yet.
    """
    for path in (RAW_DATA, CLEAN_DATA):
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        "No dataset found. Run `python -m src.prepare_dataset` and "
        "`python -m src.preprocessing` from the project root."
    )


@st.cache_data(show_spinner=False)
def load_profile() -> dict:
    """Cached statistical profile produced by ``python -m src.eda``."""
    from src.config import DATASET_PROFILE

    if not DATASET_PROFILE.exists():
        return {}
    with open(DATASET_PROFILE, encoding="utf-8") as handle:
        return json.load(handle)


def figure_path(name: str):
    """Return a figure path when it exists, otherwise ``None``."""
    path = IMAGE_DIR / name
    return path if path.exists() else None


def trained_on() -> str:
    """Human-readable training timestamp."""
    bundle_metrics = load_metrics()
    stamp = bundle_metrics.get("generated_at")
    if not stamp:
        return "unknown"
    try:
        return datetime.fromisoformat(stamp).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        return stamp
