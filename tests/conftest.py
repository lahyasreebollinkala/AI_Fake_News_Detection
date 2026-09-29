"""Shared pytest fixtures for the Veritas test-suite.

The suite assumes the pipeline has already been built (see ``README.md``).
Artifacts are asserted in :func:`pytest_configure` so a missing pipeline fails
loudly instead of silently skipping the ML assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import BEST_MODEL, CLEAN_DATA, FEATURE_NAMES, METRICS_FILE, VECTORIZER  # noqa: E402

REQUIRED_ARTIFACTS = (CLEAN_DATA, BEST_MODEL, VECTORIZER, FEATURE_NAMES, METRICS_FILE)


def pytest_configure(config: pytest.Config) -> None:
    missing = [str(path.name) for path in REQUIRED_ARTIFACTS if not path.exists()]
    if missing:
        raise pytest.UsageError(
            "Missing pipeline artifacts: "
            + ", ".join(missing)
            + ". Run: python -m src.prepare_dataset, python -m src.preprocessing, "
            "python -m src.feature_engineering, python -m src.train_models"
        )


@pytest.fixture(scope="session")
def metrics() -> dict:
    import json

    with open(METRICS_FILE, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def bundle():
    from app.services.artifacts import ModelBundle
    import joblib

    return ModelBundle(
        model=joblib.load(BEST_MODEL),
        vectorizer=joblib.load(VECTORIZER),
        feature_names=list(joblib.load(FEATURE_NAMES)),
        metrics={},
        best_model_name="test",
    )


@pytest.fixture(scope="session")
def app_test():
    from streamlit.testing.v1 import AppTest

    def _run(relative_path: str):
        return AppTest.from_file(str(ROOT_DIR / relative_path), default_timeout=240).run()

    return _run
