"""Inference, editorial-signal extraction and local explanations.

Three concerns, all pure functions over a :class:`ModelBundle`:

* :func:`analyze` -- score one document and collect everything the UI needs.
* :func:`analyze_frame` -- the same, vectorised, for a dataframe column.
* :func:`explain` -- per-document attribution: exact additive contributions for
  linear models, occlusion over the document's active terms for tree/kernel
  models. Both return a signed list of terms, so the UI can always say which
  way each term pushed the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils import clean_text, extract_article_metadata, text_statistics
from app.services.artifacts import ModelBundle

LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "Real", LABEL_FAKE: "Fake"}
LABEL_BADGES = {LABEL_REAL: "green", LABEL_FAKE: "red"}

#: Upper bound on terms probed by the occlusion explainer.
MAX_EXPLAINED_TERMS = 60

#: Predictions below this share of probability are flagged as low confidence.
LOW_CONFIDENCE_BAND = 80.0

#: A document shorter than this cannot be vectorised meaningfully.
MIN_WORDS_FOR_PREDICTION = 5


@dataclass
class Verdict:
    """Everything the UI needs to render one analysis."""

    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str
    statistics: dict
    signals: dict
    terms: list[dict] = field(default_factory=list)
    warning: str | None = None

    @property
    def is_real(self) -> bool:
        return self.label == LABEL_REAL

    @property
    def badge(self) -> str:
        return LABEL_BADGES[self.label]


@dataclass
class BatchResult:
    """Summary of a batch scoring run."""

    frame: pd.DataFrame
    total: int
    real: int
    fake: int
    mean_confidence: float
    low_confidence: int

    @property
    def fake_share(self) -> float:
        return self.fake / self.total if self.total else 0.0


# --------------------------------------------------------------------------
# Core scoring
# --------------------------------------------------------------------------

def _positive_scores(bundle: ModelBundle, matrix) -> np.ndarray:
    model = bundle.model
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, LABEL_FAKE])
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(matrix))
    return np.zeros(matrix.shape[0])


def _full_probabilities(bundle: ModelBundle, matrix) -> np.ndarray:
    """Return an ``(n, 2)`` array ordered ``[P(real), P(fake)]``."""
    model = bundle.model

    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix), dtype=float)

    fake = _positive_scores(bundle, matrix)
    fake = np.clip(fake, 0.0, 1.0)
    return np.column_stack([1.0 - fake, fake])


def analyze(
    bundle: ModelBundle,
    text: str,
    title: str = "",
    *,
    explain: bool = True,
) -> Verdict | None:
    """Score a single document.

    Returns ``None`` when the input is too short to be meaningful, so callers
    can render their own empty/validation state.
    """
    body = (text or "").strip()
    headline = (title or "").strip()
    combined = f"{headline}. {body}".strip() if headline else body

    if len(body.split()) < MIN_WORDS_FOR_PREDICTION:
        return None

    cleaned = clean_text(combined)
    if not cleaned:
        return Verdict(
            label=LABEL_REAL,
            label_name=LABEL_NAMES[LABEL_REAL],
            confidence=0.5,
            probabilities={"Real": 0.5, "Fake": 0.5},
            cleaned_text="",
            statistics=text_statistics(combined),
            signals=extract_article_metadata(combined, headline),
            warning=(
                "Almost every word was removed during cleaning "
                "(stopwords, punctuation or very short tokens). "
                "The prediction falls back to an even split."
            ),
        )

    vector = bundle.vectorizer.transform([cleaned])
    probabilities = _full_probabilities(bundle, vector)[0]
    label = int(bundle.model.predict(vector)[0])

    return Verdict(
        label=label,
        label_name=LABEL_NAMES[label],
        confidence=float(probabilities[label]),
        probabilities={
            LABEL_NAMES[LABEL_REAL]: float(probabilities[LABEL_REAL]),
            LABEL_NAMES[LABEL_FAKE]: float(probabilities[LABEL_FAKE]),
        },
        cleaned_text=cleaned,
        statistics=text_statistics(combined),
        signals=extract_article_metadata(combined, headline),
        terms=_explain(bundle, vector, label) if explain else [],
    )


def prepare_batch_frame(
    frame: pd.DataFrame,
    text_column: str,
    title_column: str | None = None,
    min_words: int = MIN_WORDS_FOR_PREDICTION,
) -> tuple[pd.DataFrame, int]:
    """Keep only the rows that contain enough text to classify.

    A CSV upload often contains blank rows, spacer rows or very short notes.
    Vectorising those rows produces an empty document and a meaningless
    prediction, so they are removed here and counted for the UI instead of
    being scored silently.

    Args:
        frame: The uploaded dataframe.
        text_column: Name of the column holding the article body.
        title_column: Optional headline column, prepended to the body.
        min_words: Minimum number of whitespace-separated words required.

    Returns:
        tuple: ``(usable_frame, skipped_count)`` with a fresh row index.

    Raises:
        KeyError: If ``text_column`` is not present in the frame.
    """
    if text_column not in frame.columns:
        raise KeyError(f"Text column '{text_column}' is not present in the uploaded file.")

    working = frame.copy()
    if title_column and title_column in working.columns:
        titles = working[title_column].fillna("").astype(str)
    else:
        titles = pd.Series("", index=working.index)

    combined = [
        f"{title}. {body}".strip() if title else str(body)
        for title, body in zip(titles, working[text_column].fillna("").astype(str))
    ]
    keep = np.array([len(document.split()) >= min_words for document in combined])

    usable = working.loc[keep].reset_index(drop=True)
    skipped = int((~keep).sum())
    return usable, skipped


def analyze_frame(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    text_column: str,
    title_column: str | None = None,
) -> BatchResult:
    """Score every row of ``frame`` in a single vectorised pass."""
    working = frame.copy()

    if title_column and title_column in working.columns:
        titles = working[title_column].fillna("").astype(str)
    else:
        titles = pd.Series("", index=working.index)

    combined = [
        f"{title}. {body}".strip() if title else str(body)
        for title, body in zip(titles, working[text_column].fillna("").astype(str))
    ]

    cleaned = [clean_text(document) for document in combined]
    matrix = bundle.vectorizer.transform(cleaned)
    predictions = bundle.model.predict(matrix)
    probabilities = _full_probabilities(bundle, matrix)

    working["_cleaned"] = cleaned
    working["Prediction"] = [LABEL_NAMES[int(value)] for value in predictions]
    working["Confidence"] = [round(float(row[value]) * 100, 2) for row, value in zip(probabilities, predictions)]
    working["Real score"] = [round(float(row[LABEL_REAL]) * 100, 2) for row in probabilities]
    working["Words"] = [len(document.split()) for document in combined]
    working["Verdict band"] = [
        "Low" if float(row[value]) * 100 < LOW_CONFIDENCE_BAND else "High"
        for row, value in zip(probabilities, predictions)
    ]

    result_columns = [column for column in working.columns if not column.startswith("_")]
    output = working[result_columns]

    return BatchResult(
        frame=output,
        total=len(output),
        real=int((predictions == LABEL_REAL).sum()),
        fake=int((predictions == LABEL_FAKE).sum()),
        mean_confidence=float(output["Confidence"].mean()) if len(output) else 0.0,
        low_confidence=int((output["Verdict band"] == "Low").sum()),
    )


# --------------------------------------------------------------------------
# Local explanation
# --------------------------------------------------------------------------

def _explain(bundle: ModelBundle, vector, label: int, top_n: int = 8) -> list[dict]:
    """Attribute the prediction to individual TF-IDF terms.

    Linear models give an exact additive decomposition. Every other model uses
    occlusion: each active term is zeroed out one at a time and the change in
    ``P(fake)`` is recorded. The sign of that change is the direction.
    """
    if vector.nnz == 0:
        return []

    if hasattr(bundle.model, "coef_"):
        return _explain_linear(bundle, vector, label, top_n)
    return _explain_occlusion(bundle, vector, top_n)


def _explain_linear(bundle: ModelBundle, vector, label: int, top_n: int) -> list[dict]:
    weights = np.asarray(bundle.model.coef_).ravel()
    row = sp.csr_matrix(vector)
    indices = row.indices
    values = row.data * weights[indices]

    fake_contribution = values if label == LABEL_FAKE else -values

    order = np.argsort(np.abs(fake_contribution))[::-1][:top_n]

    return [
        {
            "term": bundle.feature_names[index],
            "weight": float(fake_contribution[position]),
            "direction": LABEL_NAMES[LABEL_FAKE if fake_contribution[position] > 0 else LABEL_REAL],
        }
        for position, index in zip(order, indices[order])
    ]


def _explain_occlusion(bundle: ModelBundle, vector, top_n: int, budget: int = MAX_EXPLAINED_TERMS) -> list[dict]:
    row = sp.csr_matrix(vector)
    indices = np.asarray(row.indices)

    if len(indices) > budget:
        indices = indices[np.argsort(np.abs(row.data))[::-1][:budget]]

    baseline = float(_positive_scores(bundle, row)[0])

    # One copy of the document per candidate term, each with that term removed.
    variants = sp.vstack([row] * len(indices)).tocsr()
    for position, index in enumerate(indices):
        variants[position, index] = 0.0

    deltas = baseline - np.asarray(_positive_scores(bundle, variants))
    order = np.argsort(np.abs(deltas))[::-1][:top_n]

    return [
        {
            "term": bundle.feature_names[int(indices[position])],
            "weight": float(deltas[position]),
            "direction": LABEL_NAMES[LABEL_FAKE if deltas[position] > 0 else LABEL_REAL],
        }
        for position in order
    ]


def strongest_corpus_terms(bundle: ModelBundle, dataset: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Terms that appear most often in each class of the training corpus."""
    if "clean_text" not in dataset.columns or "label" not in dataset.columns:
        return pd.DataFrame(columns=["term", "Real", "Fake"])

    rows = []
    for label, name in ((LABEL_REAL, "Real"), (LABEL_FAKE, "Fake")):
        documents = dataset.loc[dataset["label"] == label, "clean_text"].fillna("").astype(str)
        counts: dict[str, int] = {}
        for document in documents:
            for token in set(document.split()):
                counts[token] = counts.get(token, 0) + 1
        rows.append(pd.Series(counts, name=name).sort_values(ascending=False).head(top_n * 3))

    combined = pd.concat(rows, axis=1).fillna(0).astype(int)
    combined.index.name = "term"

    real_rank = combined["Real"].rank(ascending=False)
    fake_rank = combined["Fake"].rank(ascending=False)
    combined["Real rank"] = real_rank
    combined["Fake rank"] = fake_rank

    return combined.sort_values(["Fake rank"]).head(top_n).reset_index()
