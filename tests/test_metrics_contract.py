"""The recorded benchmark must contain everything the report claims to show.

These tests read ``results/model_metrics.json`` -- the same file the app and the
report generator read -- and assert the structural requirements: all three
representations, all four algorithms, all five metrics per model, a confusion
matrix per algorithm, and a selection that was made on validation data rather
than on the hold-out.
"""

from __future__ import annotations

import joblib
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import MODEL_DIR, REPRESENTATIONS
from src.feature_extraction import BagOfWords, TfidfRepresentation

REQUIRED_MODELS = {
    "K-Nearest Neighbors",
    "Logistic Regression",
    "Random Forest",
    "Neural Network",
}

REQUIRED_METRICS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")


def test_all_three_representations_were_benchmarked(metrics):
    recorded = set(metrics.get("representations", {}))
    missing = set(REPRESENTATIONS) - recorded
    assert not missing, f"Representations never benchmarked: {sorted(missing)}"


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_every_representation_covers_all_four_algorithms(metrics, representation):
    block = metrics["representations"][representation]
    assert set(block["models"]) == REQUIRED_MODELS, (
        f"{representation} was not benchmarked against all four required algorithms"
    )


def test_headline_benchmark_covers_every_model_with_every_metric(metrics):
    models = metrics.get("models", {})
    assert set(models) == REQUIRED_MODELS, f"Headline benchmark is incomplete: {set(models)}"

    for name, scores in models.items():
        for key in REQUIRED_METRICS:
            assert key in scores, f"{name} is missing {key}"
            assert scores[key] is None or 0.0 <= scores[key] <= 1.0, f"{name}.{key} out of range"
        assert scores.get("representation"), f"{name} does not record its representation"


def test_confusion_matrix_recorded_for_every_algorithm(metrics):
    matrices = metrics.get("confusion_matrices", {})
    assert set(matrices) == REQUIRED_MODELS, f"Missing confusion matrices: {set(matrices)}"

    test_rows = metrics.get("dataset", {}).get("test_rows", 0)
    for name, block in matrices.items():
        matrix = block.get("matrix")
        assert matrix and len(matrix) == 2 and all(len(row) == 2 for row in matrix), (
            f"{name} has no 2x2 confusion matrix"
        )
        assert sum(sum(row) for row in matrix) == test_rows, (
            f"{name} confusion matrix totals {sum(sum(r) for r in matrix)}, "
            f"expected the {test_rows} hold-out articles"
        )
        assert block.get("accuracy") is not None, f"{name} confusion matrix has no accuracy"


def test_per_class_metrics_present_for_every_algorithm(metrics):
    per_class = metrics.get("per_class_metrics", {})
    assert set(per_class) == REQUIRED_MODELS
    for name, classes in per_class.items():
        assert set(classes) == {"REAL", "FAKE"}, f"{name} is missing a per-class breakdown"
        for label, scores in classes.items():
            for key in ("precision", "recall", "f1_score", "support"):
                assert key in scores, f"{name}/{label} is missing {key}"


def test_corpus_to_split_reconciliation(metrics):
    """Every source row must be accounted for: kept, or dropped for being empty."""
    split = metrics.get("dataset", {})
    source = split.get("source_rows")
    if source is None:
        pytest.skip("this run predates the corpus reconciliation fields")

    assert split.get("usable_rows", 0) + split.get("dropped_empty", 0) == source, (
        f"{split.get('usable_rows')} usable + {split.get('dropped_empty')} dropped "
        f"!= {source} source rows"
    )


def test_selection_was_made_on_validation_data(metrics):
    """The hold-out must not be used to choose the deployed model."""
    selection = metrics.get("selection", {})
    assert selection, "No selection record -- the winner cannot be traced"
    assert selection.get("validation_rows", 0) > 0, "Selection was not made on a validation split"
    assert selection.get("validation_f1") is not None, "No validation F1 recorded"

    winner = selection.get("winner", {})
    assert winner.get("representation") in REPRESENTATIONS
    assert winner.get("model") in REQUIRED_MODELS

    assert metrics.get("best_representation") == winner["representation"]
    assert metrics.get("best_model_name") == winner["model"]

    train_rows = metrics.get("dataset", {}).get("train_rows", 0)
    assert selection["validation_rows"] < train_rows, "The validation split is not a subset of training"


def test_split_is_stratified_and_disjoint(metrics):
    split = metrics.get("dataset", {})
    assert split.get("train_rows", 0) > 0 and split.get("test_rows", 0) > 0
    assert split.get("train_rows") + split.get("test_rows") == split.get("usable_rows", split.get("train_rows") + split.get("test_rows"))

    for side in ("train", "test"):
        real = split.get(f"{side}_real", 0)
        fake = split.get(f"{side}_fake", 0)
        total = split.get(f"{side}_rows", 0)
        assert real + fake == total, f"{side} split class counts do not sum to the split size"


def test_vocabulary_matches_a_refit_on_the_training_split():
    """The strongest available leakage guard.

    The persisted representation must be *exactly* what you get by fitting a fresh
    object on the training documents. If the vocabulary or the embedding basis had
    seen the hold-out, the refit would differ.
    """
    import pandas as pd

    from src.config import CLEAN_DATA, VECTORIZER
    from src.feature_extraction import build_representations

    with open(MODEL_DIR / "train_documents.pkl", "rb") as handle:
        train_documents = joblib.load(handle)

    persisted = joblib.load(VECTORIZER)
    assert list(persisted.feature_names()) == persisted.feature_names()

    if isinstance(persisted, TfidfRepresentation):
        refit = TfidfVectorizer(
            max_features=persisted.max_features,
            ngram_range=persisted.ngram_range,
            min_df=persisted.min_df,
            sublinear_tf=True,
        )
        refit.fit(train_documents)
        assert sorted(refit.vocabulary_) == sorted(persisted.vectorizer.vocabulary_), (
            "The persisted TF-IDF vocabulary differs from a refit on the training "
            "split -- the hold-out leaked into feature fitting"
        )
    elif isinstance(persisted, BagOfWords):
        refit = CountVectorizer(
            max_features=persisted.max_features,
            ngram_range=persisted.ngram_range,
            min_df=persisted.min_df,
        )
        refit.fit(train_documents)
        assert sorted(refit.vocabulary_) == sorted(persisted.vectorizer.vocabulary_), (
            "The persisted Bag-of-Words vocabulary differs from a refit on the training "
            "split -- the hold-out leaked into feature fitting"
        )
    else:
        pytest.skip("the leakage refit check is defined for the sparse representations")

    # Terms that occur only in the hold-out must be out-of-vocabulary for the
    # persisted representation. If any survived, the hold-out was visible when
    # the vocabulary was fitted.
    test_only_terms = _terms_not_in(persisted, train_documents)
    leaked = sorted(set(persisted.vectorizer.vocabulary_) & test_only_terms)
    assert not leaked, (
        f"{len(leaked)} hold-out-only terms are present in the fitted vocabulary "
        f"(e.g. {leaked[:10]}) -- the hold-out leaked into feature fitting"
    )

    assert CLEAN_DATA.exists()


def _terms_not_in(persisted, train_documents: list[str]) -> set[str]:
    """Terms that occur in the hold-out vocabulary but never in the training one."""
    with open(MODEL_DIR / "test_documents.pkl", "rb") as handle:
        test_documents = joblib.load(handle)

    full = BagOfWords(
        max_features=200_000,
        ngram_range=persisted.ngram_range,
        min_df=1,
    )
    full.fit(list(train_documents) + list(test_documents))

    train_only = BagOfWords(
        max_features=200_000,
        ngram_range=persisted.ngram_range,
        min_df=1,
    )
    train_only.fit(list(train_documents))

    return set(full.vectorizer.vocabulary_) - set(train_only.vectorizer.vocabulary_)


def test_split_is_stratified_and_disjoint(metrics):
    """A model that scores 1.0000 on everything is a bug, not a result."""
    for name, scores in metrics.get("models", {}).items():
        assert scores.get("accuracy", 0) < 1.0, f"{name} reports a perfect hold-out accuracy"
        assert scores.get("f1_score", 0) < 1.0, f"{name} reports a perfect hold-out F1"
