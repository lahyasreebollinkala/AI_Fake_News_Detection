"""The three representations must fit, transform and label their own features.

The selection and scoring tests read the recorded benchmark; these build small
representations from scratch, so a refactor that breaks the fit/transform
contract fails here rather than silently producing a broken ``vectorizer.pkl``.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src.config import REPRESENTATIONS
from src.feature_extraction import build_representations

DOCUMENTS = [
    "senate committee approves the budget amendment on tuesday evening",
    "the committee approved a budget amendment after debate in the senate",
    "shocking truth you wont believe what the officials did next",
    "you wont believe this shocking truth about the officials",
    "a new study reports lower unemployment across the region this year",
    "regional unemployment fell this year according to a new study",
] * 4
LABELS = [0, 0, 1, 1, 0, 0] * 4

SMALL = {"n_components": 8}


def _fit(name: str):
    representation = build_representations()[name]
    if name == "Embeddings (LSA)":
        representation = build_representations()[name]
        representation.n_components = 8
    return representation, representation.fit_transform(DOCUMENTS)


@pytest.fixture(params=list(REPRESENTATIONS))
def fitted(request):
    return _fit(request.param)


def test_build_representations_returns_all_three():
    built = build_representations()
    assert set(built) == set(REPRESENTATIONS)
    for name, representation in built.items():
        assert representation.name == name


def test_fit_transform_shape_matches_the_described_dimensions(fitted):
    representation, matrix = fitted
    described = representation.describe()
    assert matrix.shape[0] == len(DOCUMENTS)
    assert matrix.shape[1] == described["dimensions"], (
        f"{representation.name} produced {matrix.shape[1]} columns but describes "
        f"{described['dimensions']}"
    )
    values = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    assert np.isfinite(values).all(), f"{representation.name} produced non-finite values"


def test_transform_does_not_refit(fitted):
    """A second transform must not change the fitted vocabulary -- that is the
    whole reason the fitted object is persisted for serving."""
    representation, matrix = fitted
    before = matrix.shape
    again = representation.transform(DOCUMENTS)
    assert again.shape == before
    if hasattr(matrix, "toarray"):
        assert np.allclose(matrix.toarray(), again.toarray())


def test_unseen_documents_transform_without_changing_the_vocabulary(fitted):
    representation, matrix = fitted
    size = matrix.shape[1]
    out = representation.transform(["a completely unseen article about fiscal policy reform"])
    assert out.shape[0] == 1
    assert out.shape[1] == size, "Transforming new text must not grow or shrink the vocabulary"


def test_feature_names_are_unique_and_descriptive(fitted):
    representation, matrix = fitted
    names = list(representation.feature_names())
    assert len(names) == matrix.shape[1]
    assert len(set(names)) == len(names), "Duplicate feature names"
    assert all(name.strip() for name in names), "Feature names must be non-empty"

    if representation.name == "Embeddings (LSA)":
        assert all(
            any(character.isalpha() for character in name) for name in names
        ), "Each latent dimension must be named after the terms that load on it"
    else:
        assert all(
            len(name.split()) <= 2 for name in names
        ), "Bag-of-Words features are unigrams and bigrams"


def test_sparsity_is_declared_honestly(fitted):
    representation, matrix = fitted
    assert representation.is_dense == (not hasattr(matrix, "toarray"))
    assert representation.describe()["sparse"] is (not representation.is_dense)


def test_embeddings_are_l2_normalised(fitted):
    """Cosine-similarity embeddings require unit-norm rows."""
    representation, matrix = fitted
    if representation.name != "Embeddings (LSA)":
        pytest.skip("only defined for the embedding representation")
    values = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    norms = np.linalg.norm(values, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), f"Row norms are {norms}"


@pytest.mark.parametrize(
    "estimator",
    [
        KNeighborsClassifier(n_neighbors=3),
        LogisticRegression(max_iter=500),
        RandomForestClassifier(n_estimators=10, random_state=42),
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=120, random_state=42),
    ],
    ids=["knn", "logreg", "random_forest", "mlp"],
)
def test_every_algorithm_trains_on_every_representation(estimator):
    """The benchmark claims all four algorithms score on all three
    representations; verify the claim on a miniature corpus."""
    for name in REPRESENTATIONS:
        representation, matrix = _fit(name)
        dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        estimator.fit(dense, LABELS)
        assert estimator.predict(dense).shape == (len(DOCUMENTS),), (
            f"{type(estimator).__name__} could not score {name}"
        )
