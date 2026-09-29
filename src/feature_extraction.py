"""Stage 3 -- the three required feature-extraction representations.

The assignment requires three ways of turning text into numbers:

===================  ==========================================================
Bag-of-Words         :class:`BagOfWords` -- raw term counts (``CountVectorizer``)
TF-IDF               :class:`TfidfRepresentation` -- tf x idf weighting
Embeddings           :class:`LsaEmbeddings` -- dense 200-d latent vectors built
                      by truncated SVD over the TF-IDF matrix (latent semantic
                      analysis), L2-normalised so Euclidean distance behaves
                      like cosine similarity.
===================  ==========================================================

Every representation is fitted **only on the training split** and then applied to
the hold-out split, so neither the vocabulary, the IDF weights nor the embedding
basis is ever informed by the evaluation data.

Each class exposes the same three-part contract used by the rest of the project:

``fit_transform(documents)``
    Learn the representation from *training* documents.
``transform(documents)``
    Project new documents into the already-fitted space.
``feature_names()``
    Human-readable column labels, used by the app's term-attribution panel and
    by the evaluation figures.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer

from src.config import (
    EMBEDDING_COMPONENTS,
    EMBEDDING_SEED,
    MAX_FEATURES,
    MIN_DF,
    NGRAM_RANGE,
)

#: How many vocabulary terms describe one embedding dimension.
TERMS_PER_COMPONENT = 3


class TextRepresentation:
    """Base class defining the fit / transform / describe contract."""

    name = "Representation"

    def fit(self, documents) -> "TextRepresentation":
        raise NotImplementedError

    def transform(self, documents):
        raise NotImplementedError

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)

    def feature_names(self) -> list[str]:
        return []

    def describe(self) -> dict:
        return {"name": self.name}

    @property
    def is_dense(self) -> bool:
        return False


# --------------------------------------------------------------------------
# 1. Bag-of-Words
# --------------------------------------------------------------------------

class BagOfWords(TextRepresentation):
    """Raw term counts -- the most basic (and most interpretable) encoding.

    A document becomes a row of integer counts, one column per vocabulary term.
    No weighting is applied, so frequent terms dominate the representation; that
    is precisely the weakness TF-IDF was designed to remove, and benchmarking
    both makes the difference visible.
    """

    name = "Bag-of-Words"

    def __init__(self, max_features: int = MAX_FEATURES,
                 ngram_range: tuple[int, int] = NGRAM_RANGE, min_df: int = MIN_DF) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            strip_accents="unicode",
        )

    def fit(self, documents) -> "BagOfWords":
        self.vectorizer.fit(documents)
        return self

    def transform(self, documents):
        return self.vectorizer.transform(documents)

    def feature_names(self) -> list[str]:
        if not hasattr(self.vectorizer, "vocabulary_"):
            return []
        return [str(term) for term in self.vectorizer.get_feature_names_out()]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "method": "CountVectorizer",
            "weighting": "raw term counts",
            "sparse": True,
            "dimensions": len(self.feature_names()),
        }


# --------------------------------------------------------------------------
# 2. TF-IDF
# --------------------------------------------------------------------------

class TfidfRepresentation(TextRepresentation):
    """Term Frequency x Inverse Document Frequency over unigrams and bigrams.

    Every term is scaled by how rare it is across the corpus, so boilerplate
    ("said", "reported") is damped and discriminative vocabulary survives.
    ``sublinear_tf`` additionally uses ``1 + log(tf)`` so that a term repeated
    fifty times does not swamp the representation.
    """

    name = "TF-IDF"

    def __init__(self, max_features: int = MAX_FEATURES,
                 ngram_range: tuple[int, int] = NGRAM_RANGE, min_df: int = MIN_DF) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            strip_accents="unicode",
        )

    def fit(self, documents) -> "TfidfRepresentation":
        self.vectorizer.fit(documents)
        return self

    def transform(self, documents):
        return self.vectorizer.transform(documents)

    def feature_names(self) -> list[str]:
        if not hasattr(self.vectorizer, "vocabulary_"):
            return []
        return [str(term) for term in self.vectorizer.get_feature_names_out()]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "method": "TfidfVectorizer(sublinear_tf=True)",
            "weighting": "tf x idf, sublinear term frequency",
            "sparse": True,
            "dimensions": len(self.feature_names()),
        }


# --------------------------------------------------------------------------
# 3. Embeddings
# --------------------------------------------------------------------------

class LsaEmbeddings(TextRepresentation):
    """Dense sentence embeddings via latent semantic analysis.

    A word co-occurrence matrix has one axis per term and is far too wide to
    feed to a distance-based model. Truncated SVD compresses the TF-IDF matrix
    to ``n_components`` orthogonal axes, each representing a *topic*; the
    resulting dense row is a fixed-length vector in which documents about the
    same subject sit close together even when they share no vocabulary. That is
    the defining property of a sentence embedding, obtained here without a GPU,
    a pretrained checkpoint or an extra dependency.

    Each dimension is labelled by its strongest loading terms so the app can
    still show *what* an embedding component represents.
    """

    name = "Embeddings (LSA)"

    def __init__(self, n_components: int = EMBEDDING_COMPONENTS,
                 max_features: int = MAX_FEATURES,
                 ngram_range: tuple[int, int] = NGRAM_RANGE,
                 min_df: int = MIN_DF,
                 random_state: int = EMBEDDING_SEED) -> None:
        self.requested_components = n_components
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.random_state = random_state
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.svd: TruncatedSVD | None = None
        self.normalizer: Normalizer | None = None
        self._component_terms: list[str] = []

    # -- fitting ------------------------------------------------------------

    def fit(self, documents) -> "LsaEmbeddings":
        tfidf = self.vectorizer.fit_transform(documents)
        self.n_components = self._safe_rank(tfidf)

        self.svd = TruncatedSVD(
            n_components=self.n_components,
            n_iter=12,
            random_state=self.random_state,
        )
        self.svd.fit(tfidf)
        self.normalizer = Normalizer(copy=False)
        self.normalizer.fit(self.svd.transform(tfidf))
        self._component_terms = self._label_components()
        return self

    def _safe_rank(self, matrix) -> int:
        """SVD cannot exceed the smaller of the two matrix dimensions."""
        limit = min(matrix.shape) - 1
        return max(2, min(self.requested_components, limit))

    def _label_components(self) -> list[str]:
        """Name each latent axis after the vocabulary terms that load on it."""
        if self.svd is None:
            return []

        terms = np.asarray(self.vectorizer.get_feature_names_out())
        components = self.svd.components_

        labels = []
        for row in components:
            top = np.argsort(np.abs(row))[::-1][:TERMS_PER_COMPONENT]
            labels.append(" + ".join(str(terms[index]) for index in top))
        return labels

    # -- transform ----------------------------------------------------------

    def transform(self, documents) -> np.ndarray:
        if self.svd is None or self.normalizer is None:
            raise RuntimeError("LsaEmbeddings.transform() called before fit().")
        return self.normalizer.transform(self.svd.transform(self.vectorizer.transform(documents)))

    def feature_names(self) -> list[str]:
        return list(self._component_terms)

    @property
    def is_dense(self) -> bool:
        return True

    def explain_variance(self) -> float:
        """Share of the TF-IDF variance captured by the retained components."""
        if self.svd is None:
            return 0.0
        return float(self.svd.explained_variance_ratio_.sum())

    def describe(self) -> dict:
        return {
            "name": self.name,
            "method": "TfidfVectorizer -> TruncatedSVD -> L2 normalise",
            "weighting": "dense latent topic coordinates (cosine-scaled)",
            "sparse": False,
            "dimensions": self.n_components,
            "requested_dimensions": self.requested_components,
            "explained_variance": round(self.explain_variance(), 4),
        }


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

#: Canonical name -> constructor. The order is the order they appear in reports.
REPRESENTATION_BUILDERS = {
    BagOfWords.name: BagOfWords,
    TfidfRepresentation.name: TfidfRepresentation,
    LsaEmbeddings.name: LsaEmbeddings,
}


def build_representations() -> dict[str, TextRepresentation]:
    """One unfitted instance of each required representation."""
    return {name: builder() for name, builder in REPRESENTATION_BUILDERS.items()}


def as_dense(matrix) -> np.ndarray:
    """Coerce a feature matrix to a dense array (embeddings are already dense)."""
    if sp.issparse(matrix):
        return np.asarray(matrix.todense())
    return np.asarray(matrix)
