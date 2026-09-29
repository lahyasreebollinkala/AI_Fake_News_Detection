"""Dense document embeddings, implemented from scratch in NumPy.

This module is a **reference implementation**, not part of the served pipeline.
The deployed embedding is :class:`src.feature_extraction.LsaEmbeddings`, which
shares the same scikit-learn transformer protocol and is persisted alongside the
trained model. Everything here is covered by ``tests/test_embeddings.py`` and
kept because it answers a question the report raises but the benchmark does not
settle: is a *neural* embedding better than a *factorisation* one on a corpus
this small?

Two techniques are provided:

``Word2VecEmbedder``
    Skip-gram with negative sampling (SGNS), written directly against NumPy.
    A context window is sampled around every token, the centre word predicts its
    neighbours, and ``negative_samples`` corrupted tokens supply the
    unigram^0.75 noise distribution. This is the same objective Mikolov et al.
    introduced; the implementation here is a compact NumPy version rather than
    the C++ original, because ``gensim`` has no wheel for CPython 3.14.

``LSAEmbedder``
    Latent Semantic Analysis. A TF-IDF matrix is factorised with a truncated
    SVD so that each document becomes a point in a dense, low-dimensional
    semantic space. Unlike Word2Vec this is a corpus-statistics method, not a
    neural one, which is exactly why comparing the two is informative. It is the
    method the deployed representation uses; ``feature_extraction.LsaEmbeddings``
    wraps the same computation and adds the interpretable per-axis term labels
    and the ``describe()`` contract the rest of the pipeline relies on.

Both classes follow the scikit-learn transformer protocol -- ``fit``,
``transform``, ``fit_transform`` -- so they drop into the same pipeline as the
BoW and TF-IDF vectorisers and are persisted with ``joblib`` like everything
else.

Fitting is always performed on the **training split only** by the caller. That
is what keeps the benchmark honest: an embedding vector for a test document must
not be influenced by the test document itself.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

#: Word2Vec hyper-parameters, tuned on this corpus. The vocabulary is small
#: (about 2k documents, 458k tokens) and far below the scale word2vec was
#: designed for, so the settings are deliberately conservative: a long linear
#: learning-rate decay and enough epochs to separate related words, without so
#: many that every vector drifts toward a common direction and cosine
#: similarity collapses (the "hubness" failure mode seen on small corpora).
EMBEDDING_DIM = 100
WINDOW_SIZE = 5
MIN_COUNT = 2
NEGATIVE_SAMPLES = 5
EPOCHS = 20
LEARNING_RATE = 0.025
INIT_SCALE = 0.1
SUBSAMPLE_THRESHOLD = 1e-4


class Word2VecEmbedder:
    """Skip-gram with negative sampling, trained on the token stream.

    Parameters mirror the vocabulary of the original paper so the method is
    recognisable: ``vector_size``, ``window``, ``min_count``,
    ``negative``, ``epochs``, ``sg`` is always 1 (skip-gram) and ``seed`` keeps
    the negative sampling reproducible.
    """

    name = "Word2Vec"

    def __init__(
        self,
        vector_size: int = EMBEDDING_DIM,
        window: int = WINDOW_SIZE,
        min_count: int = MIN_COUNT,
        negative: int = NEGATIVE_SAMPLES,
        epochs: int = EPOCHS,
        learning_rate: float = LEARNING_RATE,
        subsample: float = SUBSAMPLE_THRESHOLD,
        init_scale: float = INIT_SCALE,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.init_scale = init_scale
        self.seed = seed

        self.vocabulary_: dict[str, int] = {}
        self.inverse_vocabulary_: list[str] = []
        self.vectors_: np.ndarray | None = None
        self.context_vectors_: np.ndarray | None = None

    # -- vocabulary ---------------------------------------------------------

    def _build_vocabulary(self, documents: list[list[str]]) -> None:
        """Count tokens, drop rare ones, and freeze the index mapping."""
        counts: dict[str, int] = {}
        for tokens in documents:
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

        # Sort by descending frequency, then alphabetically, so the vocabulary
        # -- and therefore every downstream artifact -- is fully deterministic.
        kept = sorted(
            (token for token, count in counts.items() if count >= self.min_count),
            key=lambda token: (-counts[token], token),
        )

        self.inverse_vocabulary_ = kept
        self.vocabulary_ = {token: index for index, token in enumerate(kept)}

    def _subsampled_stream(self, token_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Frequent-word subsampling (Mikolov & Mikolov, 2013).

        A token of frequency ``f`` is dropped with probability
        ``1 - (sqrt(f / s) + 1) * (s / f)``, where ``s`` is ``subsample``.
        This stops words like "the" from dominating every context window.
        """
        if token_ids.size == 0:
            return token_ids

        frequency = np.bincount(token_ids, minlength=len(self.inverse_vocabulary_))
        total = max(token_ids.size, 1)
        ratio = frequency / total

        threshold = self.subsample
        with np.errstate(divide="ignore", invalid="ignore"):
            keep_probability = np.sqrt(ratio / threshold) + 1.0
            keep_probability = (threshold / np.maximum(ratio, 1e-12)) * keep_probability

        keep_probability = np.clip(keep_probability, 0.0, 1.0)
        draws = rng.random(token_ids.size)
        return token_ids[draws < keep_probability[token_ids]]

    # -- training -----------------------------------------------------------

    def _training_pairs(self, token_ids: np.ndarray, rng: np.random.Generator) -> tuple:
        """Build (centre, context) pairs from a dynamically sized window.

        The original paper shrinks the window for large documents so that very
        long articles do not generate quadratic numbers of pairs. The same
        ``b + w'`` dynamic window is reproduced here.

        The window is symmetric, so it is unrolled offset by offset rather than
        looping over positions: offset ``+d`` pairs ``(i, i+d)`` and offset
        ``-d`` pairs ``(i, i-d)``, which is identical to a Python loop but
        vectorised.
        """
        if token_ids.size < 2:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        effective_window = min(
            self.window,
            max(1, int(round(token_ids.size / 100 * self.window))),
        )

        centres: list[np.ndarray] = []
        contexts: list[np.ndarray] = []

        for offset in range(1, effective_window + 1):
            forward_centres = token_ids[: token_ids.size - offset]
            forward_contexts = token_ids[offset:]

            backward_centres = token_ids[offset:]
            backward_contexts = token_ids[: token_ids.size - offset]

            centres.append(forward_centres)
            contexts.append(forward_contexts)
            centres.append(backward_centres)
            contexts.append(backward_contexts)

        if not centres:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        return np.concatenate(centres), np.concatenate(contexts)

    def _noise_distribution(self, token_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample negative tokens from the unigram^0.75 distribution."""
        if token_ids.size == 0:
            return np.empty((0, self.negative), dtype=np.int64)

        counts = np.bincount(token_ids, minlength=len(self.inverse_vocabulary_)).astype(np.float64)
        noise = np.power(counts, 0.75)
        total = noise.sum()
        if total <= 0:
            return rng.integers(0, len(self.inverse_vocabulary_), size=(token_ids.size, self.negative))

        probabilities = noise / total
        return rng.choice(
            len(self.inverse_vocabulary_),
            size=(token_ids.size, self.negative),
            p=probabilities,
        )

    def fit(self, documents: list[list[str]]) -> "Word2VecEmbedder":
        """Train the skip-gram vectors on ``documents``.

        One SGD pass is made per epoch over every document, so the embedding
        quality improves with ``epochs`` at a linear cost.
        """
        if not documents:
            raise ValueError("Word2VecEmbedder.fit received no documents.")

        self._build_vocabulary(documents)
        size = len(self.vocabulary_)
        if size == 0:
            raise ValueError(
                "No token reached min_count="
                f"{self.min_count}; the corpus is too small to embed."
            )

        rng = np.random.default_rng(self.seed)

        # Init scale matters more than it looks. A dot product of two
        # N(0, s) vectors in d dimensions has standard deviation s^2 * sqrt(d).
        # Too large and the logistic function saturates before training, so
        # positive pairs get almost no gradient; too small and every word
        # receives an identical update, the embeddings collapse onto a single
        # direction, and random pairs end up at cosine 1.0. 0.05 keeps initial
        # scores near the linear region of the sigmoid while still breaking
        # symmetry between words.
        self.vectors_ = rng.normal(0.0, self.init_scale, size=(size, self.vector_size))
        self.context_vectors_ = np.zeros((size, self.vector_size), dtype=np.float64)

        encoded = [
            np.asarray([self.vocabulary_[token] for token in tokens if token in self.vocabulary_], dtype=np.int64)
            for tokens in documents
        ]
        encoded = [stream for stream in encoded if stream.size > 0]

        history: list[float] = []

        for epoch in range(self.epochs):
            # Linearly decaying learning rate, as in the original.
            alpha = self.learning_rate * (1.0 - epoch / max(self.epochs, 1))
            epoch_loss = 0.0
            pairs_seen = 0

            order = rng.permutation(len(encoded))
            for index in order:
                stream = self._subsampled_stream(encoded[index], rng)
                centres, contexts = self._training_pairs(stream, rng)
                if centres.size == 0:
                    continue

                noise = self._noise_distribution(centres, rng)
                loss = self._sgns_step(centres, contexts, noise, alpha)
                epoch_loss += loss
                pairs_seen += centres.size

            history.append(epoch_loss / max(pairs_seen, 1))

        self.training_loss_ = history
        return self

    def _sgns_step(
        self,
        centres: np.ndarray,
        contexts: np.ndarray,
        noise: np.ndarray,
        alpha: float,
    ) -> float:
        """One vectorised SGD step over every (centre, context) pair.

        The objective to *maximise* is the binary logistic score of one positive
        context word against ``negative`` sampled negatives::

            J = log sigma(v_c . v_w) + sum_k log sigma(-v_neg_k . v_c)

        The implementation minimises ``L = -J``. Writing
        ``sig(x) = 1 / (1 + exp(-x))`` and differentiating each log-sigmoid
        term gives::

            d/ds [-log sig(s)]      =  sig(s) - 1
            d/ds [-log sig(-s)]     =  sig(s)

        so with ``s_pos = v_c . v_w`` and ``s_neg_k = v_c . v_neg_k``::

            dL/dv_c     = sum_pos (sig(s_pos) - 1) * v_w  +  sum_neg sig(s_neg_k) * v_neg_k
            dL/dv_w     = (sig(s_pos) - 1) * v_c
            dL/dv_neg_k = sig(s_neg_k) * v_c

        Every gradient is negative-going along the useful direction, so SGD
        subtracts it and the pair scores rise. The centre index repeats across
        many pairs, so ``np.add.at`` accumulates updates rather than letting the
        last write win.
        """
        assert self.vectors_ is not None and self.context_vectors_ is not None

        centre_vectors = self.vectors_[centres]           # (n, d)
        context_vectors = self.context_vectors_[contexts]  # (n, d)
        noise_vectors = self.context_vectors_[noise]      # (n, neg, d)

        # Scalar scores.
        positive_scores = np.einsum("nd,nd->n", centre_vectors, context_vectors)
        negative_scores = np.einsum("nd,nkd->nk", centre_vectors, noise_vectors)

        # dL/ds for the positive and every negative pair.
        positive_gradient = (_sigmoid(positive_scores) - 1.0)          # (n,)
        negative_gradient = _sigmoid(negative_scores)                  # (n, neg)

        # Output-side gradients.
        context_gradient = positive_gradient[:, None] * centre_vectors
        noise_gradient = negative_gradient[:, :, None] * centre_vectors[:, None, :]

        # Input-side gradient for the centre word.
        centre_gradient = positive_gradient[:, None] * context_vectors + (
            negative_gradient[:, :, None] * noise_vectors
        ).sum(axis=1)

        # SGD step: subtract the gradient.
        np.add.at(self.vectors_, centres, -alpha * centre_gradient)
        np.add.at(self.context_vectors_, contexts, -alpha * context_gradient)
        np.add.at(
            self.context_vectors_,
            noise.reshape(-1),
            (-alpha * noise_gradient).reshape(-1, self.vector_size),
        )

        loss = float(
            -np.log(_sigmoid(positive_scores) + 1e-10).sum()
            - np.log(_sigmoid(-negative_scores) + 1e-10).sum()
        )
        return loss

    # -- inference ----------------------------------------------------------

    def document_vector(self, tokens: list[str]) -> np.ndarray:
        """Mean of the token vectors, ignoring out-of-vocabulary tokens.

        Averaging is the standard sentence-composition rule for word vectors and
        needs no extra training.
        """
        assert self.vectors_ is not None

        indices = [self.vocabulary_[token] for token in tokens if token in self.vocabulary_]
        if not indices:
            return np.zeros(self.vector_size, dtype=np.float64)
        return self.vectors_[indices].mean(axis=0)

    def transform(self, documents: list[list[str]]) -> np.ndarray:
        """Embed a batch of tokenised documents into an ``(n, d)`` matrix.

        Rows are L2-normalised so that downstream distance-based models such as
        KNN behave sensibly and no single long document dominates the norm.
        """
        if self.vectors_ is None:
            raise ValueError("Word2VecEmbedder must be fitted before transform.")

        matrix = np.vstack(
            [self.document_vector(tokens) for tokens in documents]
        )
        return normalize(matrix, norm="l2")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Scikit-learn transformer protocol, for use inside a Pipeline."""
        return np.asarray(
            [f"w2v_dim_{index}" for index in range(self.vector_size)],
            dtype=object,
        )

    def nearest_neighbours(self, word: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return the ``top_k`` most similar vocabulary words to ``word``.

        Exposed because inspecting neighbours is the standard sanity check for
        a trained word-embedding space, and the Model Lab page reports it.
        """
        if self.vocabulary_ is None or word not in self.vocabulary_:
            return []

        vector = self.vectors_[self.vocabulary_[word]]
        norms = np.linalg.norm(self.vectors_, axis=1)
        similarity = (self.vectors_ @ vector) / np.maximum(norms * np.linalg.norm(vector), 1e-9)
        order = np.argsort(-similarity)[: top_k + 1]

        return [
            (self.inverse_vocabulary_[index], float(similarity[index]))
            for index in order
            if index != self.vocabulary_[word]
        ][:top_k]


class LSAEmbedder:
    """Latent Semantic Analysis over a TF-IDF matrix.

    The corpus vocabulary matrix is factorised by a truncated SVD, so every
    document becomes a dense point in the top ``n_components`` latent semantic
    dimensions. Truncating the SVD discards the least significant components,
    which removes much of the sparsity and noise that raw counts carry.
    """

    name = "LSA (SVD)"

    def __init__(self, n_components: int = 200, max_features: int = 5000, min_df: int = 2) -> None:
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df

        self.vectorizer_: TfidfVectorizer | None = None
        self.svd_: TruncatedSVD | None = None
        self.feature_names_: list[str] = []

    def fit(self, documents: list[str]) -> "LSAEmbedder":
        if not documents:
            raise ValueError("LSAEmbedder.fit received no documents.")

        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=(1, 1),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        matrix = self.vectorizer_.fit_transform(documents)

        # n_components must stay below the matrix rank or SVD cannot converge.
        max_components = max(2, min(matrix.shape) - 1)
        self.svd_ = TruncatedSVD(
            n_components=min(self.n_components, max_components),
            random_state=42,
        )
        self.svd_.fit(matrix)

        self.feature_names_ = [
            f"latent_{index + 1}" for index in range(self.svd_.n_components)
        ]
        self.explained_variance_ = float(self.svd_.explained_variance_ratio_.sum())
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if self.vectorizer_ is None or self.svd_ is None:
            raise ValueError("LSAEmbedder must be fitted before transform.")
        matrix = self.vectorizer_.transform(documents)
        return normalize(self.svd_.transform(matrix), norm="l2")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Scikit-learn transformer protocol, for use inside a Pipeline."""
        return np.asarray(self.feature_names_, dtype=object)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def tokenize_documents(documents: list[str]) -> list[list[str]]:
    """Split already-cleaned text into token lists for the word embedder."""
    return [document.split() for document in documents]
