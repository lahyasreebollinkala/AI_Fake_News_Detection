"""Tests for the from-scratch NumPy embedding implementations.

Run with:  python -m pytest tests/test_embeddings.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings import (  # noqa: E402
    LSAEmbedder,
    Word2VecEmbedder,
    _sigmoid,
    tokenize_documents,
)

SAMPLE = [
    "the president spoke about the economy and the market",
    "the president met with congress about the economy",
    "trump said the economy was strong and the market rose",
    "police arrested a man after the court hearing downtown",
    "the court heard the case and police arrested a suspect",
    "a man was arrested downtown after a court hearing",
    "the team won the game after a strong second half",
    "the team scored after a strong attack in the game",
    "the team won the match with a strong performance",
] * 6


class TestSigmoid:
    def test_matches_scipy_style_reference(self):
        values = np.array([-50.0, -2.0, -0.5, 0.0, 0.5, 2.0, 50.0])
        expected = 1.0 / (1.0 + np.exp(-values))
        assert np.allclose(_sigmoid(values), expected, atol=1e-9)

    def test_is_numerically_stable_at_extremes(self):
        extreme = np.array([-800.0, 800.0])
        result = _sigmoid(extreme)
        assert np.all(np.isfinite(result))
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)

    def test_is_monotonic(self):
        values = np.linspace(-20, 20, 400)
        assert np.all(np.diff(_sigmoid(values)) > 0)


class TestWord2VecEmbedder:
    def test_fit_learns_and_shapes_are_correct(self):
        model = Word2VecEmbedder(vector_size=25, epochs=3, min_count=2).fit(
            tokenize_documents(SAMPLE)
        )
        matrix = model.transform(tokenize_documents(SAMPLE))
        assert matrix.shape == (len(SAMPLE), 25)
        assert model.vectors_.shape == (len(model.vocabulary_), 25)

    def test_rows_are_l2_normalised(self):
        model = Word2VecEmbedder(vector_size=16, epochs=2).fit(
            tokenize_documents(SAMPLE)
        )
        matrix = model.transform(tokenize_documents(SAMPLE))
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-9)

    def test_is_deterministic_for_a_fixed_seed(self):
        tokens = tokenize_documents(SAMPLE)
        first = Word2VecEmbedder(vector_size=20, epochs=3, seed=7).fit(tokens)
        second = Word2VecEmbedder(vector_size=20, epochs=3, seed=7).fit(tokens)
        assert np.array_equal(first.vectors_, second.vectors_)

    def test_min_count_drops_rare_words(self):
        tokens = tokenize_documents(SAMPLE + ["unicorn quixotic"])
        model = Word2VecEmbedder(min_count=3, epochs=2).fit(tokens)
        assert "unicorn" not in model.vocabulary_
        assert "president" in model.vocabulary_

    def test_out_of_vocabulary_document_becomes_zero_vector(self):
        model = Word2VecEmbedder(vector_size=12, epochs=2, min_count=2).fit(
            tokenize_documents(SAMPLE)
        )
        vector = model.document_vector(["zzzznotaword", "qqqqalsomissing"])
        assert vector.shape == (12,)
        assert np.allclose(vector, 0.0)

    def test_transform_relevant_documents(self):
        """Related documents must land closer than unrelated ones.

        This is the real test of whether the skip-gram objective is learning
        context similarity rather than just churning weights.
        """
        model = Word2VecEmbedder(vector_size=40, epochs=25, seed=1).fit(
            tokenize_documents(SAMPLE)
        )
        matrix = model.transform(tokenize_documents(SAMPLE))

        def cosine(a, b):
            return float(matrix[a] @ matrix[b])

        # "economy" cluster
        assert cosine(0, 2) > cosine(0, 6)
        # "court / police" cluster
        assert cosine(3, 4) > cosine(3, 0)
        # "team" cluster
        assert cosine(6, 7) > cosine(6, 3)

    def test_training_loss_decreases(self):
        """The SGNS objective must be minimised, not maximised."""
        model = Word2VecEmbedder(vector_size=30, epochs=5, seed=3).fit(
            tokenize_documents(SAMPLE)
        )
        assert model.training_loss_[-1] < model.training_loss_[0]

    def test_related_words_score_above_random_pairs(self):
        model = Word2VecEmbedder(vector_size=50, epochs=30, seed=5, learning_rate=0.05).fit(
            tokenize_documents(SAMPLE)
        )
        related = []
        for a, b in (("president", "economy"), ("court", "police"), ("team", "game")):
            vectors = [model.vectors_[model.vocabulary_[w]] for w in (a, b)]
            norm = np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
            related.append(float(vectors[0] @ vectors[1] / norm))

        rng = np.random.default_rng(0)
        index = rng.integers(0, len(model.vocabulary_), 200)
        block = model.vectors_[index]
        block = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-9)
        random_pairs = float(np.mean(block @ block.T))

        assert np.mean(related) > random_pairs

    def test_nearest_neighbours_are_topically_related(self):
        model = Word2VecEmbedder(vector_size=50, epochs=30, seed=11).fit(
            tokenize_documents(SAMPLE)
        )
        neighbours = dict(model.nearest_neighbours("court", top_k=5))
        # "court", "police", "arrested" and "suspect" all describe the same
        # crime-scene cluster in the fixture.
        crime_cluster = {"police", "arrested", "suspect", "hearing", "case"}
        assert crime_cluster & set(neighbours)

    def test_nearest_neighbours_of_unknown_word_is_empty(self):
        model = Word2VecEmbedder(vector_size=16, epochs=2).fit(
            tokenize_documents(SAMPLE)
        )
        assert model.nearest_neighbours("zzzznotaword") == []

    def test_get_feature_names_out_matches_dimension(self):
        model = Word2VecEmbedder(vector_size=33, epochs=1).fit(
            tokenize_documents(SAMPLE)
        )
        assert len(model.get_feature_names_out()) == 33

    def test_rejects_empty_corpus(self):
        with pytest.raises(ValueError):
            Word2VecEmbedder().fit([])

    def test_rejects_corpus_with_no_frequent_words(self):
        with pytest.raises(ValueError):
            Word2VecEmbedder(min_count=999).fit(tokenize_documents(SAMPLE))

    def test_transform_before_fit_raises(self):
        with pytest.raises(ValueError):
            Word2VecEmbedder().transform(tokenize_documents(SAMPLE))


class TestLSAEmbedder:
    def test_output_shape_and_normalisation(self):
        model = LSAEmbedder(n_components=20).fit(SAMPLE)
        matrix = model.transform(SAMPLE)
        assert matrix.shape == (len(SAMPLE), 20)
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-9)

    def test_is_deterministic(self):
        first = LSAEmbedder(n_components=15).fit(SAMPLE).transform(SAMPLE)
        second = LSAEmbedder(n_components=15).fit(SAMPLE).transform(SAMPLE)
        assert np.allclose(first, second)

    def test_components_do_not_exceed_matrix_rank(self):
        """A tiny corpus has low rank, so the SVD must adapt its dimension."""
        model = LSAEmbedder(n_components=500).fit(SAMPLE)
        assert model.svd_.n_components < 500
        assert model.transform(SAMPLE).shape[1] == model.svd_.n_components

    def test_explained_variance_is_a_fraction(self):
        model = LSAEmbedder(n_components=20).fit(SAMPLE)
        # Tolerance is needed because the ratio is a sum of many terms and can
        # land a few ULPs above 1.0.
        assert 0.0 < model.explained_variance_ <= 1.0 + 1e-9

    def test_similar_documents_score_higher_than_unrelated(self):
        matrix = LSAEmbedder(n_components=30).fit(SAMPLE).transform(SAMPLE)
        assert float(matrix[0] @ matrix[2]) > float(matrix[0] @ matrix[6])
        assert float(matrix[3] @ matrix[4]) > float(matrix[3] @ matrix[0])

    def test_get_feature_names_out(self):
        model = LSAEmbedder(n_components=20).fit(SAMPLE)
        assert list(model.get_feature_names_out()) == model.feature_names_

    def test_rejects_empty_corpus(self):
        with pytest.raises(ValueError):
            LSAEmbedder().fit([])

    def test_transform_before_fit_raises(self):
        with pytest.raises(ValueError):
            LSAEmbedder().transform(SAMPLE)


class TestTokenize:
    def test_splits_on_whitespace(self):
        assert tokenize_documents(["a b c", "d"]) == [["a", "b", "c"], ["d"]]

    def test_empty_string_yields_no_tokens(self):
        assert tokenize_documents(["", "a"]) == [[], ["a"]]
