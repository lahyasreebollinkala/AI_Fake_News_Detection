"""Batch-input validation: short and empty rows must never be scored."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.inference import MIN_WORDS_FOR_PREDICTION, prepare_batch_frame


def test_rows_below_word_floor_are_removed_and_counted():
    frame = pd.DataFrame(
        {
            "text": [
                "This article contains more than five useful words for the model.",
                "Too short.",
                "",
                "Another complete article with enough words to be classified.",
            ]
        }
    )

    usable, skipped = prepare_batch_frame(frame, "text")

    assert skipped == 2
    assert len(usable) == 2
    assert usable.index.tolist() == [0, 1]


def test_title_words_are_included_in_the_word_count():
    frame = pd.DataFrame({"title": ["Breaking news headline"], "body": ["short note"]})

    usable, skipped = prepare_batch_frame(frame, "body", "title")

    assert skipped == 0
    assert len(usable) == 1


def test_missing_text_column_raises_a_clear_error():
    frame = pd.DataFrame({"body": ["Some text that is long enough to score."]})

    with pytest.raises(KeyError, match="Text column 'text'"):
        prepare_batch_frame(frame, "text")


def test_threshold_is_configurable():
    frame = pd.DataFrame({"text": ["one two three four"]})

    usable, skipped = prepare_batch_frame(
        frame, "text", min_words=MIN_WORDS_FOR_PREDICTION
    )

    assert usable.empty
    assert skipped == 1
