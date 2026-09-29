"""Stage 3 -- corpus split.

Produces the **only** place the train/test boundary is drawn, before any
vectoriser, IDF table or embedding basis exists. Persists:

* ``models/train_documents.pkl`` -- the preprocessed training documents
* ``models/test_documents.pkl``  -- the held-out documents
* ``models/y_train.pkl`` / ``models/y_test.pkl`` -- the matching labels

Stage 4 (:mod:`src.train_models`) fits every representation on the training
documents *only*, then transforms the held-out documents with the fitted object.
Because the split is materialised here and no representation is fitted until
afterwards, vocabulary, IDF weights and the latent basis are structurally
incapable of seeing the evaluation data.

Run with::

    python -m src.feature_engineering
"""

from __future__ import annotations

import joblib
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_NAMES,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    ensure_dirs,
)
from src.data_loader import load_training_frame

TRAIN_DOCUMENTS = MODEL_DIR / "train_documents.pkl"
TEST_DOCUMENTS = MODEL_DIR / "test_documents.pkl"


def split_corpus(documents: list[str], labels) -> tuple[list[str], list[str], object, object]:
    """Stratified, seeded split of the corpus into train and hold-out halves."""
    return train_test_split(
        documents,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def main() -> None:
    ensure_dirs()

    print("=" * 66)
    print("STAGE 3  |  CORPUS SPLIT")
    print("=" * 66)

    frame = load_training_frame()

    documents = frame["clean_text"].fillna("").astype(str).tolist()
    labels = frame["label"].astype(int).to_numpy()

    # Rows whose cleaning produced nothing carry no signal; dropping them keeps
    # the split balanced without an all-zero row in either matrix.
    keep = [index for index, text in enumerate(documents) if text.strip()]
    dropped = len(documents) - len(keep)
    documents = [documents[index] for index in keep]
    labels = labels[keep]

    x_train, x_test, y_train, y_test = split_corpus(documents, labels)

    joblib.dump(x_train, TRAIN_DOCUMENTS)
    joblib.dump(x_test, TEST_DOCUMENTS)
    joblib.dump(y_train, MODEL_DIR / "y_train.pkl")
    joblib.dump(y_test, MODEL_DIR / "y_test.pkl")

    print(f"Documents            : {len(documents)}")
    print(f"Dropped (no content) : {dropped}")
    print(f"Train / test rows    : {len(x_train)} / {len(x_test)}")
    print(f"Test  {LABEL_NAMES[0]:<4}         : {int((y_test == 0).sum())}")
    print(f"Test  {LABEL_NAMES[1]:<4}         : {int((y_test == 1).sum())}")
    print(f"random_state         : {RANDOM_STATE}   test_size: {TEST_SIZE}")
    print("-" * 66)
    print("No vectoriser has been fitted yet -- the split is drawn first, so the")
    print("vocabulary, the IDF table and the embedding basis are learned from the")
    print("training documents only.")
    for path in (TRAIN_DOCUMENTS, TEST_DOCUMENTS):
        print(f"saved -> {path}")
    print("=" * 66)


if __name__ == "__main__":
    main()
