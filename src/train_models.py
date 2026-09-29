"""Stage 4 -- training and benchmarking over every representation x algorithm.

The matrix that reaches this stage is a list of *preprocessed documents* plus a
list of labels. For each of the three required representations this module

1. fits the representation on the **training documents only**,
2. transforms the held-out documents with the fitted object,
3. trains the four required classifiers (KNN, Logistic Regression, Random Forest,
   a small multi-layer perceptron),
4. scores accuracy, precision, recall, F1, ROC-AUC and a confusion matrix from
   the real hold-out predictions,

and finally persists the winning (representation, algorithm) pair so the app
serves exactly what was benchmarked.

Artifacts written:

* ``models/vectorizer.pkl``   -- the fitted representation of the winner
* ``models/best_model.pkl``   -- the winning classifier
* ``models/feature_names.pkl``-- column labels for the winner's matrix
* ``models/X_train.pkl`` / ``models/X_test.pkl`` -- the winner's matrices
* ``results/model_metrics.json`` -- every score in this run
* ``results/representation_benchmark.csv`` -- the full 3 x 4 grid

Run with::

    python -m src.train_models
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src.config import (
    BEST_MODEL,
    CLEAN_DATA,
    FEATURE_NAMES,
    LABEL_NAMES,
    MAX_FEATURES,
    METRICS_FILE,
    MIN_DF,
    MODEL_DIR,
    NGRAM_RANGE,
    RANDOM_STATE,
    REPRESENTATION_REPORT,
    RESULT_DIR,
    TEST_SIZE,
    VALIDATION_SIZE,
    XTRAIN,
    XTEST,
    ensure_dirs,
)
from src.feature_extraction import build_representations, as_dense

TOP_FEATURE_COUNT = 25

#: Label order used by every stored confusion matrix.
MATRIX_LABELS = [0, 1]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def build_models() -> dict:
    """The four required classifiers, all seeded for reproducibility."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=600, early_stopping=True, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }


#: Human-readable model-family tags used by the comparison discussion.
MODEL_FAMILY = {
    "Logistic Regression": "Parametric",
    "K-Nearest Neighbors": "Non-parametric",
    "Random Forest": "Non-parametric (ensemble)",
    "Neural Network": "Parametric (discriminative)",
}

#: One-line role statement per algorithm, surfaced in the report and the app.
MODEL_ROLE = {
    "Logistic Regression": (
        "Linear baseline. Learns one signed weight per term, so every decision is "
        "traceable to the vocabulary that produced it."
    ),
    "K-Nearest Neighbors": (
        "Instance-based. Stores the training set and classifies by majority vote of "
        "the closest documents -- no training, but prediction cost grows with it."
    ),
    "Random Forest": (
        "Bagged decision trees. Captures non-linear interactions between terms and is "
        "robust to the high dimensionality of a sparse text matrix."
    ),
    "Neural Network": (
        "Multi-layer perceptron. Learns non-linear feature interactions through "
        "back-propagation, at a much higher training cost."
    ),
}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _positive_scores(model, matrix) -> np.ndarray | None:
    """P(FAKE) when the estimator exposes it, otherwise the decision function."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(matrix), dtype=float)
    return None


def score_model(model, X_test, y_test) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix.

    Every value is derived from the predictions this model makes on the held-out
    split -- nothing is hard-coded or reused from an earlier run.
    """
    predictions = model.predict(X_test)

    scores = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    }

    positive = _positive_scores(model, X_test)
    if positive is None:
        scores["roc_auc"] = None
    else:
        try:
            scores["roc_auc"] = float(roc_auc_score(y_test, positive))
        except ValueError:  # single-class hold-out
            scores["roc_auc"] = None

    matrix = confusion_matrix(y_test, predictions, labels=MATRIX_LABELS).tolist()
    scores["confusion_matrix"] = {
        "labels": [LABEL_NAMES[index] for index in MATRIX_LABELS],
        "matrix": matrix,
        "accuracy": scores["accuracy"],
        "total": int(sum(sum(row) for row in matrix)),
        "errors": {
            f"true_{LABEL_NAMES[row]}_predicted_{LABEL_NAMES[column]}": matrix[row][column]
            for row in range(len(MATRIX_LABELS))
            for column in range(len(MATRIX_LABELS))
            if row != column
        },
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=MATRIX_LABELS, zero_division=0
    )
    scores["per_class"] = {
        LABEL_NAMES[label]: {
            "precision": round(float(precision[position]), 6),
            "recall": round(float(recall[position]), 6),
            "f1_score": round(float(f1[position]), 6),
            "support": int(support[position]),
        }
        for position, label in enumerate(MATRIX_LABELS)
    }

    return scores


# --------------------------------------------------------------------------
# Feature attribution
# --------------------------------------------------------------------------

def _class_separation(matrix, y_train, feature_names: list[str]) -> np.ndarray:
    """Signed per-column score: mean weight in FAKE documents minus REAL.

    Tree ensembles expose unsigned importances, so a raw importance cannot say
    which way a term pushes the decision. Projecting each term onto the
    difference between the two class means supplies that direction using the
    training documents only.
    """
    dense = as_dense(matrix)
    if dense.ndim != 2 or dense.shape[1] != len(feature_names):
        return np.zeros(len(feature_names))

    real = np.asarray(dense[y_train == 0].mean(axis=0)).ravel()
    fake = np.asarray(dense[y_train == 1].mean(axis=0)).ravel()
    return fake - real


def top_features(model, matrix, y_train, feature_names: list[str]) -> list[dict]:
    """The strongest columns of the winner, tagged with the class they favour."""
    if hasattr(model, "coef_"):
        weights = np.asarray(model.coef_).ravel()
        kind = "coefficient"
    elif hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_, dtype=float).ravel()
        kind = "importance"
    else:
        return []

    if len(weights) != len(feature_names):
        return []

    separation = _class_separation(matrix, y_train, feature_names)
    order = np.argsort(np.abs(weights))[::-1][:TOP_FEATURE_COUNT]

    return [
        {
            "feature": feature_names[index],
            "weight": round(float(weights[index]), 6),
            "direction": LABEL_NAMES[1 if separation[index] > 0 else 0],
            "weight_kind": kind,
        }
        for index in order
    ]


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------

def select_candidate(
    x_train: list[str], y_train: np.ndarray
) -> tuple[str, str, pd.DataFrame]:
    """Choose the (representation, algorithm) pair on a validation split.

    The evaluation split is never consulted here. Carving a validation set out of
    the *training* documents means the reported hold-out scores are an honest
    estimate rather than the maximum of twelve attempts.
    """
    fit_documents, validation_documents, y_fit, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print(f"  selection split: {len(fit_documents)} fit / {len(validation_documents)} validation")

    rows = []
    for representation_name, representation in build_representations().items():
        representation.fit(fit_documents)
        fit_matrix = representation.transform(fit_documents)
        validation_matrix = representation.transform(validation_documents)

        for model_name, model in build_models().items():
            model.fit(fit_matrix, y_fit)
            predictions = model.predict(validation_matrix)
            rows.append(
                {
                    "Representation": representation_name,
                    "Model": model_name,
                    "validation_f1": float(f1_score(y_validation, predictions, zero_division=0)),
                    "validation_accuracy": float(accuracy_score(y_validation, predictions)),
                    "validation_rows": int(len(validation_documents)),
                    "fit_rows": int(len(fit_documents)),
                }
            )

    table = pd.DataFrame(rows).sort_values("validation_f1", ascending=False)
    chosen = table.iloc[0]

    print(f"  validation winner: {chosen['Representation']} + {chosen['Model']} "
          f"(F1 {chosen['validation_f1']:.4f})")
    return str(chosen["Representation"]), str(chosen["Model"]), table


def run_benchmark(
    x_train, x_test, y_train, y_test, chosen_representation: str, chosen_model: str
) -> tuple[dict, dict, dict, pd.DataFrame]:
    """Fit every representation, train every classifier, score on the hold-out.

    Returns ``(representations, headline, best, selection)`` where
    ``representations`` is the full 3 x 4 grid, ``headline`` maps each algorithm
    to its strongest representation, ``best`` is the deployed pair chosen on the
    validation split, and ``selection`` is the selection table.
    """
    representations: dict[str, dict] = {}
    fitted_representations: dict[str, object] = {}
    matrices: dict[str, tuple] = {}

    for name, representation in build_representations().items():
        print(f"\n  representation: {name}")
        representation.fit(x_train)
        train_matrix = representation.transform(x_train)
        test_matrix = representation.transform(x_test)
        matrices[name] = (train_matrix, test_matrix)
        fitted_representations[name] = representation

        description = representation.describe()
        print(
            f"    matrix {train_matrix.shape} -> {test_matrix.shape} "
            f"({description.get('dimensions')} dims, dense={representation.is_dense})"
        )

        scores: dict[str, dict] = {}
        for model_name, model in build_models().items():
            started = time.perf_counter()
            model.fit(train_matrix, y_train)
            elapsed = time.perf_counter() - started

            result = score_model(model, test_matrix, y_test)
            result["train_seconds"] = round(elapsed, 3)
            result["params"] = {
                key: str(value)
                for key, value in model.get_params().items()
                if key
                in {
                    "C",
                    "penalty",
                    "n_estimators",
                    "n_neighbors",
                    "min_samples_leaf",
                    "hidden_layer_sizes",
                    "max_iter",
                }
            }
            result["selected"] = (
                name == chosen_representation and model_name == chosen_model
            )
            scores[model_name] = result

            marker = "  <- selected" if result["selected"] else ""
            print(
                f"      {model_name:<20} acc={result['accuracy']:.4f} "
                f"prec={result['precision']:.4f} rec={result['recall']:.4f} "
                f"f1={result['f1_score']:.4f} "
                f"auc={result['roc_auc'] if result['roc_auc'] is not None else float('nan'):.4f} "
                f"({elapsed:.2f}s){marker}"
            )

        representations[name] = {**description, "models": scores}

    # Headline table: each algorithm quoted at its strongest representation.
    headline: dict[str, dict] = {}
    for model_name in build_models():
        best_representation = max(
            representations,
            key=lambda rep: representations[rep]["models"][model_name]["f1_score"],
        )
        entry = dict(representations[best_representation]["models"][model_name])
        entry["representation"] = best_representation
        entry["family"] = MODEL_FAMILY[model_name]
        entry["role"] = MODEL_ROLE[model_name]
        headline[model_name] = entry

    best = {
        "model_name": chosen_model,
        "representation": chosen_representation,
        "model": build_models()[chosen_model],
        "representation_object": fitted_representations[chosen_representation],
        "matrices": matrices[chosen_representation],
    }

    return representations, headline, best


def _roc_payload(model, matrix, y_test) -> dict:
    positive = _positive_scores(model, matrix)
    if positive is None:
        return {"fpr": [], "tpr": [], "auc": None}

    fpr, tpr, _ = roc_curve(y_test, positive)
    return {
        "fpr": [round(float(value), 5) for value in fpr],
        "tpr": [round(float(value), 5) for value in tpr],
        "auc": float(roc_auc_score(y_test, positive)),
    }


def _write_grid(representations: dict, headline: dict, selection: "pd.DataFrame") -> None:
    rows = []
    for representation_name, block in representations.items():
        for model_name, scores in block["models"].items():
            rows.append(
                {
                    "Representation": representation_name,
                    "Model": model_name,
                    "Family": MODEL_FAMILY[model_name],
                    "Accuracy": round(scores["accuracy"], 6),
                    "Precision": round(scores["precision"], 6),
                    "Recall": round(scores["recall"], 6),
                    "F1": round(scores["f1_score"], 6),
                    "ROC-AUC": scores["roc_auc"],
                    "Train seconds": scores["train_seconds"],
                    "Dimensions": block.get("dimensions"),
                    "Validation F1": round(
                        float(
                            selection.loc[
                                (selection["Representation"] == representation_name)
                                & (selection["Model"] == model_name),
                                "validation_f1",
                            ].iloc[0]
                        ),
                        6,
                    )
                    if not selection.empty
                    else None,
                    "Selected": bool(scores.get("selected")),
                    "Headline": model_name in headline
                    and headline[model_name]["representation"] == representation_name,
                }
            )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPRESENTATION_REPORT, index=False)


def main() -> None:
    ensure_dirs()

    print("=" * 66)
    print("STAGE 4  |  MODEL TRAINING")
    print("=" * 66)

    x_train = joblib.load(MODEL_DIR / "train_documents.pkl")
    x_test = joblib.load(MODEL_DIR / "test_documents.pkl")
    y_train = joblib.load(MODEL_DIR / "y_train.pkl")
    y_test = joblib.load(MODEL_DIR / "y_test.pkl")

    # Rows dropped because cleaning left no tokens at all: the count is recorded
    # so the report can state the corpus-to-split reconciliation honestly.
    try:
        prepared = pd.read_csv(CLEAN_DATA, usecols=["clean_text"])
        source_rows = int(len(prepared))
        dropped_empty = int((prepared["clean_text"].fillna("").str.strip() == "").sum())
    except (FileNotFoundError, OSError, ValueError):
        source_rows = len(y_train) + len(y_test)
        dropped_empty = 0

    print("\nModel selection (validation split, hold-out untouched)")
    chosen_representation, chosen_model, selection = select_candidate(x_train, y_train)

    print("\nBenchmark on the held-out split")
    representations, headline, best = run_benchmark(
        x_train, x_test, y_train, y_test, chosen_representation, chosen_model
    )

    model = best["model"]
    representation_object = best["representation_object"]
    train_matrix, test_matrix = best["matrices"]
    feature_names = representation_object.feature_names()

    model.fit(train_matrix, y_train)

    joblib.dump(representation_object, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(model, BEST_MODEL)
    joblib.dump(feature_names, FEATURE_NAMES)
    joblib.dump(train_matrix, XTRAIN)
    joblib.dump(test_matrix, XTEST)

    winner_scores = representations[chosen_representation]["models"][chosen_model]
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model_name": best["model_name"],
        "best_representation": best["representation"],
        "best_f1_score": winner_scores["f1_score"],
        "selection": {
            "protocol": "highest F1 on a stratified validation split carved from the training data",
            "validation_size": VALIDATION_SIZE,
            "validation_rows": int(selection.iloc[0].get("validation_rows", 0)),
            "validation_f1": float(selection.iloc[0]["validation_f1"]),
            "validation_accuracy": float(selection.iloc[0]["validation_accuracy"]),
            "winner": {
                "representation": chosen_representation,
                "model": chosen_model,
            },
            "table": selection.to_dict(orient="records"),
        },
        "models": headline,
        "representations": representations,
        "confusion_matrices": {
            name: {
                "representation": headline[name]["representation"],
                "f1_score": headline[name]["f1_score"],
                **representations[headline[name]["representation"]]["models"][name]["confusion_matrix"],
            }
            for name in headline
        },
        "per_class_metrics": {
            name: representations[headline[name]["representation"]]["models"][name]["per_class"]
            for name in headline
        },
        "headline_representation": {
            name: headline[name]["representation"] for name in headline
        },
        "dataset": {
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_real": int((y_train == 0).sum()),
            "train_fake": int((y_train == 1).sum()),
            "test_real": int((y_test == 0).sum()),
            "test_fake": int((y_test == 1).sum()),
            "source_rows": int(source_rows),
            "dropped_empty": int(dropped_empty),
            "usable_rows": int(len(y_train) + len(y_test)),
        },
        "features": {
            "representation": best["representation"],
            **{
                key: value
                for key, value in representation_object.describe().items()
                if key != "name"
            },
            "max_features": MAX_FEATURES,
            "ngram_range": list(NGRAM_RANGE),
            "min_df": MIN_DF,
            "vocabulary_size": len(feature_names),
        },
        "split": {
            "test_size": TEST_SIZE,
            "validation_size": VALIDATION_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "label_names": {str(key): value for key, value in LABEL_NAMES.items()},
        "best_model": {
            "name": best["model_name"],
            "representation": best["representation"],
            "family": MODEL_FAMILY[best["model_name"]],
            "confusion_matrix": winner_scores["confusion_matrix"],
            "roc_curve": _roc_payload(model, test_matrix, y_test),
            "top_features": top_features(model, train_matrix, y_train, feature_names),
        },
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    _write_grid(representations, headline, selection)

    print("-" * 66)
    print(f"Best combination : {best['representation']} + {best['model_name']}")
    print(f"F1 score         : {metrics['best_f1_score']:.4f}")
    print(f"Model            -> {BEST_MODEL}")
    print(f"Representation    -> {MODEL_DIR / 'vectorizer.pkl'}")
    print(f"Metrics          -> {METRICS_FILE}")
    print(f"Grid             -> {REPRESENTATION_REPORT}")
    print("=" * 66)


if __name__ == "__main__":
    main()
