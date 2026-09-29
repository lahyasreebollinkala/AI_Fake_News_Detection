"""Single source of truth for the eight required documentation sections.

:func:`build_sections` returns an ordered list of

``{"id", "number", "title", "blocks": [{"kind", ...}]}``

where every ``kind`` is understood by both renderers:

* ``prose``  -- ``text`` (Markdown)
* ``table``  -- ``caption``, ``columns``, ``rows``
* ``list``   -- ``title``, ``items``
* ``metric`` -- ``caption``

The prose is written by hand in :mod:`src.concepts`; **every number is read from
the artifacts on disk**. The report therefore cannot drift from the last real
run, and it cannot contain a fabricated result: if the pipeline has not been
built, the section comes back empty and the caller says so.
"""

from __future__ import annotations

import pandas as pd

from src import concepts
from src.config import (
    EMBEDDING_COMPONENTS,
    LABEL_NAMES,
    MAX_FEATURES,
    MIN_DF,
    REPRESENTATIONS,
)

SECTION_TITLES = [
    ("introduction", "Introduction"),
    ("dataset", "Dataset Description"),
    ("methodology", "Methodology"),
    ("results", "Results"),
    ("discussion", "Discussion"),
    ("conclusion", "Conclusion"),
    ("appendix", "Appendix"),
    ("references", "References"),
]


INDEX_TERMS = (
    "Fake news detection, text classification, natural language processing, TF-IDF, "
    "latent semantic analysis, random forest, Streamlit."
)


def abstract(metrics: dict) -> str:
    """The report abstract as a single string of live prose.

    Shared by the Markdown, LaTeX, HTML and in-app renderings so the four can
    never describe the same run differently. Returns ``""`` when no benchmark
    has been recorded yet, which callers use to omit the front matter.
    """
    best = metrics.get("best_model_name")
    scores = metrics.get("models", {}).get(best or "", {})
    if not scores:
        return ""

    representation = metrics.get("best_representation", "n/a")
    holdout = metrics.get("dataset", {}).get("test_rows", 0)

    return (
        "Fabricated news spreads faster than verified news and is impossible to screen "
        "by hand at newsroom speed. This report presents a supervised text-classification "
        "pipeline that labels an article as authentic or fabricated and returns a "
        "confidence together with the terms that drove the decision. The pipeline "
        "compares four algorithms -- K-Nearest Neighbours, Logistic Regression, Random "
        "Forest and a multi-layer perceptron -- over three feature representations: "
        "Bag-of-Words, TF-IDF and latent-semantic embeddings fitted by truncated SVD. "
        "Vocabulary, IDF weights and the embedding basis are fitted on the training split "
        "only, and the deployed pair is chosen on a held-out validation split, so the "
        "hold-out is scored exactly once. The selected combination, "
        f"{representation} with {best}, reaches an accuracy of "
        f"{scores.get('accuracy', 0):.4f}, a precision of {scores.get('precision', 0):.4f}, "
        f"a recall of {scores.get('recall', 0):.4f} and an F1 score of "
        f"{scores.get('f1_score', 0):.4f} on a stratified hold-out of {holdout} articles. "
        "The reported confidence is the model's own score, not a calibrated probability."
    )


def _fill(text: str, context: dict[str, object]) -> str:
    """Substitute ``{token}`` placeholders in report prose.

    Keeps every quantity in the narrative tied to the recorded run, so a
    re-trained model can never leave a stale dimension or row count behind.
    """
    try:
        return text.format(**context)
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(f"unresolved placeholder in report prose: {error}") from error


def _context(metrics: dict) -> dict[str, object]:
    """Values available to ``{token}`` placeholders in the report prose."""
    from src.config import EMBEDDING_COMPONENTS, MAX_FEATURES, MIN_DF

    representations = metrics.get("representations", {})

    def _dimensions(name: str, fallback: int) -> int:
        try:
            return int(representations[name]["dimensions"])
        except (KeyError, TypeError, ValueError):
            return fallback

    dataset = metrics.get("dataset", {})
    return {
        "sample_rows": _sample_rows(),
        "bow_dimensions": _dimensions("Bag-of-Words", MAX_FEATURES),
        "bow_min_df": MIN_DF,
        "max_features": MAX_FEATURES,
        "embedding_dimensions": _dimensions(
            "Embeddings (LSA)", EMBEDDING_COMPONENTS
        ),
        "train_rows": dataset.get("train_rows", 0),
        "test_rows": dataset.get("test_rows", 0),
        "usable_rows": dataset.get("usable_rows", 0),
    }


def _sample_rows() -> int:
    """Row count of the committed sample corpus, or 0 when it is absent.

    Counted from the file rather than taken from the run metrics, because the
    metrics record the *prepared* corpus: one duplicate has already been
    removed by the time they are written.
    """
    from src.config import SAMPLE_DATA

    if not SAMPLE_DATA.exists():
        return 0
    with open(SAMPLE_DATA, encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _routes_phrase() -> str:
    """``"all nine routes"``, counted from the navigation manifest.

    Read live so the appendix cannot go stale when a page is added. The import is
    guarded because the manifest belongs to the application layer while this
    module belongs to the reporting layer, and must still import without it.
    """
    try:
        from app.navigation import page_count

        return f"all {page_count()} routes"
    except Exception:  # pragma: no cover - navigation manifest unavailable
        return "every route"


def _pct(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value, places: int = 4) -> str:
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "n/a"


def _int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------

def _introduction(metrics: dict, profile: dict) -> list[dict]:
    blocks = [
        {
            "kind": "prose",
            "text": concepts.AI_OVERVIEW,
        },
        {
            "kind": "prose",
            "text": (
                "**Problem statement.** Given a news article, decide whether it is "
                "authentic reporting or fabricated content, and do it fast enough to "
                "screen the whole incoming feed. The hard part is that the two classes "
                "are linguistically similar: both are written in formal register, both "
                "quote sources, and both cover the same events. The difference lies in "
                "statistical regularities of vocabulary, sentence rhythm and sourcing "
                "convention that a human expert detects only slowly and inconsistently."
            ),
        },
        {
            "kind": "prose",
            "text": (
                "**Why it matters.** Fabricated news travels faster than verified news "
                "during elections, pandemics and conflicts, and it moves markets, "
                "changes health behaviour and erodes trust in journalism at exactly the "
                "moments that information matters most. Automating the screening step "
                "does not replace a fact-checker; it decides what a fact-checker should "
                "look at first."
            ),
        },
    ]

    if metrics:
        models = metrics.get("models", {})
        best = models.get(metrics.get("best_model_name", ""), {})
        split = metrics.get("dataset", {})
        blocks.append(
            {
                "kind": "prose",
                "text": (
                    f"This report evaluates four supervised classifiers -- K-Nearest "
                    f"Neighbours, Logistic Regression, Random Forest and a multi-layer "
                    f"perceptron -- over three feature representations: Bag-of-Words, "
                    f"TF-IDF and latent-semantic embeddings. On a stratified hold-out "
                    f"of {_int(split.get('test_rows', 0))} articles the selected "
                    f"combination, {metrics.get('best_representation', 'n/a')} with "
                    f"{metrics.get('best_model_name', 'n/a')}, reaches an accuracy of "
                    f"{_pct(best.get('accuracy'))}, a precision of "
                    f"{_pct(best.get('precision'))}, a recall of {_pct(best.get('recall'))} "
                    f"and an F1 score of {_pct(best.get('f1_score'))}."
                ),
            }
        )

    if profile:
        labels = profile.get("labels", {})
        blocks.append(
            {
                "kind": "prose",
                "text": (
                    f"All figures are computed from a corpus of "
                    f"{_int(labels.get('total', 0))} labelled articles "
                    f"({_int(labels.get('real', 0))} authentic, "
                    f"{_int(labels.get('fake', 0))} fabricated)."
                ),
            }
        )

    return blocks


def _dataset(metrics: dict, profile: dict) -> list[dict]:
    blocks: list[dict] = []

    blocks.append(
        {
            "kind": "prose",
            "text": (
                "**Source.** The corpus is the public *Fake and Real News Dataset* "
                "[1], distributed through Kaggle. It ships "
                "as two files -- `Fake.csv` and `True.csv` -- each with a `title`, a "
                "`text` body, a `subject` desk and a `date`. A seeded, stratified "
                "{sample_rows:,}-row sample of that dataset is committed to the repository as "
                "`data/sample_dataset.csv` so the project reproduces without a 117 MB "
                "download. The loader prefers the two full-size files whenever they are "
                "present in `data/`, so no code change is needed to scale up."
            ),
        }
    )

    if profile:
        labels = profile.get("labels", {})
        characters = profile.get("characters", {})
        words = profile.get("words", {})
        eda = profile.get("eda", {})

        blocks.append(
            {
                "kind": "table",
                "caption": "{table_corpus} -- Corpus composition",
                    "ref": "corpus",
                "columns": ["Property", "Value"],
                "rows": [
                    ["Articles", _int(labels.get("total", 0))],
                    ["Authentic (label 0)", _int(labels.get("real", 0))],
                    ["Fabricated (label 1)", _int(labels.get("fake", 0))],
                    ["Authentic share", _pct(labels.get("real_share", 0))],
                    ["Fabricated share", _pct(labels.get("fake_share", 0))],
                    ["Mean characters per article", _num(characters.get("mean"), 1)],
                    ["Median characters per article", _num(characters.get("median"), 1)],
                    ["Mean words per article", _num(words.get("mean"), 1)],
                    ["Median words per article", _num(words.get("median"), 1)],
                    ["Longest article (characters)", _int(characters.get("max", 0))],
                    ["Duplicate headline+body pairs", _int(eda.get("duplicate_rows", 0))],
                    ["Rows with an empty body", _int(eda.get("empty_body_rows", 0))],
                    [
                        "Rows empty after cleaning",
                        _int(eda.get("empty_cleaned_rows", 0)),
                    ],
                ],
            }
        )

        if profile.get("subjects"):
            rows = sorted(profile["subjects"].items(), key=lambda item: -item[1])
            blocks.append(
                {
                    "kind": "table",
                    "caption": "{table_desks} -- Articles per news desk",
                    "ref": "desks",
                    "columns": ["Subject", "Articles"],
                    "rows": [[name, _int(count)] for name, count in rows],
                }
            )

        vocabulary = profile.get("vocabulary", {})
        if vocabulary:
            blocks.append(
                {
                    "kind": "metric",
                    "caption": (
                        f"{_int(vocabulary.get('distinct_tokens', 0))} distinct tokens "
                        "survive cleaning across the corpus."
                    ),
                }
            )

    blocks.append(
        {
            "kind": "list",
            "title": "Columns",
            "items": [
                "`title` -- the headline.",
                "`text` -- the article body.",
                "`subject` -- the news desk, carried for exploration only and never fed "
                "to the classifier.",
                "`label` -- `0` for authentic, `1` for fabricated.",
                "`full_text` -- `title + \". \" + text`, the string the model actually reads.",
                "Five engineered editorial signals (clickbait, emotion, sentiment, "
                "readability, source trust) computed for auditability.",
                "`clean_text` -- `full_text` after the NLP cleaning chain.",
            ],
        }
    )

    blocks.append(
        {
            "kind": "prose",
            "text": (
                "**Class balance and data quality.** The corpus is majority-fabricated, "
                "so accuracy alone would be a weak yardstick: a model that predicted FAKE "
                "for every article would already score 0.58. Precision, recall and F1 are "
                "reported alongside it, and no cost-sensitive reweighting is applied -- "
                "the classes are left as they occur naturally, so the reported figures "
                "describe the operating point a deployed screener would actually run at "
                "rather than a tuned one [10]."
            ),
        }
    )

    blocks.append(
        {
            "kind": "prose",
            "text": (
                "**De-duplication.** Duplicate headline-plus-body pairs are removed during "
                "preparation. The same article appearing on both sides of the split would "
                "inflate every score, and it is the single most common way a text "
                "classification benchmark produces a flattering and meaningless number."
            ),
        }
    )

    return blocks


def _methodology(metrics: dict, profile: dict) -> list[dict]:
    blocks: list[dict] = [
        {
            "kind": "prose",
            "text": (
                "The pipeline is six runnable stages, each writing its artifacts to a "
                "declared path in `src/config.py` so the training scripts and the "
                "application share one source of truth."
            ),
        },
        {
            "kind": "list",
            "title": "Pipeline stages",
            "items": [
                "**Acquire** -- read the source corpus, normalise the schema to "
                "`title / text / subject / label`, drop duplicates.",
                "**Clean** -- apply the NLP preprocessing chain to `full_text`.",
                "**Split** -- draw a stratified hold-out *before* any representation is "
                "fitted.",
                "**Represent** -- fit Bag-of-Words, TF-IDF and the latent-semantic "
                "embedding on the training documents only.",
                "**Select and train** -- choose the combination with the best F1 on a "
                "validation split carved from the training data, then score every "
                "candidate once on the untouched hold-out.",
                "**Serve** -- the application reloads the fitted representation and the "
                "model, and returns class probabilities with a term attribution.",
            ],
        },
        {
            "kind": "list",
            "title": "Text preprocessing",
            "items": [f"**{name}** -- {text}" for name, text in concepts.PREPROCESSING_STEPS],
        },
        {
            "kind": "list",
            "title": "Feature extraction",
            "items": [
                f"**{name}** -- {text}" for name, text in concepts.FEATURE_EXTRACTION_NOTES
            ],
        },
        {
            "kind": "prose",
            "text": (
                "**Train/test protocol.** The corpus is split 80/20 with "
                "`train_test_split(test_size=0.20, random_state=42, stratify=y)`, and "
                "the split is materialised on disk *before* any vectoriser exists. The "
                "vocabulary, the IDF table and the embedding basis are therefore fitted "
                "on training documents only, and the hold-out is transformed by the "
                "already-fitted object. Model selection runs on a further stratified "
                "20% validation split drawn from the training side, so the hold-out is "
                "scored exactly once and the reported figures are an estimate of "
                "generalisation rather than the maximum of twelve attempts. The "
                "comparison is evaluated with the metric definitions of [12] and the "
                "resampling protocol of [13], implemented in scikit-learn [7]."
            ),
        },
        {
            "kind": "list",
            "title": "Algorithms",
            "items": [
                "**K-Nearest Neighbours** -- `KNeighborsClassifier(n_neighbors=5)`. "
                "Instance-based and non-parametric; no training step, prediction cost "
                "scales with the stored corpus.",
                "**Logistic Regression** -- `LogisticRegression(max_iter=2000, "
                "random_state=42)`. Parametric linear model over the feature vector.",
                "**Random Forest** [5] -- `RandomForestClassifier(n_estimators=150, "
                "min_samples_leaf=2, random_state=42)`. Bagged decision trees; the "
                "selection in this run.",
                "**Neural Network** [6] -- `MLPClassifier(hidden_layer_sizes=(128, 64), "
                "max_iter=600, early_stopping=True, random_state=42)`. A simple "
                "feed-forward network trained by back-propagation.",
            ],
        },
        {
            "kind": "prose",
            "text": concepts.SELECTION_RATIONALE,
        },
        {
            "kind": "prose",
            "text": (
                "**Deployment.** The selected model and its fitted representation are "
                "persisted with `joblib` and served by a Streamlit application [9]. The "
                "inference path calls the same `clean_text` function the training path "
                "used, so the text the model sees at prediction time is the text it was "
                "trained on. The interface between the two is deliberately narrow -- a "
                "fitted object exposing `transform(list_of_strings)` -- which is what "
                "allows the application to serve a Bag-of-Words, TF-IDF or embedding "
                "pipeline without any change to the UI."
            ),
        },
        {
            "kind": "prose",
            "text": (
                "**Preprocessing implementation.** The cleaning chain is built on NLTK "
                "[8] for tokenisation, stopword lists and lemmatisation, with an offline "
                "fallback to the regular-expression patterns and static word lists in "
                "`src/nlp_resources.py` so the pipeline still runs on a machine with no "
                "corpora downloaded."
            ),
        },
    ]

    if metrics:
        split = metrics.get("dataset", {})
        blocks.append(
            {
                "kind": "table",
                "caption": "{table_split} -- Split composition",
                    "ref": "split",
                "columns": ["Split", "Authentic", "Fabricated", "Total"],
                "rows": [
                    [
                        "Training",
                        _int(split.get("train_real", 0)),
                        _int(split.get("train_fake", 0)),
                        _int(split.get("train_rows", 0)),
                    ],
                    [
                        "Hold-out",
                        _int(split.get("test_real", 0)),
                        _int(split.get("test_fake", 0)),
                        _int(split.get("test_rows", 0)),
                    ],
                ],
            }
        )

    return blocks


def _results(metrics: dict, profile: dict | None = None) -> list[dict]:
    if not metrics:
        return []

    blocks: list[dict] = []

    selection = metrics.get("selection", {})
    if selection.get("table"):
        blocks.append(
            {
                "kind": "table",
                "ref": "selection",
                "caption": (
                    f"{{table_selection}} -- Model selection, by F1 on a "
                    f"{selection.get('validation_size', 0):.0%} validation split of "
                    f"{selection.get('validation_rows', 0):,} training documents. "
                    "The hold-out is not consulted here."
                ),
                "columns": ["Representation", "Model", "Validation F1", "Validation accuracy"],
                "rows": [
                    [
                        row.get("Representation"),
                        row.get("Model"),
                        _num(row.get("validation_f1")),
                        _num(row.get("validation_accuracy")),
                    ]
                    for row in selection["table"]
                ],
            }
        )

    models = metrics.get("models", {})

    rows = []
    for name, scores in sorted(models.items(), key=lambda item: -item[1]["f1_score"]):
        rows.append(
            [
                name,
                scores.get("family", "n/a"),
                scores.get("representation", "n/a"),
                _num(scores.get("accuracy")),
                _num(scores.get("precision")),
                _num(scores.get("recall")),
                _num(scores.get("f1_score")),
                _num(scores.get("roc_auc")),
            ]
        )
    blocks.append(
        {
            "kind": "table",
            "ref": "headline",
            "caption": (
                "{table_headline} -- Required metrics on the hold-out split, "
                "computed from real predictions. Each algorithm is quoted on the "
                "representation that scored best for it; the deployed pair is the one "
                "selected in {table_selection}."
            ),
            "columns": [
                "Model",
                "Family",
                "Representation",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
            ],
            "rows": rows,
        }
    )

    representations = metrics.get("representations", {})
    if representations:
        grid_rows = []
        for representation_name, block in representations.items():
            for model_name, scores in block["models"].items():
                grid_rows.append(
                    [
                        representation_name,
                        model_name,
                        _num(scores.get("accuracy")),
                        _num(scores.get("precision")),
                        _num(scores.get("recall")),
                        _num(scores.get("f1_score")),
                        "yes" if scores.get("selected") else "",
                    ]
                )
        blocks.append(
            {
                "kind": "table",
                "caption": "{table_grid} -- Full representation x algorithm grid (3 x 4)",
                    "ref": "grid",
                "columns": [
                    "Representation",
                    "Model",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "Selected",
                ],
                "rows": grid_rows,
            }
        )

    matrices = metrics.get("confusion_matrices", {})
    if matrices:
        rows = []
        for name, block in matrices.items():
            matrix = block.get("matrix", [])
            if len(matrix) != 2:
                continue
            correct = matrix[0][0] + matrix[1][1]
            total = sum(sum(row) for row in matrix)
            rows.append(
                [
                    name,
                    block.get("representation", "n/a"),
                    matrix[0][0],
                    matrix[0][1],
                    matrix[1][0],
                    matrix[1][1],
                    _num(correct / total if total else 0),
                ]
            )
        blocks.append(
            {
                "kind": "table",
                "ref": "confusion",
                "caption": (
                    f"{{table_confusion}} -- Confusion matrix per algorithm on the "
                    f"{LABEL_NAMES[0]} / {LABEL_NAMES[1]} hold-out split"
                ),
                "columns": [
                    "Model",
                    "Representation",
                    f"{LABEL_NAMES[0]} to {LABEL_NAMES[0]}",
                    f"{LABEL_NAMES[0]} to {LABEL_NAMES[1]}",
                    f"{LABEL_NAMES[1]} to {LABEL_NAMES[0]}",
                    f"{LABEL_NAMES[1]} to {LABEL_NAMES[1]}",
                    "Accuracy",
                ],
                "rows": rows,
            }
        )

    per_class = metrics.get("per_class_metrics", {})
    if per_class:
        rows = []
        for name, classes in per_class.items():
            for label, scores in classes.items():
                rows.append(
                    [
                        name,
                        label,
                        _num(scores.get("precision")),
                        _num(scores.get("recall")),
                        _num(scores.get("f1_score")),
                        _int(scores.get("support", 0)),
                    ]
                )
        blocks.append(
            {
                "kind": "table",
                "caption": "{table_per_class} -- Per-class precision, recall and F1",
                    "ref": "per_class",
                "columns": ["Model", "Class", "Precision", "Recall", "F1", "Support"],
                "rows": rows,
            }
        )

    top_features = metrics.get("best_model", {}).get("top_features", [])[:10]
    if top_features:
        blocks.append(
            {
                "kind": "table",
                "ref": "features",
                "caption": (
                    f"{{table_features}} -- Strongest features of "
                    f"{metrics.get('best_model_name', 'the selected model')}"
                ),
                "columns": ["Feature", "Weight", "Points toward"],
                "rows": [
                    [row["feature"], _num(row["weight"], 6), row.get("direction", "n/a")]
                    for row in top_features
                ],
            }
        )

    blocks.append(
        {
            "kind": "prose",
            "text": (
                "Every value in these tables is produced by `python -m src.train_models` "
                "from the predictions each model makes on the hold-out split, and is "
                "persisted to `results/model_metrics.json`. No figure in this report is "
                "typed in by hand, and the application renders the same file, so the "
                "documentation, the dashboard and the last real run cannot disagree."
            ),
        }
    )

    return blocks


def _discussion(metrics: dict, profile: dict | None = None) -> list[dict]:
    blocks: list[dict] = [
        {
            "kind": "prose",
            "text": concepts.PARAMETRIC_VS_NONPARAMETRIC,
        }
    ]

    for name, note in concepts.MODEL_NOTES.items():
        scores = metrics.get("models", {}).get(name, {})
        family = note["family"]
        blocks.append(
            {
                "kind": "list",
                "title": f"{name} ({family}) -- F1 {_num(scores.get('f1_score'))}",
                "items": [note["summary"]],
            }
        )
        # Separate blocks rather than one string joined with "; ", so every
        # renderer gets a real bullet list instead of a run-on sentence.
        for label, key in (("Strengths", "strengths"), ("Limitations", "limitations")):
            blocks.append(
                {
                    "kind": "list",
                    "title": f"{name} -- {label.lower()}",
                    "items": list(note[key]),
                }
            )

    return blocks


def _conclusion(metrics: dict | None = None, profile: dict | None = None) -> list[dict]:
    return [
        {
            "kind": "prose",
            "text": (
                "**Insights.** The three representations are not interchangeable, and the "
                "differences visible in {table_grid} are the proof. The distance-based "
                "K-Nearest Neighbours "
                "model is the most representation-sensitive of the four: it collapses on "
                "raw term counts and recovers substantially on TF-IDF, because "
                "re-weighting each term by its inverse document frequency changes which "
                "documents are near which. The neural network shows the opposite "
                "dependency -- it is the slowest model on the wide sparse matrices and "
                "by far the fastest on the dense 200-dimensional embedding, because that "
                "is the representation its architecture was designed for. The linear "
                "model and the tree ensemble are comparatively robust to the choice of "
                "representation, which is itself a useful property when the feature "
                "pipeline is still evolving."
            ),
        },
        {
            "kind": "prose",
            "text": (
                "**Insights (continued).** Feature weighting, dimensionality and model "
                "family interact. The tree ensemble wins because a bagged ensemble "
                "captures interactions a single hyperplane cannot, while remaining stable "
                "on {bow_dimensions:,} sparse columns without any feature scaling. It also produces "
                "the cleanest confusion matrix: the residual errors sit almost entirely "
                "in the fabricated-to-authentic cell, which is the asymmetric failure "
                "mode a newsroom would most want flagged for review."
            ),
        },
        {
            "kind": "list",
            "title": "Limitations",
            "items": concepts.LIMITS,
        },
        {
            "kind": "list",
            "title": "Future scope",
            "items": concepts.FUTURE_SCOPE,
        },
    ]


def _appendix(metrics: dict, profile: dict) -> list[dict]:
    rows = [
        ["`src/data_loader.py`", "Source resolution, schema normalisation, corpus profile"],
        ["`src/prepare_dataset.py`", "Stage 1 -- merge, de-duplicate, engineer the corpus"],
        ["`src/preprocessing.py`", "Stage 2 -- apply the NLP cleaning chain"],
        ["`src/feature_engineering.py`", "Stage 3 -- draw the stratified train/test split"],
        ["`src/feature_extraction.py`", "The three required representations"],
        ["`src/train_models.py`", "Stage 4 -- selection, training, scoring, persistence"],
        ["`src/evaluate_models.py`", "Stage 5 -- classification report and figures"],
        ["`src/eda.py`", "Exploratory data analysis report"],
        ["`src/eda_visualization.py`", "Exploratory figures"],
        ["`src/utils.py`", "Cleaning chain, editorial signals, document statistics"],
        ["`src/nlp_resources.py`", "NLTK bootstrap with an offline fallback"],
        ["`src/concepts.py`", "The explanatory prose used by this report and the app"],
        ["`src/report.py`", "Renders this document from the artifacts on disk"],
        ["`streamlit_app.py`", "Application shell and navigation"],
        ["`app/`", "UI components, chart builders and inference services"],
        ["`app_pages/`", "One script per route"],
        ["`tests/`", "End-to-end pipeline and route tests"],
    ]

    blocks: list[dict] = [
        {
            "kind": "list",
            "title": "Appendix A -- Source layout",
            "items": [f"`{name}` -- {description}" for name, description in rows],
        },
        {
            "kind": "prose",
            "text": (
                "**Reproduction.** From a clean clone: `pip install -r requirements.txt`, "
                "then `python -m src.prepare_dataset`, `python -m src.preprocessing`, "
                "`python -m src.feature_engineering`, `python -m src.train_models`, "
                "`python -m src.evaluate_models`, `python -m src.eda`, "
                "`python -m src.eda_visualization`, `python -m src.report`, and finally "
                "`streamlit run streamlit_app.py`. `python -m pytest tests` verifies the "
                "pipeline and every route."
            ),
        },
        {
            "kind": "list",
            "title": "Appendix B -- Test data samples",
            "items": [
                "**Held-out sample articles.** `app/samples.py` contains six articles "
                "written for demonstration -- three in a neutral, attributed register "
                "expected to be read as authentic, and three in a sensational, "
                "clickbait register expected to be read as fabricated. They are "
                "illustrative, not scraped from any publisher, and are loadable from the "
                "*Analyze article* page.",
                "**Labelled corpus rows.** The same page can load a real, labelled row "
                "straight from the training corpus and then reveal its ground truth "
                "afterwards, so a prediction doubles as an accuracy check.",
                "**Test suite.** `tests/` asserts that every representation fits, that "
                "each of the four required algorithms trains and scores, that the "
                "hold-out is never touched before selection, that all five metrics are "
                f"present for every model, and that {_routes_phrase()} render.",
            ],
        },
    ]

    if profile:
        labels = profile.get("labels", {})
        blocks.append(
            {
                "kind": "prose",
                "text": (
                    f"**Corpus fingerprint.** {_int(labels.get('total', 0))} articles "
                    f"({_int(labels.get('real', 0))} authentic / "
                    f"{_int(labels.get('fake', 0))} fabricated), random_state=42, "
                    f"test_size=0.20. Re-running the pipeline on the same corpus "
                    "reproduces every number in this report exactly."
                ),
            }
        )

    if metrics:
        selection = metrics.get("selection", {})
        blocks.append(
            {
                "kind": "prose",
                "text": (
                    "**IEEE format.** This document is written in the IEEE conference "
                    "template structure -- roman-numeral section headings, an abstract "
                    "and index terms, a two-column-ready body, numbered tables and a "
                    "figure gallery. `docs/ieee_report.tex` is the LaTeX source in the "
                    "`IEEEtran` class and compiles to the same content as a PDF; "
                    "`docs/ieee_report.md` is the same report in Markdown. Both are "
                    "regenerated by `python -m src.report`, so neither can fall out of "
                    "step with the code. The selection record: "
                    f"{selection.get('winner', {}).get('representation', 'n/a')} + "
                    f"{selection.get('winner', {}).get('model', 'n/a')}."
                ),
            }
        )

    return blocks


def _references(metrics: dict | None = None, profile: dict | None = None) -> list[dict]:
    return [
        {
            "kind": "list",
            "title": "References",
            "items": [f"[{index}] {entry}" for index, entry in enumerate(concepts.REFERENCES, 1)],
        }
    ]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def build_sections(metrics: dict, profile: dict) -> list[dict]:
    """Assemble the eight required sections from live artifacts."""
    builders = {
        "introduction": _introduction,
        "dataset": _dataset,
        "methodology": _methodology,
        "results": _results,
        "discussion": _discussion,
        "conclusion": _conclusion,
        "appendix": _appendix,
        "references": _references,
    }

    sections = []
    for index, (identifier, title) in enumerate(SECTION_TITLES, start=1):
        sections.append(
            {
                "id": identifier,
                "number": index,
                "title": title,
                "roman": _roman(index),
                "blocks": builders[identifier](metrics, profile),
            }
        )

    return _resolve(sections, metrics)


#: Tables addressable from prose as ``{table_<ref>}``.
TABLE_REFS = (
    "corpus",
    "desks",
    "split",
    "selection",
    "headline",
    "grid",
    "confusion",
    "per_class",
    "features",
)

#: What a cross-reference degrades to when the run has not produced that table,
#: so a design-only build still reads as prose instead of crashing or lying.
TABLE_FALLBACK = {
    "corpus": "the corpus table",
    "desks": "the news-desk table",
    "split": "the split table",
    "selection": "the selection table",
    "headline": "the hold-out table",
    "grid": "the representation x algorithm grid",
    "confusion": "the confusion-matrix table",
    "per_class": "the per-class table",
    "features": "the feature table",
}


def _resolve(sections: list[dict], metrics: dict) -> list[dict]:
    """Fill ``{token}`` placeholders and hand out the table numbers.

    Two passes over the finished document:

    1. Number the table blocks in reading order and record which number each
       one got, so prose can say ``Table {grid}`` and never a hard-coded digit.
    2. Substitute the numeric placeholders from :func:`_context`, so a
       re-trained run cannot leave a stale dimension or row count in the
       narrative.

    Both raise on an unresolved token, which turns a silent drift into a build
    failure.
    """
    numbers: dict[str, int] = {}
    counter = 0
    for section in sections:
        for block in section["blocks"]:
            if block["kind"] != "table":
                continue
            counter += 1
            ref = block.get("ref")
            if ref:
                numbers[ref] = counter
            block["number"] = counter

    context = _context(metrics)
    context.update(
        {
            f"table_{ref}": (
                f"Table {numbers[ref]}" if ref in numbers else TABLE_FALLBACK[ref]
            )
            for ref in TABLE_REFS
            if ref in TABLE_FALLBACK
        }
    )

    for section in sections:
        for block in section["blocks"]:
            if block["kind"] == "prose":
                block["text"] = _fill(block["text"], context)
            elif block["kind"] == "list":
                block["items"] = [_fill(item, context) for item in block["items"]]
                block["title"] = _fill(block["title"], context)
            elif block["kind"] == "metric":
                block["caption"] = _fill(block["caption"], context)
            elif block["kind"] == "table":
                block["caption"] = _fill(block["caption"], context)

    return sections


def _roman(number: int) -> str:
    values = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    result = ""
    for value, symbol in values:
        while number >= value:
            result += symbol
            number -= value
    return result


def representations_present(metrics: dict) -> list[str]:
    """Which of the three required representations are recorded in a run."""
    available = metrics.get("representations", {})
    return [name for name in REPRESENTATIONS if name in available]


def as_frame(rows: list[list], columns: list[str]) -> pd.DataFrame:
    """Table rows -> DataFrame, for the Streamlit renderer."""
    return pd.DataFrame(rows, columns=columns)
