# Veritas — News Authenticity Analyzer

A Streamlit application that classifies news articles as **real** or **fabricated**
and shows the evidence behind every verdict. Built as an academic NLP and machine
learning project: three feature representations and four classifiers are
benchmarked on the same hold-out split, the winner is chosen on a validation
split, and it is served alongside an explanation of the terms that drove each
decision.

> Veritas estimates authenticity from text alone. It is a research tool, not a
> fact-checker — a confident verdict is a model prediction, not a guarantee.

---

## Contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Application tour](#application-tour)
- [How it works](#how-it-works)
- [Results](#results)
- [The report](#the-report)
- [Project structure](#project-structure)
- [The pipeline](#the-pipeline)
- [Data](#data)
- [Design system](#design-system)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## Highlights

- **Twelve combinations, one honest selection.** Three representations × four
  classifiers, all fitted on the training split only. The winner is chosen on a
  validation split carved from the training side, so the hold-out is scored
  exactly once.
- **Three required representations.** Bag-of-Words (raw counts), TF-IDF
  (inverse document frequency) and latent-semantic embeddings (TF-IDF → truncated
  SVD → L2 norm, 200 dimensions).
- **Evidence, not just a label.** Every prediction ships with a term-attribution
  chart, class probabilities, document statistics and five editorial signals
  benchmarked against the corpus average.
- **Model-agnostic explanations.** Linear models are explained by signed
  coefficient; tree ensembles by term occlusion on the feature matrix, so the
  feature works whichever model wins the benchmark.
- **A confusion matrix per algorithm,** with per-class precision, recall, F1 and
  support, plus the parametric versus non-parametric discussion the assignment
  asks for.
- **Batch scoring.** Upload a CSV, map the text and headline columns, score up to
  20,000 rows in one vectorised pass, filter the results and download them.
- **A real audit trail.** Every analysis is appended to a local CSV with its
  confidence, filterable and exportable from the UI.
- **Auditable corpus browser.** Search and filter all 1,999 prepared articles,
  inspect distributions, and read the full text of any record.
- **A generated IEEE report.** `python -m src.report` writes the abstract, eight
  numbered sections, nine measured tables and a figure gallery to `docs/` as
  Markdown, LaTeX, HTML and PDF — every number read from `results/`, never
  typed in by hand.
- **Honest empty states.** If the pipeline has not been built, the app says so
  and prints the exact commands to build it.
- **No custom CSS.** The entire visual system is native Streamlit primitives and
  theme tokens, so light and dark mode both work.

## Quick start

```bash
git clone <your-fork-url>
cd AI_Fake_News_Detection

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Build the pipeline once, then launch:

```bash
python -m src.prepare_dataset     # merge, de-duplicate, label and engineer the corpus
python -m src.preprocessing       # NLP cleaning
python -m src.feature_engineering # stratified 80/20 split, written before any fitting
python -m src.train_models        # benchmark 3 x 4, select on validation, persist the winner

streamlit run streamlit_app.py
```

The app opens on <http://localhost:8501>. On the bundled corpus the cleaning
stage takes about a minute and training about ten seconds.

## Application tour

| Page | What it does |
|---|---|
| **Overview** | What the tool is, the selected model's headline metrics, the transformation pipeline, the confusion matrix and the ROC curve. |
| **Analyze article** | The main flow. Paste text, load a curated example, or pull a labelled corpus row and watch the verdict resolve against ground truth. |
| **Batch analysis** | Upload a CSV, map its columns, score every row, filter, and download the results. |
| **Model lab** | The full representation × algorithm grid, a rankable metric chart, a confusion matrix per algorithm, the ROC curve, the parametric versus non-parametric discussion, per-model roles and the terms the model leans on. |
| **Dataset explorer** | Class balance, per-desk counts, length distributions, engineered-signal comparisons and a searchable record browser. |
| **Concepts** | How AI, supervised machine learning, NLP and evaluation fit together — the preprocessing chain, what each representation measures, and what the benchmark can and cannot tell you. |
| **Prediction log** | Every analysis this installation has recorded, with filters, a confidence trend, CSV export and deletion. |
| **Report** | The full IEEE-formatted write-up, rendered from the same artifacts the rest of the app reads, with downloads for the Markdown and LaTeX sources. |
| **Pipeline & setup** | Artefact health, the methodology step by step, and the commands to rebuild everything. |

## How it works

### Text pipeline

Every document -- at training time and at inference time -- passes through the
same chain:

1. **Normalise** — lowercase, fold Unicode, strip URLs, HTML, digits and punctuation.
2. **Tokenise** — split on whitespace, drop stopwords and tokens under three characters.
3. **Lemmatise** — reduce each token to a shared dictionary form.
4. **Represent** — one of three fitted representations (below).
5. **Classify** — a probability for real versus fake.

Steps 1–3 live in `src/utils.py`; step 4 in `src/feature_extraction.py`.

### The three representations

| Representation | Method | What it preserves |
|---|---|---|
| **Bag-of-Words** | `CountVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)` | How often a term appears |
| **TF-IDF** | `TfidfVectorizer(same arguments, sublinear_tf=True)` | How *discriminative* a term is |
| **Embeddings (LSA)** | TF-IDF → `TruncatedSVD(200)` → L2 normalise | Topical similarity between documents that share no vocabulary |

Each is a small class with the same `fit` / `transform` / `feature_names()`
contract, so the application serves whichever one won the benchmark without any
change to the UI.

### Editorial signals

Five rule-based features are computed alongside the model:

| Feature | Reads as |
|---|---|
| `clickbait_score` | Headline phrasing, shouting, exclamation marks |
| `emotional_word_ratio` | Share of loaded, alarmist vocabulary |
| `sentiment_score` | Negative values read as hostile framing |
| `readability_score` | Flesch Reading Ease; higher is plainer |
| `source_trust_score` | Neutral prior — the corpus has no publisher identity |

They are **not** fed to the classifier. The benchmark therefore measures the
text model rather than hand-written rules, and the signals are shown separately
as an editorial reading of the document. The corpus averages they are compared
against are measured from the built dataset at runtime — there are no hard-coded
fallbacks.

### Model selection

Selection is by **F1 score on a validation split** carved out of the *training*
documents. F1 is the only headline metric that weighs precision and recall
together, which matters when a missed fabricated article costs more than a false
alarm; and choosing on validation rather than on the hold-out means the reported
hold-out numbers are an estimate of generalisation rather than the best of
twelve attempts.

### Explanations

| Winning model | Method |
|---|---|
| Linear (`coef_` available) | Signed coefficient per term, scaled by the term's feature value. |
| Everything else | Occlusion: the top terms are zeroed one at a time in the feature matrix and the change in the predicted probability is the contribution. |

Both paths return the same structure, so the UI never branches on model type.

## Results

1,999 prepared articles (836 real, 1,163 fake); 1,997 survive cleaning and are
split 80/20 with `random_state=42`, leaving 1,597 training and 400 hold-out
documents (167 real, 233 fake).

### Full representation × algorithm grid — hold-out F1

| Representation | Random Forest | Logistic Regression | Neural Network | K-Nearest Neighbors |
|---|---:|---:|---:|---:|
| **Bag-of-Words** | **0.9957** ◀ selected | 0.9935 | 0.9829 | 0.7993 |
| **TF-IDF** | 0.9935 | 0.9872 | 0.9829 | 0.8995 |
| **Embeddings (LSA)** | 0.9427 | 0.9871 | 0.9850 | 0.9131 |

### The four required algorithms

Each quoted on the representation that scored best for it; the deployed model is
the one selected on validation, which is also the highest-scoring combination.

| Model | Family | Representation | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Random Forest** ◀ selected | Non-parametric (ensemble) | Bag-of-Words | **0.9950** | **1.0000** | **0.9914** | **0.9957** | **0.9998** | 0.62 s |
| Logistic Regression | Parametric | Bag-of-Words | 0.9925 | 1.0000 | 0.9871 | 0.9935 | 0.9994 | 0.06 s |
| Neural Network | Parametric (discriminative) | Embeddings (LSA) | 0.9825 | 0.9829 | 0.9871 | 0.9850 | 0.9993 | 0.95 s |
| K-Nearest Neighbors | Non-parametric | Embeddings (LSA) | 0.9025 | 0.9491 | 0.8798 | 0.9131 | 0.9662 | 0.00 s |

### Confusion matrix per algorithm

| Model | Real → Real | Real → Fake | Fake → Real | Fake → Fake |
|---|---:|---:|---:|---:|
| **Random Forest** (deployed) | 167 | 0 | 2 | 231 |
| Logistic Regression | 167 | 0 | 3 | 230 |
| Neural Network | 163 | 4 | 3 | 230 |
| K-Nearest Neighbors | 156 | 11 | 28 | 205 |

The deployed model's residual errors sit almost entirely in the
fabricated-to-authentic cell — the asymmetric failure mode a newsroom would most
want flagged for review.

Strongest features of the selected model: `reuters` (toward REAL), `video` and
`via` (toward FAKE), `said` and `washington reuters` (toward REAL).

The full record lives in `results/model_metrics.json` and the grid in
`results/representation_benchmark.csv`; both are what the app renders, so the UI
can never drift from the last real run. Every figure above is regenerated by
`python -m src.train_models` and asserted by the test suite.

## The report

```bash
python -m src.report
```

writes:

- `docs/ieee_report.md` — abstract, index terms, eight numbered sections
  (Introduction, Dataset Description, Methodology, Results, Discussion,
  Conclusion, Appendix, References), nine measured tables and a figure gallery
  covering every generated figure.
- `docs/ieee_report.tex` — the same document in the `IEEEtran` conference class,
  with wide tables spanning both columns and the figure gallery as full-width
  floats.
- `docs/ieee_report.html` — a two-column, printable rendering.
- `docs/ieee_report.pdf` — produced from whichever toolchain the machine has:
  `pdflatex` if a TeX distribution is installed, otherwise the HTML build
  printed by a headless Chrome or Edge. If neither is present the three source
  files are still written and the **Report** page still renders the whole
  document, so the step never fails the pipeline.

The `.html` and `.pdf` are build outputs and are git-ignored; regenerate them
with `python -m src.report`. All four renderings plus the in-app **Report** page
come from the same `src/report_sections.py` builder and the same `abstract()`
string, so the document, the dashboard and the last training run cannot
disagree.

## Project structure

```
AI_Fake_News_Detection/
├── streamlit_app.py            # entry point: config, state, sidebar, navigation
├── .streamlit/
│   └── config.toml             # light/dark theme, typography, semantic colours
├── app/
│   ├── navigation.py           # the route map, shared by the shell and the tests
│   ├── samples.py              # illustrative sample articles
│   ├── components/             # reusable UI units
│   │   ├── cards.py            # KPI strips, verdict panel, evidence tables
│   │   ├── charts.py           # Altair builders (one chart per function)
│   │   └── layout.py           # brand, page headers, empty states, footer
│   └── services/               # all logic, no rendering
│       ├── artifacts.py        # cached model/metric/dataset loading
│       ├── corpus_stats.py     # corpus signal averages, measured at runtime
│       ├── history.py          # bounded, atomic CSV prediction log
│       └── inference.py        # predict, explain, batch-score
├── app_pages/                  # one script per route; UI only
│   ├── overview.py
│   ├── analyze.py
│   ├── batch.py
│   ├── model_lab.py
│   ├── dataset.py
│   ├── concepts.py
│   ├── history.py
│   ├── report.py
│   └── pipeline.py
├── src/                        # the ML pipeline
│   ├── config.py               # every path and hyper-parameter
│   ├── data_loader.py          # source selection and schema normalisation
│   ├── prepare_dataset.py      # merge, de-duplicate, label, engineer
│   ├── preprocessing.py        # NLP cleaning
│   ├── nlp_resources.py        # NLTK download and lookups
│   ├── utils.py                # cleaning and editorial signals
│   ├── feature_extraction.py   # the three representations
│   ├── feature_engineering.py  # the leakage-safe stratified split
│   ├── train_models.py         # 3 x 4 benchmark, validation selection, persistence
│   ├── evaluate_models.py      # metrics, ROC, feature importance, figures
│   ├── eda.py                  # exploratory report and dataset profile
│   ├── eda_visualization.py    # exploratory figures
│   ├── concepts.py             # the explanatory prose for the app and the report
│   ├── report_sections.py      # the eight document sections, built from artifacts
│   ├── report.py               # writes the Markdown, LaTeX and HTML report
│   └── report_pdf.py           # LaTeX / headless-browser PDF rendering
├── data/                       # see data/README.md
├── docs/                       # generated report (md + tex tracked)
├── models/                     # generated
├── results/                    # generated
├── images/                     # generated
└── tests/
    ├── conftest.py
    ├── test_representations.py     # the fit/transform/label contract
    ├── test_metrics_contract.py    # the benchmark and leakage guarantees
    └── test_app_pages.py           # every route renders; a real prediction works
```

Pages are plain scripts that only render. Business logic lives in
`app/services`, the ML pipeline lives in `src`, and every path comes from
`src/config.py` — there are no hard-coded strings pointing at the filesystem.

## The pipeline

Run the stages you need from the project root:

```bash
python -m src.prepare_dataset      # merge, de-duplicate, engineer -> data/GlobalFakeNews_Research2026_v1.csv
python -m src.preprocessing        # NLP cleaning               -> data/cleaned_news.csv
python -m src.feature_engineering  # stratified 80/20 split     -> models/{train,test}_documents.pkl
python -m src.train_models         # select on validation, then score once -> models/*.pkl, results/
python -m src.evaluate_models      # benchmark + figures        -> images/
python -m src.eda                  # corpus profile             -> results/dataset_profile.json
python -m src.eda_visualization    # exploratory charts         -> images/
python -m src.report               # IEEE report               -> docs/ieee_report.{md,tex,html,pdf}
```

Only `models/best_model.pkl` and `models/vectorizer.pkl` are required to return
a verdict. Everything else is supporting evidence.

## Data

The bundled `data/sample_dataset.csv` is a seeded, stratified 2,000-row sample
with a 13 MB footprint — reproducible in git without a large download. Add
`Fake.csv` and `True.csv` to `data/` and re-run the pipeline to switch to the
full 44,898-article dataset; the loader prefers them automatically and no code
changes are needed.

See [data/README.md](data/README.md) for the schema, the class balance and
provenance.

## Design system

The visual language is expressed entirely through native Streamlit primitives
and theme tokens in `.streamlit/config.toml`:

- **Type** — Outfit for headings, Inter for body, JetBrains Mono for code.
- **Colour** — a single indigo accent, semantic green for real and rose for fake,
  every colour defined per theme so light and dark both hold contrast.
- **Shape** — `baseRadius` and per-widget radii, hairline borders, and bordered
  `st.container` blocks in place of hand-drawn cards.
- **Layout** — wide layout, wide content gutters, horizontal KPI strips that
  wrap on narrow viewports.

There is no `st.markdown` with raw HTML and no injected stylesheet.

## Testing

```bash
python -m pytest tests -q
```

The suite has three layers:

- `tests/test_representations.py` builds each representation from scratch and
  asserts the fit/transform/label contract: shapes match the described
  dimensions, a second `transform` does not refit, unseen documents cannot grow
  the vocabulary, embedding rows are unit-norm, and all four algorithms train on
  all three representations.
- `tests/test_metrics_contract.py` audits `results/model_metrics.json`: all three
  representations, all four algorithms, all five metrics per model, a confusion
  matrix per algorithm that sums to the hold-out size, per-class breakdowns, a
  stratified split that reconciles back to the source row count, and a selection
  made on validation data. It also performs the leakage check — the persisted
  vocabulary must equal a refit on the training documents, and no term that
  occurs only in the hold-out may appear in it.
- `tests/test_app_pages.py` runs every route headlessly with Streamlit's own
  harness, asserts the report page renders all eight sections, and drives a real
  prediction end to end.

Individually, a page can be run the same way without a browser:

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app_pages/analyze.py", default_timeout=120).run()
at.text_area(key="draft_body").set_value("Some article text long enough to score.")
at.button(key="analyze_run").click().run()

assert not at.exception
```

## Troubleshooting

**The sidebar is empty and each page shows a red banner.**
The trained artifacts are missing. Run, in order:

```bash
python -m src.prepare_dataset
python -m src.preprocessing
python -m src.feature_engineering
python -m src.train_models
```

Then reload. The **Pipeline & setup** page lists every artifact and its path.

**`UsageError: Missing pipeline artifacts` from pytest.**
The test suite asserts the artifacts exist rather than skipping. Build the
pipeline with the four commands above, then re-run.

**`LookupError` or a NLTK error on first run.**
`src/nlp_resources.py` downloads the `punkt`, `stopwords` and `wordnet`
resources on demand. If the machine is offline, the pipeline falls back to
built-in stopword lists and skips lemmatisation; accuracy will be lower and
`results/model_metrics.json` will reflect that.

**`python -m src.report` skipped the PDF.**
No TeX distribution and no headless browser were found. The Markdown, LaTeX and
HTML sources are still written and the **Report** page still renders the full
document. Install MiKTeX or TeX Live and re-run to get the `IEEEtran` build, or
open `docs/ieee_report.html` in any browser and print it.

**A verdict looks wrong for a short article.**
Inputs under five words are rejected rather than guessed at. Paste the full body.

**Port 8501 is already in use.**

```bash
streamlit run streamlit_app.py --server.port 8502
```

## Limitations

- The model reads **text only**. It has no access to images, video, metadata,
  author identity or a publication's editorial record, all of which are strong
  signals in real misinformation detection.
- The corpus is drawn from a limited set of newsrooms and political contexts.
  Performance on unrelated topics, other languages or contemporary events will be
  lower than the benchmark above suggests.
- A high F1 score on a held-out split from the same distribution is not a
  measure of real-world generalisation. Treat the numbers as a floor, not a
  forecast.
- The corpus carries no verifiable publisher identity, so the model can learn
  which outlets appear in fabricated articles without learning anything about
  whether an outlet is trustworthy.
- Bag-of-words features discard word order, so a claim and its negation are
  represented identically.
- Veritas reports a probability, not a fact-check. A confident **real** verdict
  is not a guarantee that a claim is true.

---

Built for educational and research use. Python · Streamlit · scikit-learn ·
NLTK · Altair.
