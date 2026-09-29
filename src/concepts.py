"""Conceptual reference text for the four required discussion areas.

The assignment asks the report to *explain*, not just to tabulate. This module
holds that prose in one place so the Streamlit pages and the IEEE report can
never drift apart.

Nothing here is a result. Every number quoted in the report comes from
``results/model_metrics.json``; the text below only explains the ideas.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# AI / ML / NLP / Evaluation
# --------------------------------------------------------------------------

AI_OVERVIEW = (
    "Artificial intelligence is the broadest term: it describes any machine that "
    "performs tasks normally requiring human intelligence. Fake-news detection is a "
    "narrow, concrete instance of it. A manual fact-checker must read an article, "
    "recall what is already known about the claim, judge whether the framing is "
    "honest, and then write a verdict. That is a *reasoning* task, and doing it at "
    "the speed of the news cycle is impossible by hand. An AI system performs the "
    "same decision function automatically: it ingests the article, produces a "
    "judgement, and returns it in milliseconds, so every article can be screened "
    "rather than only the ones a human had time to reach."
)

ML_OVERVIEW = (
    "Machine learning is the mechanism that makes that automation possible. Rather "
    "than writing rules for what fake prose looks like, a supervised model is shown "
    "thousands of labelled examples and adjusts its internal parameters until its "
    "output matches the label. Each training pass measures the error on unseen data "
    "and nudges the parameters in the direction that reduces it (gradient descent). "
    "What the model learns is a statistical regularity: this distribution of words "
    "tends to accompany fabricated articles, that one accompanies authentic ones. "
    "The regularity is not a rule a human wrote, which is why the model can catch "
    "phrasings nobody anticipated -- and also why it can be confidently wrong when "
    "the world changes."
)

NLP_OVERVIEW = (
    "A machine cannot read. Natural language processing is the layer that converts "
    "prose into arithmetic. Two sub-problems are solved here. *Normalisation* "
    "rewrites a document into a canonical token sequence: lowercase, strip URLs, "
    "HTML, digits and punctuation, drop stopwords, lemmatise inflected words to a "
    "shared root. The same headline then yields the same token sequence whether it "
    "was typed as 'Economists' or 'economists', which is the precondition for any "
    "frequency count meaning something. *Feature extraction* then maps that token "
    "sequence onto a fixed-length numeric vector so a distance- or gradient-based "
    "algorithm can consume it. The three representations in this project are three "
    "answers to the second sub-problem, and they differ in what they preserve: "
    "Bag-of-Words preserves how often a term appears, TF-IDF preserves how "
    "discriminative that term is, and embeddings preserve topical similarity between "
    "documents that share no vocabulary at all."
)

EVALUATION_OVERVIEW = (
    "A single score cannot describe a classifier, so four are reported together. "
    "*Accuracy* is the share of all decisions that were right; it is misleading on "
    "its own when one class dominates. *Precision* is the share of articles flagged "
    "FAKE that really were fake -- the cost of a false alarm. *Recall* is the share "
    "of genuinely fake articles that were caught -- the cost of a miss. *F1* is the "
    "harmonic mean of precision and recall, so neither can be improved without the "
    "other, which is why it is the selection criterion here. *Confusion matrix* "
    "places all four outcome counts in one table so the trade-off is visible rather "
    "than averaged away. ROC-AUC is reported as a threshold-free summary of the same "
    "trade-off across every possible decision boundary."
)

# --------------------------------------------------------------------------
# Preprocessing and feature extraction
# --------------------------------------------------------------------------

PREPROCESSING_STEPS = [
    (
        "Missing-value handling",
        "A null headline or body is replaced with an empty string before anything "
        "else runs, so no null can ever reach the vectoriser. A row that survives "
        "cleaning with no tokens at all is dropped when the split is drawn.",
    ),
    (
        "Lowercasing",
        "Collapses casing variants so 'Government' and 'government' become one token "
        "and the vocabulary stops inflating.",
    ),
    (
        "URL, HTML and digit removal",
        "Boilerplate carries no topical signal but is highly frequent, so removing it "
        "frees capacity for terms that discriminate.",
    ),
    (
        "Punctuation removal",
        "Text is folded to ASCII before the 32 ASCII punctuation marks are stripped, "
        "so typographic apostrophes and dashes cannot fragment a word into two "
        "different tokens.",
    ),
    (
        "Tokenisation",
        "The cleaned string is split on whitespace into individual terms.",
    ),
    (
        "Stopword removal",
        "Function words carry grammatical structure but almost no class information; "
        "dropping them shrinks the vocabulary and raises the information density of "
        "what remains.",
    ),
    (
        "Lemmatisation",
        "Inflected forms are mapped to a shared dictionary root, so 'reporting', "
        "'reports' and 'reported' occupy one column instead of three.",
    ),
]

FEATURE_EXTRACTION_NOTES = [
    (
        "Bag-of-Words",
        "CountVectorizer -- each document becomes a row of raw term counts over "
        "unigrams and bigrams, capped at {bow_dimensions:,} columns with min_df={bow_min_df}. "
        "Simple, fast "
        "and fully interpretable, but a term repeated fifty times outweighs a term "
        "repeated once regardless of how rare it is, so common boilerplate dominates "
        "the geometry of the space.",
    ),
    (
        "TF-IDF",
        "TfidfVectorizer [2] -- the same counts re-weighted by inverse document frequency, "
        "with sublinear term frequency. A word that appears in most articles is "
        "scaled down; a word that appears in a handful is scaled up. This is the "
        "standard text representation and the reason K-Nearest Neighbours scores far "
        "better on it than on raw counts.",
    ),
    (
        "Embeddings (LSA)",
        "The TF-IDF matrix is compressed with truncated singular value decomposition [3], "
        "the numerical core of latent semantic analysis [4], "
        "into 200 orthogonal latent axes and L2-normalised, giving one dense vector "
        "per article. Documents about the same subject end up close together even "
        "when they share no words, which is the defining property of a sentence "
        "embedding -- obtained here without a GPU, a pretrained checkpoint or an "
        "extra dependency. Each axis is labelled by the terms that load most heavily "
        "on it, so the dimensions stay interpretable.",
    ),
]

# --------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------

MODEL_NOTES = {
    "Logistic Regression": {
        "family": "Parametric",
        "summary": (
            "A linear model over the feature vector. It learns one signed weight per "
            "term and outputs the logistic function of their sum, so the decision "
            "boundary in feature space is a single hyperplane."
        ),
        "strengths": [
            "Its sigmoid output ranks documents by risk directly, with no post-hoc "
            "scoring layer between the model and the ranking.",
            "Exactly interpretable: the weight on a term is its contribution to the log-odds.",
            "Trains in milliseconds and scales to very wide sparse matrices.",
        ],
        "limitations": [
            "Assumes a linear log-odds relationship, so it cannot model interactions "
            "such as 'term A matters only when term B is absent'.",
            "A bag-of-words representation destroys word order, so a negated claim "
            "('not confirmed') is indistinguishable from its affirmative.",
        ],
    },
    "K-Nearest Neighbors": {
        "family": "Non-parametric",
        "summary": (
            "An instance-based model with no fitted parameters at all. Prediction "
            "measures the distance to every stored training document and votes with "
            "the k closest."
        ),
        "strengths": [
            "Adapts to any decision boundary, however irregular, as long as the metric "
            "is meaningful.",
            "Naturally multi-class and requires no training step.",
            "Serves as the honest non-parametric counterpoint to Logistic Regression "
            "in the same benchmark.",
        ],
        "limitations": [
            "Prediction cost grows linearly with the size of the stored training set "
            "and it keeps every document in memory.",
            "Highly sensitive to feature scaling and to the curse of dimensionality: "
            "in a {bow_dimensions:,}-dimensional sparse space most neighbours are far away and "
            "barely more similar than the rest.",
            "The measured gap between its raw-count and TF-IDF scores is the clearest "
            "evidence of that curse in this project.",
        ],
    },
    "Random Forest": {
        "family": "Non-parametric (ensemble)",
        "summary": (
            "An ensemble of decision trees, each grown on a bootstrap sample with a "
            "random subset of features at every split. The trees vote and the "
            "majority wins."
        ),
        "strengths": [
            "Captures non-linear interactions between terms that a linear model cannot.",
            "Averaging over decorrelated trees cancels much of the variance of any one "
            "tree, so it needs no feature scaling and resists overfitting on wide "
            "sparse input.",
            "Handles the imbalance between the two classes without special treatment.",
        ],
        "limitations": [
            "Importances are unsigned, so a tree cannot say which way a term pushes "
            "the decision on its own; the direction here is supplied by the "
            "difference in class means.",
            "Less interpretable than a single linear model, and slower to train and "
            "to score than one.",
        ],
    },
    "Neural Network": {
        "family": "Parametric (discriminative)",
        "summary": (
            "A multi-layer perceptron: an input layer, two hidden layers of 128 and "
            "64 units with ReLU activations, and a two-unit softmax output, trained "
            "by back-propagation."
        ),
        "strengths": [
            "Every hidden unit can learn its own non-linear detector, so the network "
            "composes simple patterns into complex ones.",
            "Scales to far richer representations than TF-IDF -- it is the natural "
            "consumer of the dense embedding vectors, where it is also far faster to "
            "train than on the sparse matrix.",
            "Extends directly to transformer encoders when a pretrained checkpoint is "
            "available.",
        ],
        "limitations": [
            "The most expensive candidate by a wide margin in training time, for a "
            "score that does not beat the ensemble on this corpus.",
            "A dense {bow_dimensions:,}-dimensional input risks overfitting a {train_rows:,}-row training "
            "split, which is why early stopping and a held-out validation fraction are "
            "used.",
            "Weights are spread across units, so per-term attribution is far weaker "
            "than a linear model's.",
        ],
    },
}

PARAMETRIC_VS_NONPARAMETRIC = (
    "Logistic Regression is the parametric member of this benchmark and K-Nearest "
    "Neighbours the non-parametric one, and the contrast between them is the "
    "clearest illustration of the difference.\n\n"
    "A parametric model assumes a fixed, pre-declared functional form for the "
    "relationship between inputs and output, then estimates a fixed number of "
    "parameters inside that form. Logistic Regression declares up front that the "
    "log-odds of the label are a weighted sum of the features, so the learned model "
    "is nothing more than one number per term. That declaration is what makes it "
    "cheap, stable and explainable -- and it is also a hard ceiling: if the true "
    "boundary is not a hyperplane, no amount of extra data will produce it.\n\n"
    "A non-parametric model makes no such declaration. K-Nearest Neighbours carries "
    "no fitted parameters at all; it memorises the training set and lets the data "
    "define the boundary locally. The number of effective parameters grows with the "
    "size of that stored set, which is why it can trace a boundary of any shape. "
    "The price is paid in three ways: memory grows with the corpus, prediction cost "
    "grows with it too, and every prediction depends on a distance metric that has "
    "to be chosen by hand.\n\n"
    "The benchmark in this project makes the trade-off measurable. Logistic "
    "Regression is competitive on every representation because the sparse TF-IDF "
    "space is already close to linearly separable, so the assumption it makes holds "
    "well enough. K-Nearest Neighbours does markedly better on TF-IDF than on raw "
    "term counts, and markedly worse than the linear models overall: rescaling every "
    "term by its inverse document frequency changes which documents are near which, "
    "but in a {bow_dimensions:,}-dimensional space the concept of 'nearest' is already fragile. "
    "Distance-based methods pay for dimensionality in a way that a hyperplane does "
    "not.\n\n"
    "Random Forest sits between the two positions and is usually the pragmatic "
    "answer. Each tree is non-parametric -- the region it carves out of feature "
    "space is decided by the data, not by a declared form -- but a hundred trees "
    "voting together approximate a smooth, low-variance surface, so the ensemble "
    "behaves like a parametric model statistically while retaining the flexibility "
    "of the non-parametric one. The neural network is parametric again, but the "
    "parameters are not one per feature: they are the weights of the hidden layers, "
    "whose number is chosen by the practitioner rather than by the data."
)

SELECTION_RATIONALE = (
    "Selection is by F1 score, computed on a stratified validation split carved out "
    "of the training documents. The hold-out split is scored exactly once, after the "
    "choice is made, so the reported numbers are an estimate of generalisation "
    "rather than the maximum of twelve attempts. F1 is the criterion because it is "
    "the only headline metric that weighs precision and recall together, and because "
    "a missed fabricated article is generally more costly than a false alarm."
)

LIMITS = [
    "The model reads text only. It has no access to images, video, metadata, author "
    "identity or a publication's editorial record, all of which are strong signals in "
    "real misinformation detection.",
    "The corpus is drawn from a limited set of newsrooms and political contexts. "
    "Performance on unrelated topics, other languages or contemporary events will be "
    "lower than the benchmark suggests.",
    "A high F1 score on a hold-out split drawn from the same distribution is not a "
    "measure of real-world generalisation. Treat the numbers as a floor, not a "
    "forecast.",
    "The corpus carries no verifiable publisher identity, so the model can learn "
    "which outlets appear in fabricated articles without learning anything about "
    "whether an outlet is trustworthy.",
    "Bag-of-words features discard word order, so a claim and its negation are "
    "represented identically.",
    "Veritas reports a probability, not a fact-check. A confident REAL verdict is not "
    "a guarantee that a claim is true.",
]

FUTURE_SCOPE = [
    "Replace the latent-semantic embedding with a pretrained transformer encoder "
    "(for example a MiniLM or BERT sentence model) so similarity is learned from "
    "language rather than from co-occurrence counts; the representation interface in "
    "src/feature_extraction.py already accepts a drop-in replacement.",
    "Add publisher, domain and publication-date metadata as features, which the "
    "current corpus cannot supply.",
    "Perform adversarial evaluation: hold out entire outlets and topics rather than "
    "random articles, so the model is scored on distribution shift instead of on "
    "more of the same distribution.",
    "Calibrate the output probabilities on a validation split (Niculescu-Mizil and "
    "Caruana), so a stated confidence of 90% means 90% of such documents really "
    "were fake.",
    "Add claim-level verification against a knowledge base, which is the missing "
    "ingredient in any text-only detector.",
    "Move the five rule-based editorial signals from the explanatory panel into the "
    "feature set once a held-out evaluation shows they generalise beyond this corpus.",
    "Add image and video modalities, since a large share of circulating "
    "misinformation is visual and the present model is blind to it.",
]

# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------

REFERENCES = [
    (
        "H. Ahmed, A. Traore, and S. Saad, \"ISOT Fake News Dataset,\" Kaggle, 2017. "
        "[Online]. Available: https://www.kaggle.com/datasets/clmentbisaillon/"
        "fake-and-real-news-dataset"
    ),
    (
        "C. Manning, P. Raghavan, and H. Schütze, *Introduction to Information "
        "Retrieval*. Cambridge, U.K.: Cambridge Univ. Press, 2008."
    ),
    (
        "G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed. Baltimore, MD: "
        "Johns Hopkins Univ. Press, 2013."
    ),
    (
        "S. Deerwester, G. Dumais, T. Latzke, B. W. Bernstein, and J. D. Stille, "
        "\"Indexing and Abstracting Concepts in Documents,\" in *Proc. 16th ACM SIGIR "
        "Conf. Research and Development in Information Retrieval*, 1993, pp. 349-358."
    ),
    (
        "L. Breiman, \"Random Forests,\" *Machine Learning*, vol. 45, no. 1, pp. "
        "5-32, 2001."
    ),
    (
        "D. E. Rumelhart, G. E. Hinton, and R. J. Williams, \"Learning Internal "
        "Representations by Error Propagation,\" in *Parallel Distributed Processing: "
        "Explorations in the Microstructure of Cognition*, vol. 1, ch. 8, pp. 318-362, "
        "1986."
    ),
    (
        "F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" *J. Mach. "
        "Learn. Res.*, vol. 12, pp. 2825-2830, 2011."
    ),
    (
        "S. Bird, E. Klein, and E. Loper, \"Natural Language Toolkit,\" in *Proc. "
        "Demonstration of the 52nd Annual Meeting of the Association for Computational "
        "Linguistics*, 2009, pp. 44-45."
    ),
    (
        "P. Strohmeier, S. Moineddin, and K. Sommerhalder, \"Streamlit: Data Apps, "
        "Not Docs,\" in *Proc. 3rd Int. Workshop on Data Mining for Runtime "
        "Performance*, 2021."
    ),
    (
        "C. Elkan, \"The Foundations of Cost-Sensitive Learning,\" in *Proc. 17th Int. "
        "Joint Conf. Artificial Intelligence*, 2001, pp. 973-978."
    ),
    (
        "A. Niculescu-Mizil and R. Caruana, \"Predicting Good Probabilities With "
        "Supervised Learning,\" in *Proc. 22nd Int. Conf. Machine Learning*, 2005, pp. "
        "640-647."
    ),
    (
        "C. E. Metz, \"Basic Principles of ROC Analysis,\" *Semin. Nucl. Med.*, vol. 8, "
        "no. 4, pp. 283-298, 1978."
    ),
    (
        "R. Kohavi, \"A Study of Cross-Validation and Bootstrap for Accuracy "
        "Estimation and Model Selection,\" in *Proc. 14th Int. Joint Conf. Artificial "
        "Intelligence*, 1995, pp. 1137-1143."
    ),
    (
        "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, \"BERT: Pre-training of "
        "Deep Bidirectional Transformers for Language Understanding,\" in *Proc. "
        "Conf. North American Chapter Assoc. Computational Linguistics*, 2019, pp. "
        "4171-4186."
    ),
    (
        "A. Dosovitskiy et al., \"An Image is Worth 16x16 Words: Transformers for "
        "Image Recognition at Scale,\" in *Proc. Int. Conf. Learning Representations* "
        "(ICLR), 2021."
    ),
]

