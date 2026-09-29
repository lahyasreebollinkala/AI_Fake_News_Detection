"""Veritas -- news authenticity detection.

Training pipeline for the fake-news classification model. Each stage is a
runnable module and writes its artifacts to the directories declared in
:mod:`src.config`::

    python -m src.prepare_dataset      # data/GlobalFakeNews_Research2026_v1.csv
    python -m src.preprocessing        # data/cleaned_news.csv
    python -m src.feature_engineering  # models/*.pkl
    python -m src.train_models         # models/best_model.pkl, results/model_metrics.json
    python -m src.evaluate_models      # images/*.png
    python -m src.eda                  # console report, results/dataset_profile.json
    python -m src.eda_visualization    # images/*_distribution.png
"""

__version__ = "3.0.0"
