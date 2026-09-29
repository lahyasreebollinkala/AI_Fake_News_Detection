from pathlib import Path

# ==========================================
# PROJECT ROOT
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# FOLDERS
# ==========================================

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
IMAGE_DIR = ROOT_DIR / "images"
RESULT_DIR = ROOT_DIR / "results"

# ==========================================
# DATA FILES
# ==========================================

RAW_DATA = DATA_DIR / "GlobalFakeNews_Research2026_v1.csv"
CLEAN_DATA = DATA_DIR / "cleaned_news.csv"

# ==========================================
# MODEL & FEATURE ARTIFACTS
# ==========================================

BEST_MODEL = MODEL_DIR / "best_model.pkl"
VECTORIZER = MODEL_DIR / "vectorizer.pkl"
SCALER = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES = MODEL_DIR / "feature_names.pkl"

XTRAIN = MODEL_DIR / "X_train.pkl"
XTEST = MODEL_DIR / "X_test.pkl"
YTRAIN = MODEL_DIR / "y_train.pkl"
YTEST = MODEL_DIR / "y_test.pkl"

# ==========================================
# RESULTS & REPORTS
# ==========================================

METRICS_FILE = RESULT_DIR / "model_metrics.json"
PREDICTION_HISTORY = RESULT_DIR / "prediction_history.csv"