import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_DATA,
    MODEL_DIR,
    VECTORIZER,
    FEATURE_NAMES,
    XTRAIN,
    XTEST,
    YTRAIN,
    YTEST,
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("FEATURE ENGINEERING (TF-IDF ONLY)")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv(CLEAN_DATA)

# Fill missing values
df["clean_text"] = df["clean_text"].fillna("")

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

feature_names = vectorizer.get_feature_names_out().tolist()

print(f"TF-IDF Matrix Shape : {X.shape}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

# Save artifacts
joblib.dump(vectorizer, VECTORIZER)
joblib.dump(feature_names, FEATURE_NAMES)

joblib.dump(X_train, XTRAIN)
joblib.dump(X_test, XTEST)

joblib.dump(y_train, YTRAIN)
joblib.dump(y_test, YTEST)

print("\nSaved TF-IDF Vectorizer")
print("Saved Feature Names")
print("Saved Train/Test Splits")

print("=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)