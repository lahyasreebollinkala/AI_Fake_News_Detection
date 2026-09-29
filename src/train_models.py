import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from src.config import (
    XTRAIN,
    XTEST,
    YTRAIN,
    YTEST,
    BEST_MODEL,
    METRICS_FILE,
    RESULT_DIR,
    IMAGE_DIR,
    FEATURE_NAMES,
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# -----------------------------
# Load Training Data
# -----------------------------
X_train = joblib.load(XTRAIN)
X_test = joblib.load(XTEST)

y_train = joblib.load(YTRAIN)
y_test = joblib.load(YTEST)

feature_names = joblib.load(FEATURE_NAMES)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    ),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42,
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
    )
}

results = {}

best_model = None
best_name = None
best_f1 = -1

print("=" * 70)
print("MODEL TRAINING")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    else:
        prob = None

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    if prob is not None:
        auc = roc_auc_score(y_test, prob)
    else:
        auc = None

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results[name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc) if auc else None,
    }

    if f1 > best_f1:
        best_f1 = f1
        best_name = name
        best_model = model

# -----------------------------
# Save Best Model
# -----------------------------
joblib.dump(best_model, BEST_MODEL)

metrics = {
    "best_model_name": best_name,
    "best_f1_score": float(best_f1),
    "models": results,
}

with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=4)

print("\nBest Model:", best_name)

# -----------------------------
# Confusion Matrix
# -----------------------------
pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.tight_layout()

plt.savefig(os.path.join(IMAGE_DIR, "confusion_matrix.png"))

plt.close()

# -----------------------------
# ROC Curve
# -----------------------------
if hasattr(best_model, "predict_proba"):

    prob = best_model.predict_proba(X_test)[:, 1]

    RocCurveDisplay.from_predictions(
        y_test,
        prob,
    )

    plt.tight_layout()

    plt.savefig(os.path.join(IMAGE_DIR, "roc_curve.png"))

    plt.close()

# -----------------------------
# Feature Importance
# -----------------------------
if hasattr(best_model, "coef_"):

    coef = np.abs(best_model.coef_[0])

    idx = np.argsort(coef)[-20:]

    plt.figure(figsize=(8,6))

    plt.barh(
        np.array(feature_names)[idx],
        coef[idx],
    )

    plt.title("Top 20 Important Features")

    plt.tight_layout()

    plt.savefig(os.path.join(IMAGE_DIR, "feature_importance.png"))

    plt.close()

print("\nTraining Completed Successfully.")