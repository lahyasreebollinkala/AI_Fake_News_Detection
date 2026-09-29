import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    RocCurveDisplay,
    accuracy_score,
    classification_report
)

from src.config import BEST_MODEL, XTEST, YTEST, FEATURE_NAMES, IMAGE_DIR

os.makedirs(IMAGE_DIR, exist_ok=True)

# Load model, test data, and feature names
model = joblib.load(BEST_MODEL)
X_test = joblib.load(XTEST)
y_test = joblib.load(YTEST)
feature_names = joblib.load(FEATURE_NAMES) if os.path.exists(FEATURE_NAMES) else None

# Predictions
y_pred = model.predict(X_test)

# -----------------------------
# 1. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real (0)", "Fake (1)"])
disp.plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix - Best Model", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# -----------------------------
# 2. ROC Curve
# -----------------------------
if hasattr(model, "predict_proba"):
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="Best Model")
    plt.plot([0, 1], [0, 1], "k--", label="Chance Level (AUC = 0.50)")
    plt.title("ROC Curve - Best Model", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "roc_curve.png", dpi=300)
    plt.close()

# -----------------------------
# 3. Top Feature Importances / Coefficients
# -----------------------------
if feature_names is not None:
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])

    if importance is not None and len(importance) == len(feature_names):
        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        top20 = df_imp.sort_values(by="Importance", ascending=False).head(20)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top20, x="Importance", y="Feature", palette="viridis")
        plt.title("Top 20 Predictive Features", fontsize=13, fontweight="bold")
        plt.xlabel("Importance / Coefficient Magnitude")
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=300)
        plt.close()

print("Evaluation graphs saved successfully to 'images/' folder!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Real", "Fake"]))