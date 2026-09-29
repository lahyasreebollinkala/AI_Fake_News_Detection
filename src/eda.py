import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/GlobalFakeNews_Research2026_v1.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

print("\nDataset Shape:", df.shape)

print("\nStatistical Summary:")
print(df.describe())

print("\nLabel Distribution:")
print(df["label"].value_counts())

# -----------------------------
# Plot 1 : Real vs Fake
# -----------------------------
plt.figure(figsize=(6,5))

df["label"].value_counts().plot(
    kind="bar",
    color=["steelblue", "tomato"]
)

plt.title("Real vs Fake News")
plt.xlabel("Label")
plt.ylabel("Count")

plt.xticks([0,1],["Real (0)","Fake (1)"])

plt.tight_layout()

plt.savefig("images/label_distribution.png")

plt.show()