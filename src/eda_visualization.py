import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/GlobalFakeNews_Research2026_v1.csv")

# Create images folder if it doesn't exist
import os
os.makedirs("images", exist_ok=True)

# ----------------------------------
# 1. Category Distribution
# ----------------------------------

plt.figure(figsize=(10,5))

df["category"].value_counts().plot(
    kind="bar",
    color="skyblue"
)

plt.title("News Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("images/category_distribution.png", dpi=300)

plt.close()

# ----------------------------------
# 2. Country Distribution
# ----------------------------------

plt.figure(figsize=(10,5))

df["country"].value_counts().plot(
    kind="bar",
    color="orange"
)

plt.title("Country Distribution")
plt.xlabel("Country")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("images/country_distribution.png", dpi=300)

plt.close()

# ----------------------------------
# 3. Source Distribution
# ----------------------------------

plt.figure(figsize=(10,5))

df["source_name"].value_counts().plot(
    kind="bar",
    color="green"
)

plt.title("News Source Distribution")
plt.xlabel("Source")
plt.ylabel("Articles")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("images/source_distribution.png", dpi=300)

plt.close()

print("All graphs saved successfully!")