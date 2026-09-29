import pandas as pd
from src.utils import extract_article_metadata

print("Loading datasets...")

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 1
true["label"] = 0

dataset = pd.concat([fake, true], ignore_index=True)

print(f"Total Articles: {len(dataset)}")

dataset["full_text"] = (
    dataset["title"].fillna("") + ". " +
    dataset["text"].fillna("")
)

print("Generating metadata...")

metadata = dataset.apply(
    lambda row: extract_article_metadata(
        row["full_text"],
        row["title"]
    ),
    axis=1
)

metadata_df = pd.DataFrame(metadata.tolist())

dataset = pd.concat([dataset, metadata_df], axis=1)

final_dataset = dataset[
    [
        "full_text",
        "label",
        "sentiment_score",
        "clickbait_score",
        "emotional_word_ratio",
        "readability_score",
        "source_trust_score"
    ]
]

final_dataset.to_csv(
    "data/GlobalFakeNews_Research2026_v1.csv",
    index=False
)

print("Done!")
print(final_dataset.head())
print(final_dataset.shape)