import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (only needed once)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load dataset
df = pd.read_csv("data/GlobalFakeNews_Research2026_v1.csv")

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Clean and preprocess news article text.
    """

    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


print("=" * 60)
print("Cleaning text...")
print("=" * 60)

# Apply preprocessing
df["clean_text"] = df["full_text"].apply(clean_text)

# Save cleaned dataset
df.to_csv("data/cleaned_news.csv", index=False)

print("\nPreprocessing Completed Successfully!")

print("\nOriginal Text:\n")
print(df["full_text"].iloc[0][:300])

print("\n")

print("Cleaned Text:\n")
print(df["clean_text"].iloc[0][:300])

print("\n")

print("Saved cleaned dataset as:")
print("data/cleaned_news.csv")