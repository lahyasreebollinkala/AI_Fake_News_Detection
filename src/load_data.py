import pandas as pd


def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)

        print("=" * 60)
        print("DATASET LOADED SUCCESSFULLY")
        print("=" * 60)

        print(f"\nRows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        print("\nColumn Names:")
        print(df.columns.tolist())

        print("\nFirst 5 Records:")
        print(df.head())

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nLabel Distribution:")
        print(df["label"].value_counts())

        return df

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    dataset_path = "data/GlobalFakeNews_Research2026_v1.csv"

    df = load_dataset(dataset_path)