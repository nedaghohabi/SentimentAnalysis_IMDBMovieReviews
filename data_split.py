import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Split data into train, validation, and test sets."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save output CSV files"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.7,
        help="Proportion of data in the train set",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Proportion of data in the validation set",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data in the test set",
    )
    parser.add_argument(
        "--stratify", action="store_true", help="Stratify the split based on labels"
    )
    parser.add_argument(
        "--target_column", type=str, default="sentiment", help="Column name for labels"
    )

    args = parser.parse_args()

    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    target_column = args.target_column
    stratify = args.stratify
    seed = args.seed

    # Ensure train + val + test = 1
    assert (
        round(train_size + val_size + test_size, 5) == 1
    ), "Train, validation, and test sizes must sum to 1."

    # Read CSV
    reviews_df = pd.read_csv(args.input_csv)

    # Ensure the target column is binary (convert if needed)
    if "label" not in reviews_df.columns:  # Avoid overwriting
        if reviews_df[target_column].dtype == "object":
            reviews_df["label"] = reviews_df[target_column].apply(
                lambda x: 1 if x == "positive" else 0
            )
        else:
            reviews_df["label"] = reviews_df[target_column]

    # First, split into train+val and test
    train_val_df, test_df = train_test_split(
        reviews_df,
        test_size=test_size,
        random_state=seed,
        stratify=reviews_df["label"] if stratify else None,
    )

    # Compute train and validation ratios within the remaining dataset
    train_ratio = train_size / (train_size + val_size)

    # Split train and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=1 - train_ratio,  # This is clearer
        random_state=seed,
        stratify=train_val_df["label"] if stratify else None,
    )

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Print data split summary
    print(f"Total samples: {len(reviews_df)}")
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    print(
        f"Train ratio: {len(train_df) / len(reviews_df):.2f}, Val ratio: {len(val_df) / len(reviews_df):.2f}, Test ratio: {len(test_df) / len(reviews_df):.2f}"
    )


if __name__ == "__main__":
    main()
