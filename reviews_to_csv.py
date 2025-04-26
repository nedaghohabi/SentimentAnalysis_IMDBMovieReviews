import argparse
import os
import pandas as pd
from pathlib import Path
import tqdm


def load_imdb_data(imdb_root):
    data = []
    sets = ['train', 'test']
    labels = {'pos': 'positive', 'neg': 'negative'}

    for dataset in sets:
        for label_name, sentiment in labels.items():
            dir_path = Path(imdb_root) / dataset / label_name
            print(f"Collecting {sentiment} samples of the {dataset} folder")
            txt_pths = list(dir_path.glob("*.txt"))
            for file in tqdm.tqdm(txt_pths):
                with open(file, encoding="utf-8") as f:
                    review = f.read().strip()
                data.append({
                    "review": review,
                    "sentiment": sentiment,
                    "set": dataset
                })
    return pd.DataFrame(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert IMDB folder to CSV")
    parser.add_argument("--imdb_dir", type=str, required=True, help="Path to IMDB aclImdb directory")
    parser.add_argument("--output_csv", type=str, default="imdb_reviews.csv", help="Path to output CSV file")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output_csv):
        overwrite = input(f"Output file '{args.output_csv}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite not in ["y", "yes"]:
            print(f"File not overwritten.")
            return
            
    df = load_imdb_data(args.imdb_dir)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
