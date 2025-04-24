from sklearn.model_selection import train_test_split
import pandas as pd

def load_and_split_data(input_csv):
    reviews_df = pd.read_csv(input_csv)
    reviews_df["label"] = reviews_df["sentiment"].apply(
        lambda x: 1 if x == "positive" else 0
    )
    
    train_val_df = reviews_df[reviews_df["set"] == "train"]
    test_df = reviews_df[reviews_df["set"] == "test"].reset_index(drop=True)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=5000,
        random_state=42,
        stratify=train_val_df["label"],
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    return train_df, val_df, test_df
