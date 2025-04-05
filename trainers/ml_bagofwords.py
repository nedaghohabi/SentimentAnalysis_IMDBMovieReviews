import sys
import os
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "..")))

import argparse
import time
import pandas as pd
import logging
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils.models_utils import train_ml_model, evaluate_ml_model
from utils.logger import Logger

# Mapping of vectorizers
VECTORIZERS = {"count": CountVectorizer, "tfidf": TfidfVectorizer}

# Mapping of ML models
MODELS = {
    "logreg": LogisticRegression,
    "svm": LinearSVC,
    "rf": RandomForestClassifier,
    "knn": KNeighborsClassifier,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text classification model.")

    # File paths
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Path to directory containing train, val, and test CSV files",
    )

    # Text Vectorization
    parser.add_argument(
        "--vectorizer",
        type=str,
        choices=VECTORIZERS.keys(),
        default="tfidf",
        help="Vectorization method: count or tfidf",
    )
    parser.add_argument(
        "--vectorizer_args",
        type=str,
        default="",
        help="Vectorizer hyperparameters as key=value pairs (comma-separated)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS.keys(),
        required=True,
        help="Model type: logreg, svm, rf",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default="",
        help="Model hyperparameters as key=value pairs (comma-separated)",
    )

    # Output directory
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save logs and model"
    )

    return parser.parse_args()


def parse_nested_args(arg_str):
    """
    Converts a string like 'C=1.0,max_iter=100,shuffle=True,solver=lbfgs'
    into a dictionary {C: 1.0, max_iter: 100, shuffle: True, solver: 'lbfgs'}.
    """
    args_dict = {}
    if arg_str:
        pairs = arg_str.split(",")
        for pair in pairs:
            pair = pair.strip()
            key, value = pair.split("=")

            # Convert types correctly
            if value.lower() in ["true", "false", "none"]:  # Boolean check
                if value.lower() == "true":
                    args_dict[key] = True
                elif value.lower() == "false":
                    args_dict[key] = False
                else:
                    args_dict[key] = None
            elif value.isdigit():  # Integer check
                args_dict[key] = int(value)
            else:
                try:
                    args_dict[key] = float(value)  # Float check
                except ValueError:
                    args_dict[key] = value  # Default to string if conversion fails
    return args_dict


def main():
    args = parse_args()
    vectorizer_params = parse_nested_args(args.vectorizer_args)
    model_params = parse_nested_args(args.model_args)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger = Logger(name="SentimentAnalysis_ML_W2V", log_file=osp.join(args.output_dir, "logging.log"))


    logger.log_message(f"Training a {args.model} model with {args.vectorizer} embeddings for sentiment analysis.")
    logger.log_message("===========================================")
    logger.log_message("Loading data...")

    train_df = pd.read_csv(os.path.join(args.csv_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.csv_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(args.csv_dir, "test.csv"))

    # Ensure expected columns exist
    for df_name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        if not ("review_cleaned" in df.columns and "label" in df.columns):
            raise ValueError(f"{df_name} CSV must contain 'review_cleaned' and 'label'")


    logger.log_message("Data loaded successfully.")
    logger.log_message("===========================================")
    logger.log_message("Vectorizing text data...")
    # Text vectorization
    vectorizer = VECTORIZERS[args.vectorizer](**vectorizer_params)
    params_str = ", ".join([f"{k}={v}" for k, v in vectorizer.get_params().items()])
    logger.log_message(f"Using {vectorizer.__class__.__name__} vectorizer with params: {params_str}")
    X_train = vectorizer.fit_transform(train_df["review_cleaned"])
    X_val = vectorizer.transform(val_df["review_cleaned"])
    X_test = vectorizer.transform(test_df["review_cleaned"])
    y_train, y_val, y_test = train_df["label"], val_df["label"], test_df["label"]
    logger.log_message("Text vectorization done. Data dimensions:")
    logger.log_message(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.log_message(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.log_message(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    logger.log_message("===========================================")

    # Initialize model
    logger.log_message("Preparing model...")
    model = MODELS[args.model](**model_params)
    params_str = ", ".join([f"{k}={v}" for k, v in model.get_params().items()])
    logger.log_message(f"Using {model.__class__.__name__} model with params: {params_str}")
    logger.log_message("Model prepared successfully.")
    logger.log_message("===========================================")
    logger.log_message("Training model...")
    model, train_time = train_ml_model(model, X_train, y_train)
    logger.log_message(f"Training done in {train_time:.2f} seconds.")
    logger.log_message("===========================================")

    # Evaluate on train, validation, and test sets
    train_preds, train_eval_time = evaluate_ml_model(model, X_train, y_train, "Train")
    train_report = classification_report(y_train, train_preds, output_dict=True)
    train_acc = accuracy_score(y_train, train_preds)
    logger.log_message(
        f"Evaluation on Train set ({len(train_df)} samples) done in {train_eval_time:.2f} seconds."
    )
    logger.log_message(
        f"\nTrain Classification Report:\n{pd.DataFrame(train_report).transpose()}"
    )

    val_preds, val_eval_time = evaluate_ml_model(model, X_val, y_val, "Validation")
    val_report = classification_report(y_val, val_preds, output_dict=True)
    val_acc = accuracy_score(y_val, val_preds)
    logger.log_message(
        f"Evaluation on Validation set ({len(val_df)} samples) done in {val_eval_time:.2f} seconds."
    )
    logger.log_message(
        f"\nValidation Classification Report:\n{pd.DataFrame(val_report).transpose()}"
    )

    test_preds, test_eval_time = evaluate_ml_model(model, X_test, y_test, "Test")
    test_report = classification_report(y_test, test_preds, output_dict=True)
    test_acc = accuracy_score(y_test, test_preds)
    logger.log_message(
        f"Evaluation on Test set ({len(test_df)} samples) done in {test_eval_time:.2f} seconds."
    )
    logger.log_message(
        f"\nTest Classification Report:\n{pd.DataFrame(test_report).transpose()}"
    )

    # Save model and vectorizer
    joblib.dump(model, osp.join(args.output_dir, "model.pkl"))
    joblib.dump(vectorizer, osp.join(args.output_dir, "vectorizer.pkl"))
    logger.log_message(f"Model and vectorizer saved to {args.output_dir}")

    # Save logs with accuracies and evaluation times
    log_data = {
        "train_time": train_time,
        "train_eval_time": train_eval_time,
        "val_eval_time": val_eval_time,
        "test_eval_time": test_eval_time,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }
    pd.DataFrame([log_data]).to_csv(
        osp.join(args.output_dir, "time_accuracy_logs.csv"), index=False
    )

    logger.log_message("Training complete. Model and logs saved!")
    logger.log_message("===========================================")
    logger.close()


if __name__ == "__main__":
    main()
