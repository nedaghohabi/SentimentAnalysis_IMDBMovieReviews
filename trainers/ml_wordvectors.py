import sys
import os
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "..")))

import argparse
import pandas as pd
import logging
import joblib
import numpy as np
import nltk
import string

from utils.logger import Logger

from pathlib import Path
from gensim.models import Word2Vec, KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils.models_utils import train_ml_model, evaluate_ml_model
from feeder.utils import load_and_split_data


# Mapping of ML models
MODELS = {"logreg": LogisticRegression, "svm": LinearSVC, "rf": RandomForestClassifier}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ML model using Word2Vec embeddings."
    )

    # File paths
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to directory containing train, val, and test CSV files",
    )

    # Word2Vec options
    parser.add_argument(
        "--w2v_mode",
        type=str,
        choices=["pretrained", "train", "finetune"],
        required=True,
        help="Use pretrained Word2Vec, train from scratch, or finetune an existing model.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained Word2Vec model (if using pretrained or finetune).",
    )
    parser.add_argument(
        "--vector_size", type=int, default=100, help="Size of Word2Vec embeddings"
    )
    parser.add_argument(
        "--window", type=int, default=5, help="Window size for Word2Vec"
    )
    parser.add_argument(
        "--min_count", type=int, default=2, help="Minimum count for Word2Vec vocabulary"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of CPU cores for training Word2Vec",
    )
    parser.add_argument(
        "--w2v_epochs",
        type=int,
        default=5,
        help="Number of epochs for training Word2Vec (if w2v_mode=train)",
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
    """Converts a string like 'C=1.0,max_iter=100' into a dictionary {C:1.0, max_iter:100}."""
    args_dict = {}
    if arg_str:
        pairs = arg_str.split(",")
        for pair in pairs:
            key, value = pair.strip().split("=")
            if value.lower() in ["true", "false"]:
                args_dict[key] = value.lower() == "true"
            elif value.isdigit():
                args_dict[key] = int(value)
            else:
                try:
                    args_dict[key] = float(value)
                except ValueError:
                    args_dict[key] = value
    return args_dict

def train_word2vec(
    sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=5
):
    """Trains a Word2Vec model on tokenized text data."""
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )
    return model


def finetune_word2vec(existing_model, new_sentences, epochs=5):
    """Fine-tunes an existing Word2Vec model on new sentences."""
    existing_model.build_vocab(new_sentences, update=True)
    existing_model.train(
        new_sentences, total_examples=len(new_sentences), epochs=epochs
    )
    return existing_model


def document_vector(w2v_vectors, doc):
    """Computes the mean Word2Vec vector for a document."""
    words = [word for word in doc if word in w2v_vectors]
    if len(words) == 0:
        return np.zeros(w2v_vectors.vector_size)
    return np.mean(w2v_vectors[words], axis=0)

def prepare_w2v_data(docs):
    """Preprocesses text by removing punctuation and tokenizing."""
    # Remove punctuation
    all_sentences = []
    for doc in docs:
        sentences = nltk.sent_tokenize(doc)
        sentences = [sent.translate(str.maketrans("", "", string.punctuation)) for sent in sentences]
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        all_sentences.extend(sentences)
    return all_sentences

def prepare_input_data(docs):
    """Preprocesses text by removing punctuation and tokenizing."""
    # Remove punctuation
    all_sentences = []
    for doc in docs:
        words = nltk.word_tokenize(doc)
        words = [word.translate(str.maketrans("", "", string.punctuation)) for word in words]
        all_sentences.append(words)
    return all_sentences


def main():
    args = parse_args()
    model_params = parse_nested_args(args.model_args)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(name="SentimentAnalysis_ML_W2V", log_file=osp.join(args.output_dir, "logging.log"))


    logger.log_message(f"Training a {args.model} model with word2vec embeddings for sentiment analysis.")
    logger.log_message("===========================================")
    logger.log_message("Loading data...")
    
    train_df, val_df, test_df = load_and_split_data(args.input_csv)

    # Ensure expected columns exist
    for df_name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        if not ("review_cleaned" in df.columns and "label" in df.columns):
            raise ValueError(f"{df_name} CSV must contain 'review_cleaned' and 'label'")


    logger.log_message("Data loaded successfully.")
    logger.log_message("===========================================")
    logger.log_message("Preparing word2vec model...")
    # Tokenize sentences

    # Word2Vec Handling
    if args.w2v_mode == "pretrained":
        logger.log_message(f"Loading pretrained Word2Vec model from {args.pretrained_path}")
        w2v_vectors = KeyedVectors.load_word2vec_format(
            args.pretrained_path, binary=True
        )
    elif args.w2v_mode == "train":
        w2v_train_sentences = prepare_w2v_data(train_df["review_cleaned"].tolist())
        logger.log_message("Training Word2Vec model from scratch...")
        w2v_model = train_word2vec(
            w2v_train_sentences,
            args.vector_size,
            args.window,
            args.min_count,
            args.workers,
            args.w2v_epochs,
        )
        w2v_vectors = w2v_model.wv
    elif args.w2v_mode == "finetune":
        logger.log_message(
            f"Loading existing Word2Vec model from {args.pretrained_path} for fine-tuning."
        )

        # Try loading as a full Word2Vec model first
        try:
            w2v_model = Word2Vec.load(args.pretrained_path)
            logger.log_message("Successfully loaded Word2Vec model for fine-tuning.")
        except Exception as e:
            logger.error(f"Failed to load Word2Vec model: {e}")
            logger.log_message("Trying to load KeyedVectors instead...")

            # If that fails, try loading as KeyedVectors (not full Word2Vec)
            w2v_vectors = KeyedVectors.load_word2vec_format(
                args.pretrained_path, binary=True
            )
            logger.log_message(
                "Loaded KeyedVectors, but cannot fine-tune without full Word2Vec model."
            )

        # If we successfully loaded a full model, fine-tune it
        if isinstance(w2v_model, Word2Vec):
            w2v_train_sentences = prepare_w2v_data(train_df["review_cleaned"].tolist())
            w2v_model = finetune_word2vec(w2v_model, w2v_train_sentences, args.w2v_epochs)
            w2v_vectors = w2v_model.wv  # Convert to KeyedVectors for compatibility

    # Vectorize text data
    logger.log_message("Word2Vec model prepared successfully.")
    logger.log_message("===========================================")

    # Vectorize text data
    logger.log_message("Vectorizing text data...")

    train_sentences = prepare_input_data(train_df["review_cleaned"].tolist())
    val_sentences = prepare_input_data(val_df["review_cleaned"].tolist())
    test_sentences = prepare_input_data(test_df["review_cleaned"].tolist())

    X_train = np.array([document_vector(w2v_vectors, text) for text in train_sentences])
    X_val = np.array([document_vector(w2v_vectors, text) for text in val_sentences])
    X_test = np.array([document_vector(w2v_vectors, text) for text in test_sentences])
    y_train, y_val, y_test = train_df["label"], val_df["label"], test_df["label"]
    logger.log_message("Text vectorization done. Data dimensions:")
    logger.log_message(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.log_message(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.log_message(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    logger.log_message("===========================================")

    # Initialize and train model
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
    w2v_vectors.save_word2vec_format(
        osp.join(args.output_dir, "word2vec_vectors.bin"), binary=True
    )
    logger.log_message(
        f"Saved Word2Vec KeyedVectors as binary to {osp.join(args.output_dir, 'word2vec_vectors.bin')}"
    )

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
