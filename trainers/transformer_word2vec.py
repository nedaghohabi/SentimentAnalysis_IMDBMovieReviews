import sys
import os
import os.path as osp

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "..")))

import argparse
import pandas as pd
import logging
import numpy as np
import time
import nltk
import string

from utils.logger import Logger

from pathlib import Path
from gensim.models import Word2Vec, KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


from model_factory.transformer import TransformerWithW2VEmbedding
from feeder.dataset import SentimentAnalysisDataset
from utils.scheduler import WarmupCosineAnnealingScheduler
from utils.helpers import unravel_metric_dict
from utils.logger import Logger
from runners.epoch_runner import EpochRunner
from metrics import accuracy, precision, recall, MetricsEvaluator
from visualization.training_visualizer import TrainingPlotter

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ML model using Word2Vec embeddings."
    )

    # File paths
    parser.add_argument(
        "--csv_dir",
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
        "--epochs", type=int, default=10, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training the model"
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.001,
        help="Initial learning rate for training",
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Hidden size for the fully connected neural network",
    )

    parser.add_argument(
        "--num_heads",  
        type=int,
        default=4,
        help="Number of attention heads for the transformer model",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of layers in the transformer model",
    )

    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=512,
        help="Dimension of the feedforward layer in the transformer model",
    )

    parser.add_argument(
        "--freeze_embeddings",
        type=bool,
        default=True,
        help="Whether to freeze the Word2Vec embeddings during training",
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="Dropout rate used in the fully connected layers.",
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


def tokenoze_vector(word2idx, doc):
    """Computes the mean Word2Vec vector for a document."""
    words = [word2idx[word] for word in doc if word in word2idx]
    if len(words) == 0:
        return []
    return words


def prepare_w2v_data(docs):
    """Preprocesses text by removing punctuation and tokenizing."""
    # Remove punctuation
    all_sentences = []
    for doc in docs:
        sentences = nltk.sent_tokenize(doc)
        sentences = [
            sent.translate(str.maketrans("", "", string.punctuation))
            for sent in sentences
        ]
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        all_sentences.extend(sentences)
    return all_sentences


def prepare_input_data(docs):
    """Preprocesses text by removing punctuation and tokenizing."""
    # Remove punctuation
    all_sentences = []
    for doc in docs:
        words = nltk.word_tokenize(doc)
        words = [
            word.translate(str.maketrans("", "", string.punctuation)) for word in words
        ]
        all_sentences.append(words)
    return all_sentences


def collate_fn(batch):
    """
    Collates a batch of data and applies padding.

    Args:
        batch (list): List of samples in the batch
        Due to the behavior of Pod5IterDataset,
        the batch size is determined in the Dataset instead of DataLoader.
        Thus, the batch size is always 1.
        We will make sure that the batch size is 1 in the DataLoader,
        and we squeeze the batch dimension here.

    Returns:
        dict: Dictionary containing the batched inputs and labels
    """
    vectors = [item["inputs"]["vector"] for item in batch]
    categories = torch.tensor([item["labels"]["sentiment"] for item in batch])

    padded_vectors = pad_sequence(vectors, batch_first=True, padding_value=0)

    return {
        "inputs": {
            "vector": padded_vectors,
        },
        "labels": {
            "sentiment": categories,
        },
    }


def main():
    args = parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(
        name="SentimentAnalysis_ML_W2V",
        log_file=osp.join(args.output_dir, "logging.log"),
    )

    # Load data
    logger.log_message(
        f"Training a fully connected neural network with Word2Vec embeddings for sentiment analysis."
    )
    logger.log_message("===========================================")
    logger.log_message("Loading data...")
    train_df = pd.read_csv(osp.join(args.csv_dir, "train.csv"))
    val_df = pd.read_csv(osp.join(args.csv_dir, "validation.csv"))
    test_df = pd.read_csv(osp.join(args.csv_dir, "test.csv"))

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
        logger.log_message(
            f"Loading pretrained Word2Vec model from {args.pretrained_path}"
        )
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
            args.epochs,
        )
        w2v_vectors = w2v_model.wv
    elif args.w2v_mode == "finetune":
        logger.log_message(
            f"Loading existing Word2Vec model from {args.pretrained_path} for fine-tuning."
        )
        try:
            w2v_model = Word2Vec.load(args.pretrained_path)
            logger.log_message("Successfully loaded Word2Vec model for fine-tuning.")
        except Exception as e:
            logger.log_exception(f"Failed to load Word2Vec model: {e}")
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
            w2v_model = finetune_word2vec(w2v_model, w2v_train_sentences, args.epochs)
            w2v_vectors = w2v_model.wv  # Convert to KeyedVectors for compatibility

    logger.log_message("Word2Vec model prepared successfully.")
    logger.log_message("===========================================")

    logger.log_message("Creating an embedding matrix from the Word2Vec word vectors...")
    train_sentences = prepare_input_data(train_df["review_cleaned"].tolist())
    vocab = {
        word for sentence in train_sentences for word in sentence if word in w2v_vectors
    }
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in word2idx.items()}
    embedding_dim = w2v_vectors.vector_size
    embedding_matrix = np.zeros((len(word2idx) + 1, embedding_dim), dtype=np.float32)
    for word, idx in word2idx.items():
        embedding_matrix[idx] = w2v_vectors[word]
    logger.log_message("Embedding matrix created successfully.")
    logger.log_message(f"Embedding matrix shape: {embedding_matrix.shape}")
    logger.log_message("===========================================")

    # Vectorize text data
    logger.log_message("Tokenizing and vectorizing text data...")

    train_sentences = prepare_input_data(train_df["review_cleaned"].tolist())
    val_sentences = prepare_input_data(val_df["review_cleaned"].tolist())
    test_sentences = prepare_input_data(test_df["review_cleaned"].tolist())

    X_train = [tokenoze_vector(word2idx, text) for text in train_sentences]
    X_val = [tokenoze_vector(word2idx, text) for text in val_sentences]
    X_test = [tokenoze_vector(word2idx, text) for text in test_sentences]
    y_train, y_val, y_test = train_df["label"], val_df["label"], test_df["label"]
    logger.log_message("Text tokenization done. Data lengths:")
    logger.log_message(f"X_train: {len(X_train)}, y_train: {len(y_train)}")
    logger.log_message(f"X_val: {len(X_val)}, y_val: {len(y_val)}")
    logger.log_message(f"X_test: {len(X_test)}, y_test: {len(y_test)}")
    logger.log_message("===========================================")

    # Initialize model
    train_dataset = SentimentAnalysisDataset(X_train, y_train)
    val_dataset = SentimentAnalysisDataset(X_val, y_val)
    test_dataset = SentimentAnalysisDataset(X_test, y_test)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    logger.log_message("Preparing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerWithW2VEmbedding(
        embedding_weights=embedding_matrix,
        freeze_embeddings=args.freeze_embeddings,
        hidden_size=args.hidden_size,  # not directly used, but still loggable
        output_size=2,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout_rate=args.dropout_rate,
        max_seq_len=1000,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
    criterion = nn.CrossEntropyLoss()
    logger.log_message("Model prepared successfully.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_message(f"Model has {num_params} trainable parameters.")
    logger.log_message("===========================================")

    def criterion_func(outputs, targets):
        """Computes combined loss for sequence classification and sequence-to-sequence tasks."""
        return criterion(outputs["sentiment"], targets["sentiment"])

    metrics_evaluator = MetricsEvaluator(
        metrics_dict={
            "sentiment": [
                (accuracy, {"from_logits": True, "binary": False}),
                (precision, {"from_logits": True, "binary": False}),
                (recall, {"from_logits": True, "binary": False}),
            ]
        }
    )

    runner = EpochRunner(
        model=model,
        optimizer=optimizer,
        loss_func=criterion_func,
        scheduler_handler=None,
        metrics_evaluator=metrics_evaluator,
        device=device,
        use_amp=False,
    )
    plotter = TrainingPlotter(["sentiment_accuracy"], args.output_dir)

    logger.log_message("Training model...")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_metrics, train_loss = runner.run_epoch("train", epoch, train_dataloader)
        val_metrics, val_loss = runner.run_epoch("validate", epoch, val_dataloader)
        train_metrics = unravel_metric_dict(train_metrics)
        val_metrics = unravel_metric_dict(val_metrics)
        logger.log_training(
            epoch + 1, train_loss, train_metrics, optimizer.param_groups[0]["lr"]
        )
        logger.log_validation(epoch + 1, val_loss, val_metrics)
        plotter.update(epoch, train_metrics, val_metrics, train_loss, val_loss)

    train_time = time.time() - start_time
    plotter.close()

    logger.log_message(f"Training done in {train_time:.2f} seconds.")
    logger.log_message("===========================================")

    logger.log_message("Evaluating model on Train, Validation, and Test sets...")
    # Evaluate model
    start_time = time.time()
    train_gts, train_logits, _ = runner.run_epoch("test", epoch, train_dataloader)
    train_eval_time = time.time() - start_time

    train_probs = torch.softmax(torch.stack(train_logits["sentiment"]), dim=1)
    train_preds = torch.argmax(train_probs, dim=1).float()
    train_gts = torch.tensor(train_gts["sentiment"])

    start_time = time.time()
    val_gts, val_logits, _ = runner.run_epoch("test", epoch, val_dataloader)
    val_eval_time = time.time() - start_time
    val_probs = torch.softmax(torch.stack(val_logits["sentiment"]), dim=1)
    val_preds = torch.argmax(val_probs, dim=1).float()
    val_gts = torch.tensor(val_gts["sentiment"])

    start_time = time.time()
    test_gts, test_logits, _ = runner.run_epoch("test", epoch, test_dataloader)
    test_eval_time = time.time() - start_time
    test_probs = torch.softmax(torch.stack(test_logits["sentiment"]), dim=1)
    test_preds = torch.argmax(test_probs, dim=1).float()
    test_gts = torch.tensor(test_gts["sentiment"])

    train_report = classification_report(train_gts, train_preds, output_dict=True)
    train_acc = accuracy_score(y_train, train_preds)
    logger.log_message(
        f"Evaluation on Train set ({len(train_df)} samples) done in {train_eval_time:.2f} seconds."
    )
    logger.log_message(
        f"\nTrain Classification Report:\n{pd.DataFrame(train_report).transpose()}"
    )

    val_report = classification_report(val_gts, val_preds, output_dict=True)
    val_acc = accuracy_score(y_val, val_preds)
    logger.log_message(
        f"Evaluation on Validation set ({len(val_df)} samples) done in {val_eval_time:.2f} seconds."
    )
    logger.log_message(
        f"\nValidation Classification Report:\n{pd.DataFrame(val_report).transpose()}"
    )

    test_report = classification_report(test_gts, test_preds, output_dict=True)
    test_acc = accuracy_score(y_test, test_preds)
    logger.log_message(
        f"Evaluation on Test set ({len(test_df)} samples) done in {test_eval_time:.2f} seconds."
    )
    logger.log_message(
        f"\nTest Classification Report:\n{pd.DataFrame(test_report).transpose()}"
    )

    # Save model and vectorizer

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
