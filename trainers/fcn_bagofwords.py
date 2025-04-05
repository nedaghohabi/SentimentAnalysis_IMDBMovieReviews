import sys
import os
import os.path as osp

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "..")))

import argparse
import time
import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

from model_factory.fcn import FCN
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

# Mapping of vectorizers
VECTORIZERS = {"count": CountVectorizer, "tfidf": TfidfVectorizer}



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

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(name="SentimentAnalysis_FCN_BOW", log_file=osp.join(args.output_dir, "logging.log"))

    # Load data
    logger.log_message(f"Training a fully connected neural network with with {args.vectorizer} embeddings for sentiment analysis.")
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
    # Tokenize sentences

    # Text vectorization
    vectorizer = VECTORIZERS[args.vectorizer](**vectorizer_params)
    params_str = ", ".join([f"{k}={v}" for k, v in vectorizer.get_params().items()])
    logger.log_message(f"Using {args.vectorizer} vectorizer with the following params:")
    logger.log_message(params_str)
    X_train = vectorizer.fit_transform(train_df["review_cleaned"])
    X_val = vectorizer.transform(val_df["review_cleaned"])
    X_test = vectorizer.transform(test_df["review_cleaned"])
    y_train, y_val, y_test = train_df["label"], val_df["label"], test_df["label"]
    logger.log_message("Text vectorization done. Data dimensions:")
    logger.log_message(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.log_message(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.log_message(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    logger.log_message("===========================================")

    train_dataset = SentimentAnalysisDataset(X_train.toarray(), y_train)
    val_dataset = SentimentAnalysisDataset(X_val.toarray(), y_val)
    test_dataset = SentimentAnalysisDataset(X_test.toarray(), y_test)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    logger.log_message("Preparing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCN(input_size=X_train.shape[1], hidden_size=128, output_size=2)
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
        val_metrics, val_loss = runner.run_epoch(
            "validate", epoch, val_dataloader
        )
        train_metrics = unravel_metric_dict(train_metrics)
        val_metrics = unravel_metric_dict(val_metrics)
        logger.log_training(epoch+1, train_loss, train_metrics, optimizer.param_groups[0]["lr"])
        logger.log_validation(epoch+1, val_loss, val_metrics)
        plotter.update(epoch, train_metrics, val_metrics, train_loss, val_loss)

    train_time = time.time() - start_time
    plotter.close()
    
    logger.log_message(f"Training done in {train_time:.2f} seconds.")
    logger.log_message("===========================================")

    logger.log_message("Evaluating model on Train, Validation, and Test sets...")
    # Evaluate model
    
    start_time = time.time()
    train_gts, train_logits, _ = runner.run_epoch(
        "test", epoch, train_dataloader
    )
    train_eval_time = time.time() - start_time

    train_probs = torch.softmax(torch.stack(train_logits["sentiment"]), dim=1)
    train_preds = torch.argmax(train_probs, dim=1).float()
    train_gts = torch.tensor(train_gts["sentiment"])

    start_time = time.time()
    val_gts, val_logits, _ = runner.run_epoch(
        "test", epoch, val_dataloader
    )
    val_eval_time = time.time() - start_time
    val_probs = torch.softmax(torch.stack(val_logits["sentiment"]), dim=1)
    val_preds = torch.argmax(val_probs, dim=1).float()
    val_gts = torch.tensor(val_gts["sentiment"])

    start_time = time.time()
    test_gts, test_logits, _ = runner.run_epoch(
        "test", epoch, test_dataloader
    )
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
    torch.save(
        model.state_dict(),
        osp.join(args.output_dir, "model.pth"),
    )
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
