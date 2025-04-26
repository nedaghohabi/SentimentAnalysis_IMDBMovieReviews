import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp

import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "..")))

import os.path as osp
import argparse
import time
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from feeder.utils import load_and_split_data
from sklearn.metrics import classification_report

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def load_and_prepare_datasets(input_csv):
    train_df, val_df, test_df = load_and_split_data(input_csv)

    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["sentiment"])
    val_df["label"] = label_encoder.transform(val_df["sentiment"])
    test_df["label"] = label_encoder.transform(test_df["sentiment"])

    train_ds = Dataset.from_pandas(train_df[["review", "label"]])
    val_ds = Dataset.from_pandas(val_df[["review", "label"]])
    test_ds = Dataset.from_pandas(test_df[["review", "label"]])

    return train_ds, val_ds, test_ds


def tokenize_datasets(train_ds, val_ds, test_ds, tokenizer_adr):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_adr)
    def tokenize(example):
        return tokenizer(example["review"], truncation=True, padding="max_length", max_length=512)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    for ds in [train_ds, val_ds, test_ds]:
        cols = ['input_ids', 'attention_mask', 'label']
        if 'token_type_ids' in ds.column_names:
            cols.append('token_type_ids')
        ds.set_format("torch", columns=cols)


    return train_ds, val_ds, test_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


def main(args):
    train_ds, val_ds, test_ds = load_and_prepare_datasets(args.input_csv)
    train_ds, val_ds, test_ds = tokenize_datasets(train_ds, val_ds, test_ds, args.tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    if args.freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="no",           # DON'T save intermediate checkpoints
        save_strategy="epoch",
        logging_dir=args.logging_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # Time training
    start_train = time.time()
    trainer.train()
    train_time = time.time() - start_train

    # Time validation
    start_val = time.time()
    val_metrics = trainer.evaluate()
    val_time = time.time() - start_val

    # Time testing
    start_test = time.time()
    test_metrics = trainer.evaluate(test_ds)
    test_time = time.time() - start_test

    print("Validation results:", val_metrics)
    print("Test results:", test_metrics)

    # Predict on test set
    preds_output = trainer.predict(test_ds)
    preds = preds_output.predictions.argmax(axis=-1)
    labels = preds_output.label_ids

    # Print classification report
    print("Test Classification Report:")
    print(classification_report(labels, preds, digits=6))

    # Optional: evaluate on training data too
    train_metrics = trainer.evaluate(train_ds)

    # Prepare log data
    log_data = {
        "train_eval_time": train_time,
        "val_eval_time": val_time,
        "test_eval_time": test_time,
        "train_accuracy": train_metrics.get("eval_accuracy", None),
        "val_accuracy": val_metrics.get("eval_accuracy", None),
        "test_accuracy": test_metrics.get("eval_accuracy", None),
    }

    
    model.save_pretrained(os.path.join(args.output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
    os.makedirs(args.output_dir, exist_ok=True)    
    pd.DataFrame([log_data]).to_csv(
        osp.join(args.output_dir, "time_accuracy_logs.csv"), index=False
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM on IMDB")

    parser.add_argument("--input_csv", type=str, default="data/imdb_split", help="path to the input csv")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save checkpoints")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased-finetuned-sst-2-english", help="Model name or path")
    parser.add_argument("--tokenizer", type=str, default="distilbert-base-uncased", help="Model name or path")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--freeze_base", action="store_true", help="Freeze base transformer layers")

    args = parser.parse_args()
    main(args)
