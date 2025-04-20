import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
    get_scheduler,
)
from sklearn.metrics import accuracy_score, classification_report
from feeder.dataset import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune DistilBERT for sentiment classification.")
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to dir with train/val/test CSVs")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save model and logs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=256)
    return parser.parse_args()


def load_csv_data(csv_dir):
    train = pd.read_csv(os.path.join(csv_dir, "train.csv"))
    val = pd.read_csv(os.path.join(csv_dir, "validation.csv"))
    test = pd.read_csv(os.path.join(csv_dir, "test.csv"))
    return train, val, test


def tokenize_dataframe(df, tokenizer, max_len):
    encoded = tokenizer(df["review_cleaned"].tolist(), padding="max_length", truncation=True, max_length=max_len)
    encoded["labels"] = df["label"].tolist()
    return Dataset.from_dict(encoded)


def prepare_dataloaders(train_df, val_df, test_df, tokenizer, max_len, batch_size):
    train_ds = tokenize_dataframe(train_df, tokenizer, max_len)
    val_ds = tokenize_dataframe(val_df, tokenizer, max_len)
    test_ds = tokenize_dataframe(test_df, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, split_name=""):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.extend(labels)

    all_logits = np.concatenate(all_logits)
    preds = np.argmax(all_logits, axis=1)
    acc = accuracy_score(all_labels, preds)
    report = classification_report(all_labels, preds, output_dict=True)
    print(f"\n{split_name} Accuracy: {acc:.4f}")
    print(pd.DataFrame(report).transpose())
    return acc, report


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_csv_data(args.csv_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_df, val_df, test_df, tokenizer, args.max_len, args.batch_size
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * args.epochs,
    )

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Avg loss: {avg_loss:.4f}")

    print("\nEvaluating on Train, Validation, and Test...")
    train_acc, train_report = evaluate(model, train_loader, device, "Train")
    val_acc, val_report = evaluate(model, val_loader, device, "Validation")
    test_acc, test_report = evaluate(model, test_loader, device, "Test")

    print("Saving model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    results = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }
    pd.DataFrame([results]).to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
