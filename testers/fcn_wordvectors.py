import sys
import os
import os.path as osp
import random

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "..")))

import argparse
import json
import numpy as np
import torch
from model_factory.fcn import FCNWithW2VEmbedding
import nltk
import string
import os.path as osp

nltk.download("punkt", quiet=True)


def preprocess(text):
    tokens = nltk.word_tokenize(text)
    return [t.translate(str.maketrans("", "", string.punctuation)) for t in tokens]


def load_word2idx(model_dir):
    with open(osp.join(model_dir, "word2idx.json")) as f:
        return json.load(f)


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = FCNWithW2VEmbedding(
        embedding_weights=checkpoint["embedding_matrix"],
        freeze_embeddings=True,  # No training at inference time
        hidden_size=checkpoint["hidden_size"],
        aggregator=checkpoint["aggregation"],
        output_size=checkpoint["output_size"],
        dropout_rate=0.0  # Disabled in eval mode
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_sentiment(text, model_dir):
    word2idx = load_word2idx(model_dir)
    model = load_model(osp.join(model_dir, "model.pth"))

    tokens = preprocess(text)
    indices = [word2idx[t] for t in tokens if t in word2idx]

    if not indices:
        return "Unknown (no known words in input)"

    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits["sentiment"], dim=1).numpy()
        pred = np.argmax(probs, axis=1)[0]

    return "Positive" if pred == 1 else "Negative"


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment using Word2Vec+FCN model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model folder")
    parser.add_argument("--text", type=str, required=True, help="Input text to classify")
    args = parser.parse_args()

    sentiment = predict_sentiment(args.text, args.model_dir)
    print(f"Provided review: {args.text}")
    print(f"Predicted sentiment: {sentiment}")


if __name__ == "__main__":
    main()
