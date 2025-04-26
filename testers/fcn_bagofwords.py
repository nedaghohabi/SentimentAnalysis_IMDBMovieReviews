import sys
import os
import os.path as osp
import random

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "..")))

import argparse
import torch
import joblib
import numpy as np
from model_factory.fcn import FCN
import os.path as osp

def load_model_from_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = FCN(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        output_size=checkpoint["output_size"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict_sentiment(text, model_dir):
    # Load vectorizer and transform input text
    vectorizer = joblib.load(osp.join(model_dir, "vectorizer.pkl"))
    X = vectorizer.transform([text]).toarray()

    # Load model from checkpoint
    model = load_model_from_checkpoint(osp.join(model_dir, "model.pth"))
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits["sentiment"], dim=1).numpy()
        pred = np.argmax(probs, axis=1)[0]
    
    return "Positive" if pred == 1 else "Negative"

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment using an FCN model with BOW/TF-IDF features.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model folder")
    parser.add_argument("--text", type=str, required=True, help="Text review to classify")
    args = parser.parse_args()

    sentiment = predict_sentiment(args.text, args.model_dir)
    print(f"Provided review: {args.text}")
    print(f"Predicted sentiment: {sentiment}")

if __name__ == "__main__":
    main()
