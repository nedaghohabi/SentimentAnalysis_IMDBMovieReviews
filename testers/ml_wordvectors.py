import argparse
import joblib
import numpy as np
import nltk
import string
from gensim.models import KeyedVectors
import os.path as osp

def preprocess_text(text):
    """Tokenize and remove punctuation."""
    words = nltk.word_tokenize(text)
    words = [word.translate(str.maketrans("", "", string.punctuation)) for word in words]
    return words

def document_vector(w2v_vectors, doc):
    """Compute mean Word2Vec vector for a document."""
    words = [word for word in doc if word in w2v_vectors]
    if len(words) == 0:
        return np.zeros(w2v_vectors.vector_size)
    return np.mean(w2v_vectors[words], axis=0)

def predict_sentiment(text, model_dir):
    model_path = osp.join(model_dir, "model.pkl")
    w2v_path = osp.join(model_dir, "word2vec_vectors.bin")

    model = joblib.load(model_path)
    w2v_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    tokens = preprocess_text(text)
    vector = document_vector(w2v_vectors, tokens).reshape(1, -1)

    pred = model.predict(vector)[0]
    label = "Positive" if pred == 1 else "Negative"
    return label

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment using Word2Vec-based model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model folder")
    parser.add_argument("--text", type=str, required=True, help="Text input to classify")
    args = parser.parse_args()

    sentiment = predict_sentiment(args.text, args.model_dir)
    print(f"Provided review: {args.text}")
    print(f"Predicted sentiment: {sentiment}")

if __name__ == "__main__":
    main()
