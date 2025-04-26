import argparse
import joblib
import os.path as osp

def load_model_and_vectorizer(model_dir):
    model_path = osp.join(model_dir, "model.pkl")
    vectorizer_path = osp.join(model_dir, "vectorizer.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = "Positive" if pred == 1 else "Negative"
    return label

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of a text review.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved model and vectorizer")
    parser.add_argument("--text", type=str, required=True, help="Text review to classify")
    args = parser.parse_args()

    model, vectorizer = load_model_and_vectorizer(args.model_dir)
    sentiment = predict_sentiment(args.text, model, vectorizer)

    print(f"Provided Review: {args.text}")
    print(f"Predicted sentiment: {sentiment}")

if __name__ == "__main__":
    main()
