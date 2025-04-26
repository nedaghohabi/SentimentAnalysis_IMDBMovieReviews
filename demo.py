import os
import tarfile
import urllib.request
import subprocess
import argparse


def run_scripts():
    parser = argparse.ArgumentParser(description="Master training script")
    parser.add_argument("--text", type=str, help="Text to classify", default="I really liked the movie")   
    parser.add_argument( "--cpu",action="store_true",help="Force CPU training",)
    args = parser.parse_args()
    
    print("Preparing dataset...")
    subprocess.run(["python", "prepare_dataset.py"])
    print()

    print("Training a logistic regression using TF-IDF document representations:")
    subprocess.run(["python", "train.py", "ml_bow", 
                    "--input_csv", "./data/imdb_cleaned.csv", 
                    "--output_dir", "./results/logreg_tfidf_l2",
                    "--vectorizer", "tfidf",
                    "--vectorizer_args", "max_features=10000",
                    "--model", "logreg",
                    "--model_args", "penalty=l2"])
    print()

    print("Performing inference on a given review")
    subprocess.run(["python", "inference.py", "ml_bow", 
                    "--model_dir", "./results/logreg_tfidf_l2", 
                    "--text", args.text])
    print()

    print("Training a Fully Connected Network using pretrained W2V word vector embeddings:")
    cmd = [
        "python", "train.py", "fcn_wv", 
        "--input_csv", "./data/imdb_cleaned.csv", 
        "--output_dir", "./results/fcn_pretrained_w2v",
        "--w2v_mode", "pretrained",
        "--pretrained_path", "data/GoogleNews-vectors-negative300.bin"
    ]
    
    if args.cpu:
        cmd.append("--cpu")
    subprocess.run(cmd)
    print()

    print("Performing inference on a given review")
    subprocess.run(["python", "inference.py", "fcn_wv", 
                    "--model_dir", "./results/fcn_pretrained_w2v", 
                    "--text", args.text])
    
if __name__ == "__main__":
    run_scripts()
