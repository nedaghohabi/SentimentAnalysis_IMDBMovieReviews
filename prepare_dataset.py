import os
import tarfile
import urllib.request
import subprocess
import argparse

def download_and_extract_dataset(download_path, extract_path):
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    print("Atempting to donwload the IMDB reviews dataset")

    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    if not os.path.exists(download_path):
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, download_path)
        print("Download complete.")
    else:
        print("File already downloaded.")

    print("Atempting to decompress the download file")
    if not os.path.exists(extract_path):
        print(f"Extracting to {extract_path}...")
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(extract_path))
        print("Extraction complete.\n")
    else:
        print("Data already extracted.\n")


def download_word2vec_hf(download_path):
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    url = "https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/resolve/main/GoogleNews-vectors-negative300.bin"

    if not os.path.exists(download_path):
        print("Downloading Word2Vec embeddings from Hugging Face...")
        urllib.request.urlretrieve(url, download_path)
        print("Download complete.")
    else:
        print("Word2Vec file already exists.")


def run_scripts():
    subprocess.run(["python", "reviews_to_csv.py", 
                    "--imdb_dir", "./data/aclImdb", 
                    "--output_csv", "./data/imdb_reviews.csv"])
    print()

    print("Running data cleaning processing script...")
    subprocess.run(["python", "preprocessing.py", 
                    "--input_csv", "./data/imdb_reviews.csv", 
                    "--output_csv", "./data/imdb_cleaned.csv",
                   "--spell_check", "--lemmatization"])
    print()
    
if __name__ == "__main__":
    download_path = "./data/aclImdb_v1.tar.gz"
    extract_path = "./data/aclImdb"
    download_and_extract_dataset(download_path, extract_path)
    download_path = "./data/GoogleNews-vectors-negative300.bin"
    download_word2vec_hf(download_path)
    run_scripts()
