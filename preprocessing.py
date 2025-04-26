import argparse
import re
import string
import pandas as pd
import nltk
import os
import urllib.request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm


def download_symspell_dictionary(dict_path):
    """Downloads the SymSpell frequency dictionary if not present."""
    if not os.path.exists(dict_path):
        print(f"Downloading SymSpell dictionary to {dict_path}...")
        url = "https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"
        urllib.request.urlretrieve(url, dict_path)
        print("Download complete.")
    else:
        print("SymSpell dictionary already exists.")


def initialize_symspell(dictionary_path):
    """Initializes SymSpell with the given dictionary."""
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell


def basic_preprocessing(text):
    """Preprocesses text: lowercasing, removing HTML, URLs, and tokenizing."""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    words = nltk.word_tokenize(text)
    return words


def spell_correction(words, sym_spell):
    """Performs spelling correction using SymSpell."""
    corrected_words = []
    for word in words:
        if word in string.punctuation:
            corrected_words.append(word)
            continue
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        suggestion = suggestions[0].term if suggestions else word
        corrected_words.append(suggestion.lower())
    return corrected_words


def remove_stopwords(words, stop_words):
    return [word for word in words if word not in stop_words]


def remove_punctuation(words):
    return [word for word in words if word not in string.punctuation]


def stemming(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]


def lemmatization(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def full_pipeline(text, sym_spell, stop_words, rem_punctuation=False, spell_check=False, use_stemming=False, use_lemmatization=False):
    words = basic_preprocessing(text)
    if rem_punctuation:
        words = remove_punctuation(words)
    words = remove_stopwords(words, stop_words)
    if use_stemming:
        words = stemming(words)
    if use_lemmatization:
        words = lemmatization(words)
    if spell_check:
        words = spell_correction(words, sym_spell)
    return " ".join(words)


def process_dataframe(df, column, sym_spell, stop_words, rem_punctuation=False, spell_check=False, use_stemming=False, use_lemmatization=False):
    tqdm.pandas(desc="Processing reviews")
    df[column + "_cleaned"] = df[column].progress_apply(
        lambda text: full_pipeline(text, sym_spell, stop_words, rem_punctuation, spell_check, use_stemming, use_lemmatization)
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Text preprocessing for sentiment analysis.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save processed CSV file")
    parser.add_argument("--rem_punctuation", action="store_true", help="Enable punctuation removal")
    parser.add_argument("--column", type=str, default="review", help="Column name to preprocess")
    parser.add_argument("--spell_check", action="store_true", help="Enable spell checking")
    parser.add_argument("--stemming", action="store_true", help="Enable stemming")
    parser.add_argument("--lemmatization", action="store_true", help="Enable lemmatization")
    
    args = parser.parse_args()

    if os.path.exists(args.output_csv):
        overwrite = input(f"Output file '{args.output_csv}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite not in ["y", "yes"]:
            print("File not overwritten.")
            return

    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    dict_path = "frequency_dictionary_en_82_765.txt"
    download_symspell_dictionary(dict_path)

    sym_spell = initialize_symspell(dict_path)
    stop_words = set(stopwords.words("english"))

    df = pd.read_csv(args.input_csv)
    df = process_dataframe(
        df,
        args.column,
        sym_spell,
        stop_words,
        args.rem_punctuation,
        args.spell_check,
        args.stemming,
        args.lemmatization,
    )
    df.to_csv(args.output_csv, index=False)

    print(f"Processed DataFrame saved to {args.output_csv}")


if __name__ == "__main__":
    main()
