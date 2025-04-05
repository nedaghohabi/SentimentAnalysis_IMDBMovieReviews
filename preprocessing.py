import argparse
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm


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
    words = nltk.word_tokenize(text)  # Tokenization BEFORE punctuation removal
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
        corrected_words.append(suggestion)
    return corrected_words


def remove_stopwords(words, stop_words):
    """Removes stopwords from a list of words."""
    return [word for word in words if word not in stop_words]


def remove_punctuation(words):
    """Removes punctuation from words."""
    return [word for word in words if word not in string.punctuation]


def stemming(words):
    """Applies stemming to words."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]


def lemmatization(words):
    """Applies lemmatization to words."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def full_pipeline(
    text,
    sym_spell,
    stop_words,
    rem_punctuation=False,
    spell_check=False,
    use_stemming=False,
    use_lemmatization=False,
):
    """Runs full preprocessing pipeline with options to enable/disable steps."""
    words = basic_preprocessing(text)  # Lowercase, HTML & URL removal, tokenization
    if rem_punctuation:
        words = remove_punctuation(words)
    words = remove_stopwords(words, stop_words)  # Stopword removal
    if use_stemming:
        words = stemming(words)  # Stemming
    if use_lemmatization:
        words = lemmatization(words)  # Lemmatization
    if spell_check:
        words = spell_correction(words, sym_spell)  # Spell correction
    return " ".join(words)


def process_dataframe(
    df,
    column,
    sym_spell,
    stop_words,
    rem_punctuation=False,
    spell_check=False,
    use_stemming=False,
    use_lemmatization=False,
):
    """Applies text preprocessing to the specified column of a DataFrame with a progress bar."""
    tqdm.pandas(desc="Processing reviews")
    df[column + "_cleaned"] = df[column].progress_apply(
        lambda text: full_pipeline(
            text, sym_spell, stop_words, rem_punctuation, spell_check, use_stemming, use_lemmatization
        )
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Text preprocessing for sentiment analysis."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Path to save processed CSV file"
    )
    parser.add_argument(
        "--rem_punctuation", action="store_true", help="Enable punctuation removal"
    )
    parser.add_argument(
        "--column", type=str, default="review", help="Column name to preprocess"
    )
    parser.add_argument(
        "--spell_check", action="store_true", help="Enable spell checking"
    )
    parser.add_argument(
        "--spelling_dict",
        type=str,
        default="./data/frequency_dictionary_en_82_765.txt",
        help="Path to SymSpell dictionary",
    )
    parser.add_argument("--stemming", action="store_true", help="Enable stemming")
    parser.add_argument(
        "--lemmatization", action="store_true", help="Enable lemmatization"
    )

    args = parser.parse_args()

    # Initialize required resources
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    sym_spell = initialize_symspell(args.spelling_dict)
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
