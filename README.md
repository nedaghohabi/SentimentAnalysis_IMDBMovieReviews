# Sentiment Classification of Movie Reviews

Contact:  
**Pooria daneshvar Kakhaki**  
Email: [daneshvarkakhaki.p@northeastern.edu](mailto:daneshvarkakhaki.p@northeastern.edu)  Department of Computer Science, Northeastern University, Boston, MA

**Neda Ghohabi Esfahani**  
Email: [ghohabiesfahani.n@northeastern.edu](mailto:ghohabiesfahani.n@northeastern.edu) Department of Bioengineering, Northeastern University

> ## ⚠️ Important Notice
> 
> **To demo the repository**, please refer to the instructions provided below.  
> Before running the demo, please ensure that all required libraries are installed and the environment is properly set up by following the instructions provided in the Environment section.
> 🔴 *Note: Failing to install the required packages or correctly set up the dataset may result in errors during the demo.*

## Demo Instructions

To quickly run a demo of this repository:

Run the demo script:
```{python}
python run_scripts.py [--text "I loved the movie" [--cpu]
```
Arguments:
- `--cpu` :To force CPU training if GPU is not available or desired
- `--text`: To classify a custom review text,


This script will:
- Download and prepare the IMDb dataset.
- Train a Logistic Regression model on TF-IDF representations.
- Perform inference on a provided sample review (default: "I really liked the movie").
- Train a Fully Connected Network (FCN) using pretrained Word2Vec embeddings.
- Perform inference again using the FCN model.

You will need additional info including logs and required files to reuse the trained models in `./results` directory.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Environment](#environment)


## Introduction: 

This project focuses on sentiment analysis of movie reviews using the [IMDb Large Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?datasetId=134715&sortBy=dateRun&tab=profile), which consists of 50,000 reviews evenly split between positive and negative sentiments. We investigate a wide range of machine learning and deep learning approaches, including classical models (Logistic Regression, Random Forest with Bag-of-Words and TF-IDF), Fully Connected Networks and LSTMs with various word embeddings, and Transformer-based architectures such as DistilBERT and RoBERTa. Models were evaluated using accuracy, F1-score, precision, and recall, offering a comprehensive comparison across embedding strategies, fine-tuning techniques, and model complexities for sentiment classification.

## Dataset
We use the Large Movie Review Dataset (IMDb), which contains 100,000 movie reviews, including 50,000 labeled examples and 50,000 unlabeled examples. ![dataset](images/dataset_sankey.png) The labeled portion is evenly split into a training set and a test set, each with 25,000 reviews (12,500 positive and 12,500 negative). The unlabeled reviews, typically used for clustering or zero-shot tasks, are excluded from all experiments in this project. Since no predefined validation split is provided, we reserve 20% of the training data for validation. This results in 20,000 training samples (10,000 positive, 10,000 negative) and 5,000 validation samples (2,500 positive, 2,500 negative).

To download and prepared the dataset, run the following script:

```shell
pip prepare_dataset.py
```

## Environment

Prepare the virtual environment:

You can use the provided `requirements.py` file:
```shell
pip install -r requirements.txt
```

Otherwise, you we need the following packages:

```shell
# Core Python Libraries
numpy
pandas
scikit-learn
argparse

# Deep Learning
torch
torchvision
torchaudio

# Transformers (Hugging Face)
transformers
datasets

# Tokenization Utils
sentencepiece

# Plotting and Visualization
matplotlib
seaborn
wordcloud

# Word2Vec
gensim

# Text Processing
nltk
joblib
symspellpy

# Training Utilities
tqdm
```



