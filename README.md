# Sentiment Classification of Movie Reviews

Contact:  
**Pooria daneshvar Kakhaki**  
Email: [daneshvarkakhaki.p@northeastern.edu](mailto:daneshvarkakhaki.p@northeastern.edu)  Department of Computer Science, Northeastern University, Boston, MA
**Neda Ghohabi Esfahani**  
Email: [ghohabiesfahani.n@northeastern.edu](mailto:ghohabiesfahani.n@northeastern.edu) Department of Bioengineering, Northeastern University

## Table of Contents

1. [Introduction](#introduction)   
3. [Dataset](#dataset)
4.  [Methodology](#methodology)  
   - [ML Models](#MLmodels)  
   - [DL Models](#DLmodels)  
5. [Environment](#environment)  
 
7. [Results and Conclusion](#results-and-conclusion)  
8. [How to Use](#how-to-use)  
9. [Limitations and Future work:](#limitations-and-future-work)
10. [References](#references)  

## Introduction: 

This project focuses on sentiment analysis of movie reviews using the [IMDb Large Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?datasetId=134715&sortBy=dateRun&tab=profile), which consists of 50,000 reviews evenly split between positive and negative sentiments. We investigate a wide range of machine learning and deep learning approaches, including classical models (Logistic Regression, Random Forest with Bag-of-Words and TF-IDF), Fully Connected Networks and LSTMs with various word embeddings, and Transformer-based architectures such as DistilBERT and RoBERTa. Models were evaluated using accuracy, F1-score, precision, and recall, offering a comprehensive comparison across embedding strategies, fine-tuning techniques, and model complexities for sentiment classification.

## Dataset
We use the Large Movie Review Dataset (IMDb), which contains 100,000 movie reviews, including 50,000 labeled examples and 50,000 unlabeled examples. The labeled portion is evenly split into a training set and a test set, each with 25,000 reviews (12,500 positive and 12,500 negative). The unlabeled reviews, typically used for clustering or zero-shot tasks, are excluded from all experiments in this project.

Since no predefined validation split is provided, we reserve 20% of the training data for validation. This results in 20,000 training samples (10,000 positive, 10,000 negative) and 5,000 validation samples (2,500 positive, 2,500 negative).

