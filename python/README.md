# Python Sentiment Analysis

This module trains a TF-IDF + Multinomial Naive Bayes classifier on the shared Twitter sentiment dataset.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python python/src/sentiment_analysis.py --dataset data/Tweets.csv
```

The script writes metrics, a word cloud, and a confusion matrix to `outputs/python/`.
