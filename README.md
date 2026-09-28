# Tweet Sentiment Analysis

Reproducible text-mining project for cleaning unstructured Twitter text, visualizing word frequency, and classifying sentiment with Python and R.

## Goal

Predict `positive`, `negative`, or `neutral` labels from a shared labeled CSV dataset.

## Pipeline

1. Clean URLs, mentions, punctuation, numbers, and whitespace.
2. Tokenize and remove English stop words.
3. Build TF-IDF features in Python or a document-term matrix in R.
4. Train a Naive Bayes classifier.
5. Report accuracy, classification metrics, and confusion matrices.
6. Generate word-frequency visualizations.

## Run with Python

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r python/requirements.txt
python python/src/sentiment_analysis.py --dataset data/Tweets.csv
```

## Run with R

Install the packages listed in `r/README.md`, then run `Rscript r/src/sentiment_analysis.R data/Tweets.csv outputs/r`.

## Limitations

These are educational baseline models. Sarcasm, context, spelling variation, emojis, and multilingual text are not handled reliably. The project does not collect live tweets.