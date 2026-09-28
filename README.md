# Tweet Sentiment Analysis

A reproducible text-mining project for preprocessing unstructured Twitter text,
visualizing word frequency, and classifying sentiment with machine-learning models
implemented in both Python and R.

## Project Goal

The project predicts one of three sentiment labels for a tweet:

- `positive`
- `negative`
- `neutral`

It uses the shared `Tweets.csv` dataset and provides two equivalent analysis
implementations for comparison and learning.

## Architecture

```text
data/
└── Tweets.csv                    # Shared source dataset

python/
├── requirements.txt              # Python dependencies
├── README.md
└── src/
    └── sentiment_analysis.py     # TF-IDF + Multinomial Naive Bayes

r/
├── README.md
└── src/
    └── sentiment_analysis.R      # Document-term matrix + Naive Bayes

docs/
├── project-presentation.pptx
└── project-report.docx

outputs/                          # Generated locally; ignored by Git
```

## Processing Pipeline

1. Load and validate the labeled Twitter dataset.
2. Normalize text by removing URLs, mentions, punctuation, numbers, and extra spaces.
3. Tokenize the cleaned text and remove English stop words.
4. Build text features using TF-IDF in Python or a document-term matrix in R.
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate accuracy, classification metrics, and a confusion matrix.
7. Generate a word cloud and frequent-word visualization.

## Python Setup

Requirements: Python 3.9 or newer.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r python/requirements.txt
python python/src/sentiment_analysis.py --dataset data/Tweets.csv
```

Python outputs are saved to `outputs/python/`.

## R Setup

Requirements: R 4.0 or newer.

Install the R packages once:

```r
install.packages(c(
  "caret", "dplyr", "ggplot2", "readr", "stringr", "tidyr",
  "tidytext", "wordcloud", "e1071", "RColorBrewer"
))
```

Run the R implementation from the project root:

```powershell
Rscript r/src/sentiment_analysis.R data/Tweets.csv outputs/r
```

R outputs are saved to `outputs/r/`.

## Dataset

The dataset contains tweet text and a sentiment label. It is used locally by both
implementations so that their preprocessing and model results can be compared.
The project does not collect live tweets from Twitter.

## Limitations

The models are educational baselines. Regex-based cleaning and bag-of-words features
do not fully understand sarcasm, context, spelling variation, emojis, or multilingual
text. A production system could compare Logistic Regression, SVM, transformer models,
and cross-validation with a larger, balanced dataset.

## Repository Hygiene

Generated images, metrics, virtual environments, R session files, and IDE metadata are
excluded through `.gitignore`. Source code, documentation, the dataset, and project
materials remain versioned.
