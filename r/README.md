# R Sentiment Analysis

This module uses the shared Twitter dataset to clean text, build a document-term
matrix, train a Naive Bayes classifier, and save evaluation and visualization outputs.

## Install packages

Run once in R:

```r
install.packages(c(
  "caret", "dplyr", "ggplot2", "readr", "stringr", "tidyr",
  "tidytext", "wordcloud", "e1071", "RColorBrewer"
))
```

## Run

From the project root:

```bash
Rscript r/src/sentiment_analysis.R data/Tweets.csv outputs/r
```

The script writes metrics, a confusion matrix, a word cloud, and a top-words chart
to `outputs/r/`.