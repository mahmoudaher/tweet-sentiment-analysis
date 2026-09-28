"""Train and evaluate a lightweight tweet sentiment classifier."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
NON_LETTER_PATTERN = re.compile(r"[^a-zA-Z\s]")


def clean_text(text: str) -> str:
    """Normalize a tweet for feature extraction."""
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = text.replace("#", " ")
    text = NON_LETTER_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load and validate the expected Twitter sentiment columns."""
    data = pd.read_csv(dataset_path, encoding="latin1")
    required_columns = {"text", "sentiment"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError("Dataset is missing columns: " + ", ".join(sorted(missing_columns)))

    data = data[["text", "sentiment"]].rename(columns={"text": "tweet", "sentiment": "label"})
    data = data.dropna(subset=["tweet", "label"])
    data["tweet"] = data["tweet"].astype(str)
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data = data[data["tweet"].str.len() > 5]
    data["clean_text"] = data["tweet"].map(clean_text)
    data = data[data["clean_text"].str.len() > 0]
    return data.reset_index(drop=True)


def train_model(data: pd.DataFrame) -> Tuple[Dict[str, object], object]:
    """Train the classifier and return metrics plus the fitted vectorizer/model."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    features = vectorizer.fit_transform(data["clean_text"])
    labels = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = MultinomialNB()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "samples": int(len(data)),
        "features": int(features.shape[1]),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "labels": sorted(labels.unique().tolist()),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return metrics, (vectorizer, model, y_test, predictions)


def save_outputs(data: pd.DataFrame, metrics: Dict[str, object], model_state: object, output_dir: Path) -> None:
    """Write machine-readable metrics and a word cloud image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    word_cloud = WordCloud(width=1200, height=600, background_color="white").generate(
        " ".join(data["clean_text"])
    )
    word_cloud.to_file(str(output_dir / "wordcloud.png"))

    _, _, y_test, predictions = model_state
    matrix = confusion_matrix(y_test, predictions)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    labels = sorted(set(y_test) | set(predictions))
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
             yticklabels=labels, xlabel="Predicted label", ylabel="True label",
             title="Sentiment confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=120)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/Tweets.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/python"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    data = load_dataset(arguments.dataset)
    metrics, model_state = train_model(data)
    save_outputs(data, metrics, model_state, arguments.output_dir)
    print(f"Analyzed {metrics['samples']} tweets")
    print(f"Vocabulary size: {metrics['features']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Outputs saved to: {arguments.output_dir}")


if __name__ == "__main__":
    main()
