from __future__ import annotations

"""Machine learning categorization utilities."""

from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = Path("receipt_category_model.joblib")


def train_model(receipt_log_path: str | Path, model_path: Path = MODEL_PATH) -> Pipeline:
    """Train a simple text model from an existing receipt log.

    Parameters
    ----------
    receipt_log_path:
        Path to the Excel log containing at least ``vendor`` and ``category``
        columns.  A ``lines`` column is optionally used for additional text.
    model_path:
        Where to save the trained model.
    """

    df = pd.read_excel(receipt_log_path)
    if "category" not in df:
        raise ValueError("receipt log must contain a 'category' column")

    texts = (
        df.get("vendor", "").fillna("")
        + " "
        + df.get("lines", "").fillna("").astype(str)
        + " "
        + df.get("total", 0).fillna(0).astype(str)
    )
    y = df["category"].astype(str)

    pipeline: Pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(texts, y)
    joblib.dump(pipeline, model_path)
    return pipeline


def load_model(model_path: Path = MODEL_PATH) -> Optional[Pipeline]:
    """Return the saved model if it exists."""
    if model_path.exists():
        return joblib.load(model_path)
    return None


def predict_category(
    vendor: str,
    receipt_text: str,
    total: float | None,
    model: Optional[Pipeline] = None,
    model_path: Path = MODEL_PATH,
) -> Tuple[Optional[str], float]:
    """Predict the category for a receipt.

    Returns the predicted category and the associated confidence score.  If no
    model is available ``(None, 0.0)`` is returned.
    """

    if model is None:
        model = load_model(model_path)
    if model is None:
        return None, 0.0

    text = f"{vendor} {receipt_text} {total or ''}".strip()
    proba = model.predict_proba([text])[0]
    idx = int(proba.argmax())
    return str(model.classes_[idx]), float(proba[idx])


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Train receipt categorization model")
    parser.add_argument("logfile", help="Path to receipt_log.xlsx")
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH, help="Where to store the trained model"
    )
    args = parser.parse_args()
    train_model(args.logfile, args.model)
    print(f"Model saved to {args.model}")

