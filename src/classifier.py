"""
Classification Models Module for Netflix ML Analytics.
Implements:
1. ContentTypeClassifier (Binary: Movie vs TV Show)
2. AudienceRatingClassifier (Multi-Class: TV-MA, TV-14, R, PG-13, TV-PG, etc.)
Includes full evaluation metrics, confusion matrix computation, and real-time inference.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

from src.data_loader import get_preprocessed_data, parse_duration


# ==============================================================================
# Task 2: Content Type Classifier (Binary: Movie vs TV Show)
# ==============================================================================

class ContentTypeClassifier:
    """
    Binary classifier predicting whether a title is a 'Movie' or 'TV Show'.
    Uses genre information, audience rating, release year, and country.
    """

    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type.lower()
        self.pipeline: Optional[Pipeline] = None
        self.metrics: Dict[str, Any] = {}
        self.classes_: Optional[np.ndarray] = None

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "text_genre",
                    TfidfVectorizer(max_features=500, ngram_range=(1, 2)),
                    "listed_in",
                ),
                (
                    "cat_features",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["rating", "primary_genre"],
                ),
                (
                    "num_features",
                    StandardScaler(),
                    ["release_year"],
                ),
            ],
            remainder="drop",
        )

        if self.model_type == "rf":
            classifier = RandomForestClassifier(
                n_estimators=120,
                max_depth=14,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1,
            )
        else:
            classifier = LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
            )

        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

    def train(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Dict[str, Any]:
        """Train classifier and compute validation metrics."""
        data = df.dropna(subset=["type", "listed_in"]).copy()
        X = data[["listed_in", "rating", "primary_genre", "release_year"]]
        y = data["type"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.classes_ = self.pipeline.named_steps["classifier"].classes_

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)

        # ROC AUC for binary
        try:
            movie_col_idx = list(self.classes_).index("Movie")
            roc = roc_auc_score(
                (y_test == "Movie").astype(int), y_proba[:, movie_col_idx]
            )
        except Exception:
            roc = None

        self.metrics = {
            "model_type": self.model_type,
            "accuracy": round(float(acc), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc), 4) if roc is not None else "N/A",
            "confusion_matrix": cm.tolist(),
            "classes": list(self.classes_),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
        return self.metrics

    def predict(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real-time inference on a single title's metadata.
        Expected keys: 'listed_in', 'rating', 'release_year'
        """
        if self.pipeline is None:
            raise ValueError("Model is not trained. Call train() first.")

        listed_in = str(metadata.get("listed_in", "Comedies, Dramas"))
        primary_genre = listed_in.split(",")[0].strip()
        rating = str(metadata.get("rating", "TV-MA"))
        release_year = int(metadata.get("release_year", 2021))

        input_df = pd.DataFrame([{
            "listed_in": listed_in,
            "rating": rating,
            "primary_genre": primary_genre,
            "release_year": release_year,
        }])

        pred_label = self.pipeline.predict(input_df)[0]
        probas = self.pipeline.predict_proba(input_df)[0]
        prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes_, probas)
        }

        return {
            "prediction": pred_label,
            "confidence": round(float(np.max(probas)), 4),
            "probabilities": prob_dict,
        }


# ==============================================================================
# Task 3: Audience Rating Classifier (Multi-Class)
# ==============================================================================

class AudienceRatingClassifier:
    """
    Multi-Class classifier predicting content rating (TV-MA, PG-13, R, TV-14, etc.)
    from genre, duration, release year, and type.
    """

    TOP_RATINGS = [
        "TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y", "PG", "TV-G"
    ]

    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type.lower()
        self.pipeline: Optional[Pipeline] = None
        self.metrics: Dict[str, Any] = {}
        self.classes_: Optional[np.ndarray] = None

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "text_genre",
                    TfidfVectorizer(max_features=600, ngram_range=(1, 2)),
                    "listed_in",
                ),
                (
                    "cat_features",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["type", "duration_unit"],
                ),
                (
                    "num_features",
                    StandardScaler(),
                    ["duration_num", "release_year"],
                ),
            ],
            remainder="drop",
        )

        if self.model_type == "rf":
            classifier = RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_split=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "dt":
            classifier = DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=6,
                class_weight="balanced",
                random_state=42,
            )
        else:
            classifier = LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                C=1.0,
                random_state=42,
            )

        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

    def train(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Dict[str, Any]:
        """Filter to top standard ratings, train multi-class classifier, and evaluate."""
        data = df[df["rating"].isin(self.TOP_RATINGS)].copy()
        X = data[["listed_in", "type", "duration_unit", "duration_num", "release_year"]]
        y = data["rating"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.classes_ = self.pipeline.named_steps["classifier"].classes_

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)

        self.metrics = {
            "model_type": self.model_type,
            "accuracy": round(float(acc), 4),
            "weighted_precision": round(float(p_w), 4),
            "weighted_recall": round(float(r_w), 4),
            "weighted_f1": round(float(f1_w), 4),
            "macro_f1": round(float(f1_m), 4),
            "confusion_matrix": cm.tolist(),
            "classes": list(self.classes_),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }
        return self.metrics

    def predict(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real-time inference for content audience rating.
        Expected keys: 'listed_in', 'type', 'duration_str' or ('duration_num', 'duration_unit'), 'release_year'
        """
        if self.pipeline is None:
            raise ValueError("Model is not trained. Call train() first.")

        listed_in = str(metadata.get("listed_in", "Dramas, International Movies"))
        content_type = str(metadata.get("type", "Movie"))
        release_year = int(metadata.get("release_year", 2021))

        if "duration_num" in metadata and "duration_unit" in metadata:
            duration_num = int(metadata["duration_num"])
            duration_unit = str(metadata["duration_unit"])
        else:
            duration_str = str(metadata.get("duration", "95 min"))
            duration_num, duration_unit = parse_duration(duration_str)

        input_df = pd.DataFrame([{
            "listed_in": listed_in,
            "type": content_type,
            "duration_unit": duration_unit,
            "duration_num": duration_num,
            "release_year": release_year,
        }])

        pred_label = self.pipeline.predict(input_df)[0]
        probas = self.pipeline.predict_proba(input_df)[0]
        prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes_, probas)
        }

        # Sort probabilities descending
        sorted_probas = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        return {
            "prediction": pred_label,
            "confidence": round(float(np.max(probas)), 4),
            "probabilities": sorted_probas,
        }


# ==============================================================================
# Helper functions
# ==============================================================================

def train_content_type_model(model_type: str = "rf") -> ContentTypeClassifier:
    """Helper to train and return a ContentTypeClassifier instance."""
    df = get_preprocessed_data()
    clf = ContentTypeClassifier(model_type=model_type)
    clf.train(df)
    return clf


def train_rating_classifier(model_type: str = "rf") -> AudienceRatingClassifier:
    """Helper to train and return an AudienceRatingClassifier instance."""
    df = get_preprocessed_data()
    clf = AudienceRatingClassifier(model_type=model_type)
    clf.train(df)
    return clf


if __name__ == "__main__":
    print("--- Testing Content Type Classifier ---")
    type_clf = train_content_type_model("rf")
    print(f"Content Type Accuracy: {type_clf.metrics['accuracy']:.4f}")
    sample_item = {
        "listed_in": "Documentaries, International Movies",
        "rating": "PG-13",
        "release_year": 2020,
    }
    pred = type_clf.predict(sample_item)
    print(f"Sample prediction: {pred}")

    print("\n--- Testing Rating Classifier ---")
    rating_clf = train_rating_classifier("rf")
    print(f"Rating Classifier Accuracy: {rating_clf.metrics['accuracy']:.4f}")
    sample_rating_item = {
        "listed_in": "Children & Family Movies, Comedies",
        "type": "Movie",
        "duration": "85 min",
        "release_year": 2021,
    }
    pred_rating = rating_clf.predict(sample_rating_item)
    print(f"Sample rating prediction: {pred_rating}")
