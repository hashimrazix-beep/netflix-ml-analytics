"""
Recommendation System Module for Netflix ML Analytics.
Implements TF-IDF vectorization and Cosine Similarity to recommend similar titles.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from src.data_loader import get_preprocessed_data


class NetflixRecommender:
    """
    Content-Based Recommendation Engine using TF-IDF and Cosine Similarity.
    Recommends titles based on genres, director, country, and title keywords.
    """

    def __init__(self, max_features: int = 8000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.tfidf_matrix = None
        self.df: Optional[pd.DataFrame] = None
        self.indices: Dict[str, int] = {}

    def _prepare_metadata_soup(self, df: pd.DataFrame) -> pd.Series:
        """Create clean feature soup combining listed_in, director, country, and title."""
        def make_soup(row):
            genres = str(row.get("listed_in", "")).replace(",", " ").lower()
            director = str(row.get("director", "")).lower()
            if director in ["unknown director", "not given"]:
                director = ""
            country = str(row.get("country", "")).lower()
            if country in ["unknown country", "not given"]:
                country = ""
            title = str(row.get("title", "")).lower()
            content_type = str(row.get("type", "")).lower()
            rating = str(row.get("rating", "")).lower()
            
            # Combine key features
            soup = f"{title} {genres} {genres} {director} {country} {content_type} {rating}"
            return " ".join(soup.split())

        return df.apply(make_soup, axis=1)

    def fit(self, df: pd.DataFrame) -> "NetflixRecommender":
        """Fit TF-IDF on the dataset and build indices."""
        self.df = df.reset_index(drop=True).copy()
        soups = self._prepare_metadata_soup(self.df)
        self.tfidf_matrix = self.vectorizer.fit_transform(soups)

        # Build case-insensitive index map
        self.indices = {
            title.strip().lower(): idx for idx, title in enumerate(self.df["title"])
        }
        return self

    def find_title(self, query: str) -> Optional[str]:
        """Find the closest matching title in the dataset."""
        q_clean = query.strip().lower()
        if q_clean in self.indices:
            idx = self.indices[q_clean]
            return self.df.loc[idx, "title"]

        # Substring search
        matches = [t for t in self.df["title"] if q_clean in t.lower()]
        if matches:
            return matches[0]

        return None

    def get_recommendations(
        self,
        title: str,
        top_n: int = 5,
        content_type_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get the top N recommendations for a given title.

        Parameters:
        - title: Name of the movie or TV show.
        - top_n: Number of recommendations to return (default 5).
        - content_type_filter: 'Movie', 'TV Show', or None.

        Returns:
        - DataFrame of recommendations with similarity score and metadata.
        """
        if self.df is None or self.tfidf_matrix is None:
            raise ValueError("Recommender has not been fitted. Call fit() first.")

        matched_title = self.find_title(title)
        if matched_title is None:
            raise KeyError(f"Title '{title}' not found in the catalog.")

        idx = self.indices[matched_title.strip().lower()]

        # Compute cosine similarity for this single item against all items
        target_vec = self.tfidf_matrix[idx]
        sim_scores = cosine_similarity(target_vec, self.tfidf_matrix).flatten()

        # Get sorted indices (descending)
        ranked_indices = np.argsort(-sim_scores)

        # Filter out self
        ranked_indices = [i for i in ranked_indices if i != idx]

        # Apply content_type filter if specified
        if content_type_filter:
            ranked_indices = [
                i for i in ranked_indices
                if str(self.df.loc[i, "type"]).lower() == content_type_filter.lower()
            ]

        top_indices = ranked_indices[:top_n]
        top_scores = [round(float(sim_scores[i]), 4) for i in top_indices]

        results = self.df.iloc[top_indices].copy()
        results["similarity_score"] = top_scores
        results["query_title"] = matched_title

        columns_to_return = [
            "title",
            "type",
            "listed_in",
            "director",
            "country",
            "rating",
            "duration",
            "release_year",
            "similarity_score",
        ]
        available_cols = [c for c in columns_to_return if c in results.columns]
        return results[available_cols].reset_index(drop=True)

    def save(self, filepath: str) -> None:
        """Persist recommender artifact."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "NetflixRecommender":
        """Load persisted recommender artifact."""
        return joblib.load(filepath)


def build_recommender(df: Optional[pd.DataFrame] = None) -> NetflixRecommender:
    """Helper to initialize and fit recommender on cleaned Netflix data."""
    if df is None:
        df = get_preprocessed_data()
    recommender = NetflixRecommender()
    recommender.fit(df)
    return recommender


if __name__ == "__main__":
    recommender = build_recommender()
    sample_title = "Dick Johnson Is Dead"
    recs = recommender.get_recommendations(sample_title, top_n=5)
    print(f"Top 5 Recommendations for '{sample_title}':\n")
    print(recs[["title", "type", "listed_in", "similarity_score"]])
