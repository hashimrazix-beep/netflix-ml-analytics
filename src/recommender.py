"""
Recommendation System Module for Movie ML Analytics.
Implements TF-IDF vectorization and Cosine Similarity to recommend similar titles.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


class MovieRecommender:
    """
    Content-Based Recommendation Engine using TF-IDF and Cosine Similarity.
    Recommends titles based on shared genres, directors, countries, and rating.
    """

    def __init__(self, max_features: int = 8000):
        # Each genre/director/country/rating is a single whole-entity token, so
        # "Kirsten Johnson" never matches an unrelated "Johnson".
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"\S+",
            lowercase=False,
            sublinear_tf=True,
            max_features=max_features,
        )
        self.tfidf_matrix = None
        self.df: Optional[pd.DataFrame] = None
        self.indices: Dict[str, int] = {}

    @staticmethod
    def _entity_tokens(prefix: str, text: str, skip: tuple = ()) -> List[str]:
        """Turn 'United States, France' into ['c_united_states', 'c_france']."""
        tokens = []
        for part in str(text).split(","):
            part = part.strip()
            if not part or part in skip or part.lower() == "nan":
                continue
            tokens.append(prefix + re.sub(r"\W+", "_", part.lower()).strip("_"))
        return tokens

    def _prepare_metadata_soup(self, df: pd.DataFrame) -> pd.Series:
        """Build an entity-token soup; genres are repeated to weigh them highest."""
        def make_soup(row) -> str:
            tokens = (
                self._entity_tokens("g_", row.get("listed_in", "")) * 2
                + self._entity_tokens("d_", row.get("director", ""), ("Unknown Director", "Not Given"))
                + self._entity_tokens("c_", row.get("country", ""), ("Unknown Country", "Not Given"))
                + self._entity_tokens("r_", row.get("rating", ""))
            )
            return " ".join(tokens)

        return df.apply(make_soup, axis=1)

    def fit(self, df: pd.DataFrame) -> "MovieRecommender":
        """Fit TF-IDF on the dataset and build indices."""
        self.df = df.reset_index(drop=True).copy()
        soups = self._prepare_metadata_soup(self.df)
        self.tfidf_matrix = self.vectorizer.fit_transform(soups)

        # Build case-insensitive index map; duplicate titles resolve to the first entry
        self.indices = {}
        for idx, title in enumerate(self.df["title"]):
            self.indices.setdefault(title.strip().lower(), idx)
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

        # Exclude the query itself and, if requested, other formats
        candidates = np.ones(len(self.df), dtype=bool)
        candidates[idx] = False
        if content_type_filter:
            candidates &= (
                self.df["type"].str.lower() == content_type_filter.lower()
            ).to_numpy()

        # Rank remaining candidates by similarity (descending, stable for ties)
        candidate_idx = np.flatnonzero(candidates)
        order = np.argsort(-sim_scores[candidate_idx], kind="stable")
        top_indices = candidate_idx[order][:top_n]
        top_scores = [round(float(sim_scores[i]), 4) for i in top_indices]

        results = self.df.iloc[top_indices].copy()
        results["similarity_score"] = top_scores

        columns_to_return = [
            "title",
            "type",
            "listed_in",
            "director",
            "country",
            "rating",
            "duration",
            "release_year",
            "date_added",
            "similarity_score",
        ]
        available_cols = [c for c in columns_to_return if c in results.columns]
        return results[available_cols].reset_index(drop=True)

    def save(self, filepath: str) -> None:
        """Persist recommender artifact."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MovieRecommender":
        """Load persisted recommender artifact."""
        return joblib.load(filepath)


def build_recommender(df: Optional[pd.DataFrame] = None) -> MovieRecommender:
    """Helper to initialize and fit recommender on the cleaned catalog."""
    if df is None:
        df = get_preprocessed_data()
    recommender = MovieRecommender()
    recommender.fit(df)
    return recommender


if __name__ == "__main__":
    recommender = build_recommender()
    sample_title = "Dick Johnson Is Dead"
    recs = recommender.get_recommendations(sample_title, top_n=5)
    print(f"Top 5 Recommendations for '{sample_title}':\n")
    print(recs[["title", "type", "listed_in", "similarity_score"]])
