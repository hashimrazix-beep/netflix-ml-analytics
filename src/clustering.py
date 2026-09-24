"""
Content Segmentation & Clustering Module for Netflix ML Analytics.
Implements K-Means clustering, PCA dimension reduction (2D and 3D),
and Plotly interactive visualizations.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.data_loader import get_preprocessed_data


class NetflixClusterer:
    """
    Unsupervised segmentation pipeline using K-Means clustering
    and PCA for 2D/3D projection and interactive visualization.
    """

    CLUSTER_PALETTE = [
        "#E50914",  # Netflix Red
        "#00D2D3",  # Vibrant Cyan
        "#54A0FF",  # Neon Blue
        "#5F27CD",  # Purple
        "#FF9F43",  # Orange
        "#10AC84",  # Emerald Green
        "#FF6B6B",  # Coral
        "#FED330",  # Yellow
    ]

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.preprocessor = None
        self.kmeans = None
        self.pca_2d = None
        self.pca_3d = None
        self.feature_matrix = None
        self.cluster_labels = None
        self.cluster_names: Dict[int, str] = {}
        self.optimal_k_results: Dict[str, Any] = {}

    def _build_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "text_genre",
                    TfidfVectorizer(max_features=300, stop_words="english"),
                    "listed_in",
                ),
                (
                    "cat_features",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["type", "rating"],
                ),
                (
                    "num_features",
                    StandardScaler(),
                    ["release_year", "duration_num"],
                ),
            ],
            remainder="drop",
        )

    def fit(self, df: pd.DataFrame) -> "NetflixClusterer":
        """Fit clustering pipeline and PCA projections."""
        self.df = df.copy().reset_index(drop=True)
        self.preprocessor = self._build_preprocessor()

        # Transform features
        X_sparse = self.preprocessor.fit_transform(self.df)
        self.feature_matrix = (
            X_sparse.toarray() if hasattr(X_sparse, "toarray") else X_sparse
        )

        # Fit KMeans
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        self.cluster_labels = self.kmeans.fit_predict(self.feature_matrix)
        self.df["cluster"] = self.cluster_labels

        # PCA 2D
        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        coords_2d = self.pca_2d.fit_transform(self.feature_matrix)
        self.df["pca_x"] = coords_2d[:, 0]
        self.df["pca_y"] = coords_2d[:, 1]

        # PCA 3D
        self.pca_3d = PCA(n_components=3, random_state=self.random_state)
        coords_3d = self.pca_3d.fit_transform(self.feature_matrix)
        self.df["pca_z"] = coords_3d[:, 2]

        # Auto-generate descriptive cluster labels
        self._generate_cluster_names()
        self.df["cluster_name"] = self.df["cluster"].map(self.cluster_names)

        return self

    def _generate_cluster_names(self) -> None:
        """Derive meaningful names for clusters based on top genre and type."""
        for c in range(self.n_clusters):
            sub = self.df[self.df["cluster"] == c]
            if len(sub) == 0:
                self.cluster_names[c] = f"Cluster {c}"
                continue

            top_type = sub["type"].mode().iloc[0] if not sub["type"].empty else "Content"
            top_genres = (
                sub["primary_genre"].value_counts().head(2).index.tolist()
            )
            genre_str = " & ".join(top_genres) if top_genres else "General"
            self.cluster_names[c] = f"Group {c}: {genre_str} ({top_type}s)"

    def compute_elbow_metrics(
        self, df: pd.DataFrame, k_range: range = range(2, 9)
    ) -> Dict[str, Any]:
        """Compute inertias and silhouette scores for range of k values."""
        preprocessor = self._build_preprocessor()
        X_sp = preprocessor.fit_transform(df)
        X = X_sp.toarray() if hasattr(X_sp, "toarray") else X_sp

        inertias = []
        silhouettes = []
        k_values = list(k_range)

        # Sample for fast silhouette scoring if dataset is large
        sample_indices = np.random.RandomState(42).choice(
            len(X), size=min(2500, len(X)), replace=False
        )
        X_sample = X[sample_indices]

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            km.fit(X)
            inertias.append(float(km.inertia_))

            labels_sample = km.predict(X_sample)
            score = silhouette_score(X_sample, labels_sample)
            silhouettes.append(round(float(score), 4))

        self.optimal_k_results = {
            "k_values": k_values,
            "inertias": inertias,
            "silhouettes": silhouettes,
        }
        return self.optimal_k_results

    def get_cluster_summary(self) -> pd.DataFrame:
        """Return a aggregated summary table for all clusters."""
        summary = (
            self.df.groupby("cluster")
            .agg(
                count=("show_id", "count"),
                dominant_type=("type", lambda x: x.mode().iloc[0]),
                dominant_rating=("rating", lambda x: x.mode().iloc[0]),
                avg_release_year=("release_year", lambda x: round(x.mean(), 1)),
                sample_titles=("title", lambda x: ", ".join(x.head(3).tolist())),
            )
            .reset_index()
        )
        summary["cluster_label"] = summary["cluster"].map(self.cluster_names)
        return summary

    def create_2d_plot(self) -> go.Figure:
        """Create a sleek dark-themed 2D PCA scatter plot using Plotly."""
        fig = px.scatter(
            self.df,
            x="pca_x",
            y="pca_y",
            color="cluster_name",
            hover_data=["title", "type", "listed_in", "rating", "release_year"],
            title=f"Netflix Content Segmentation ({self.n_clusters} Clusters, 2D PCA)",
            color_discrete_sequence=self.CLUSTER_PALETTE,
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3"),
            title_font=dict(size=18, color="#E50914"),
            legend=dict(
                title="Content Segment",
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(l=40, r=40, t=60, b=80),
        )
        return fig

    def create_3d_plot(self) -> go.Figure:
        """Create a 3D PCA scatter plot using Plotly."""
        fig = px.scatter_3d(
            self.df,
            x="pca_x",
            y="pca_y",
            z="pca_z",
            color="cluster_name",
            hover_data=["title", "type", "listed_in", "rating"],
            title=f"3D Feature Space Projection (K={self.n_clusters})",
            color_discrete_sequence=self.CLUSTER_PALETTE,
            template="plotly_dark",
            height=650,
        )
        fig.update_layout(
            paper_bgcolor="#0E1117",
            font=dict(color="#E6EDF3"),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig


def build_clusterer(df: Optional[pd.DataFrame] = None, n_clusters: int = 5) -> NetflixClusterer:
    """Helper to initialize and fit NetflixClusterer."""
    if df is None:
        df = get_preprocessed_data()
    clusterer = NetflixClusterer(n_clusters=n_clusters)
    clusterer.fit(df)
    return clusterer


if __name__ == "__main__":
    print("Fitting NetflixClusterer...")
    clusterer = build_clusterer(n_clusters=5)
    print("Clusters successfully generated!")
    summary = clusterer.get_cluster_summary()
    print(summary[["cluster", "cluster_label", "count", "dominant_type", "avg_release_year"]])
