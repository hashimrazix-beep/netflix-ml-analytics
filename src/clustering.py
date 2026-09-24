"""
Content Segmentation Module for Movie ML Analytics.
K-Means clustering over catalog metadata with 2D/3D PCA projections.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Optional

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

from src.data_loader import get_preprocessed_data


class MovieClusterer:
    """
    Unsupervised segmentation pipeline using K-Means clustering
    and PCA for 2D/3D projection and interactive visualization.
    """

    CLUSTER_PALETTE = [
        "#A78BFA",  # Lavender
        "#FBBF24",  # Marquee Gold
        "#2DD4BF",  # Teal
        "#FB7185",  # Coral
        "#60A5FA",  # Sky
        "#F472B6",  # Orchid
        "#A3E635",  # Lime
        "#FDBA74",  # Peach
    ]

    # Shared dark, transparent chart styling so plots sit on the app background
    PLOT_LAYOUT = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#C9C3EA"),
        hoverlabel=dict(bgcolor="#1A1633", bordercolor="#3B3470", font_color="#F4F1FF"),
    )

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.df: Optional[pd.DataFrame] = None
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

    def fit(self, df: pd.DataFrame) -> "MovieClusterer":
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
        """Create a dark-themed 2D PCA scatter plot."""
        fig = px.scatter(
            self.df,
            x="pca_x",
            y="pca_y",
            color="cluster_name",
            hover_data=["title", "type", "listed_in", "rating", "release_year"],
            color_discrete_sequence=self.CLUSTER_PALETTE,
        )
        fig.update_traces(marker=dict(size=6, opacity=0.75, line=dict(width=0)))
        fig.update_layout(
            **self.PLOT_LAYOUT,
            xaxis=dict(title="PC 1", showgrid=False, zeroline=False),
            yaxis=dict(title="PC 2", showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            legend=dict(title=None, orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        return fig

    def create_3d_plot(self) -> go.Figure:
        """Create a dark-themed 3D PCA scatter plot."""
        fig = px.scatter_3d(
            self.df,
            x="pca_x",
            y="pca_y",
            z="pca_z",
            color="cluster_name",
            hover_data=["title", "type", "listed_in", "rating"],
            color_discrete_sequence=self.CLUSTER_PALETTE,
        )
        fig.update_traces(marker=dict(size=3, opacity=0.8))
        fig.update_layout(
            **self.PLOT_LAYOUT,
            scene=dict(
                xaxis=dict(title="PC 1", backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(title="PC 2", backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(title="PC 3", backgroundcolor="rgba(0,0,0,0)"),
            ),
            legend=dict(title=None, orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig


def build_clusterer(df: Optional[pd.DataFrame] = None, n_clusters: int = 5) -> MovieClusterer:
    """Helper to initialize and fit MovieClusterer."""
    if df is None:
        df = get_preprocessed_data()
    clusterer = MovieClusterer(n_clusters=n_clusters)
    clusterer.fit(df)
    return clusterer


if __name__ == "__main__":
    print("Fitting MovieClusterer...")
    clusterer = build_clusterer(n_clusters=5)
    print("Clusters successfully generated!")
    summary = clusterer.get_cluster_summary()
    print(summary[["cluster", "cluster_label", "count", "dominant_type", "avg_release_year"]])
