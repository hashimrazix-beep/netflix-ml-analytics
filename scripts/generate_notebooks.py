"""
Notebook Generator Script
Builds the four task notebooks from source cells and executes them, so every
output shown in the notebooks is produced by actually running the code.

Usage:
    python scripts/generate_notebooks.py
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SETUP = """import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path.cwd().parent))
from src.data_loader import get_preprocessed_data

df = get_preprocessed_data('../data/Dataset.csv')
print(f"Loaded {len(df):,} titles with {len(df.columns)} columns.")"""


def md(text: str):
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


NOTEBOOKS = {
    "01_recommendation_engine.ipynb": [
        md("""
# Task 1: Content-Based Recommendation Engine

Recommend the top-$K$ most similar titles for any movie or TV show in the catalog using:
1. An **entity-token metadata soup** built from genres, directors, countries, and rating.
2. **TF-IDF** weighting (sublinear term frequency).
3. **Cosine similarity** ranking.
"""),
        md(r"""
## 1. Method

**TF-IDF** for term $t$ in document $d$ of corpus $D$:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left(\ln\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} + 1\right)$$

**Cosine similarity** between vectors $\mathbf{u}, \mathbf{v}$:
$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
Because TF-IDF rows are $L_2$-normalized, this reduces to a dot product.
"""),
        code(SETUP + "\ndf[['title', 'type', 'director', 'country', 'rating', 'listed_in']].head()"),
        md("""
## 2. Metadata Soup

Each genre, director, country, and rating becomes **one whole token** (e.g. `d_kirsten_johnson`),
so a director named *Johnson* never matches an unrelated title containing the word *Johnson*.
Genres are repeated so they carry the most weight.
"""),
        code("""from src.recommender import MovieRecommender

recommender = MovieRecommender().fit(df)
soups = recommender._prepare_metadata_soup(df.head(3))
for title, soup in zip(df['title'].head(3), soups):
    print(f"[{title}] {soup}")
print(f"\\nTF-IDF matrix: {recommender.tfidf_matrix.shape}")"""),
        md("## 3. Recommendations"),
        code("recommender.get_recommendations('Dick Johnson Is Dead', top_n=5)[['title', 'type', 'listed_in', 'similarity_score']]"),
        code("recommender.get_recommendations('Midnight Mass', top_n=5)[['title', 'type', 'listed_in', 'similarity_score']]"),
        code("recommender.get_recommendations('Midnight Mass', top_n=5, content_type_filter='Movie')[['title', 'type', 'listed_in', 'similarity_score']]"),
        md("""
## 4. Genre Overlap Check

A simple quality proxy: the share of recommendations that share at least one genre with the query title.
"""),
        code("""rng = np.random.default_rng(42)
sample_titles = rng.choice(df['title'].unique(), size=200, replace=False)

def genre_set(s):
    return {g.strip() for g in s.split(',')}

overlaps = []
for t in sample_titles:
    query_genres = genre_set(df.loc[df['title'] == t, 'listed_in'].iloc[0])
    recs = recommender.get_recommendations(t, top_n=5)
    overlaps += [bool(query_genres & genre_set(g)) for g in recs['listed_in']]

print(f"Top-5 genre overlap over {len(sample_titles)} random titles: {np.mean(overlaps):.1%}")"""),
        md("""
## 5. Notes

- A single sparse dot product against the full matrix keeps queries fast at this catalog size.
- For much larger catalogs, approximate nearest-neighbour search (FAISS, HNSW) would replace the full scan.
"""),
    ],
    "02_content_type_prediction.ipynb": [
        md("""
# Task 2: Content Type Prediction (Movie vs TV Show)

Binary classification predicting a title's format from its metadata.

- **Target:** $y \\in \\{\\text{Movie}, \\text{TV Show}\\}$
- **Features:** format-neutral genres, audience rating, release year
"""),
        code(SETUP + "\nprint(df['type'].value_counts(normalize=True).round(3))"),
        md("""
## 1. Avoiding Target Leakage

The catalog's raw genre labels name the format ("TV Dramas" vs "Dramas", "Docuseries" vs
"Documentaries"). Every raw label maps to only one format, so a model trained on them simply reads the
answer and reaches a meaningless ~100% accuracy.

The pipeline therefore maps each label to a **format-neutral genre** before training.
"""),
        code("""leak = (df.assign(genre=df['listed_in'].str.split(', ')).explode('genre')
        .groupby('genre')['type'].agg(lambda s: (s == 'Movie').mean()))
print(f"Raw genre labels that map to a single format: {((leak == 0) | (leak == 1)).mean():.0%}")
df[['listed_in', 'genres_neutral']].drop_duplicates().head(8)"""),
        md("## 2. Model Training & Comparison"),
        code("""from src.classifier import ContentTypeClassifier

rf_model = ContentTypeClassifier(model_type='rf')
rf_results = rf_model.train(df)
lr_model = ContentTypeClassifier(model_type='lr')
lr_results = lr_model.train(df)

pd.DataFrame([
    {"Model": name, **{k: r[k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']}}
    for name, r in [("Random Forest", rf_results), ("Logistic Regression", lr_results)]
])"""),
        md("## 3. Confusion Matrix (Random Forest)"),
        code("""classes = rf_results['classes']
pd.DataFrame(rf_results['confusion_matrix'],
             index=[f"Actual {c}" for c in classes],
             columns=[f"Pred {c}" for c in classes])"""),
        md("## 4. Inference Demo"),
        code("""test_cases = [
    {"listed_in": "Documentaries, International Movies", "rating": "PG-13", "release_year": 2020},
    {"listed_in": "Crime TV Shows, Docuseries", "rating": "TV-MA", "release_year": 2021},
    {"listed_in": "Children & Family Movies, Comedies", "rating": "TV-Y", "release_year": 2019},
]
for item in test_cases:
    res = rf_model.predict(item)
    print(f"{item['listed_in']} ({item['rating']}) -> {res['prediction']} ({res['confidence']:.1%})")"""),
        md("""
## 5. Takeaways

- With leakage removed, the remaining signal is the rating system (TV-style vs MPAA ratings), release year,
  and genres that only really exist in one format (e.g. Reality, Stand-Up & Talk).
- The resulting accuracy is an honest estimate of how well format can be inferred from descriptive metadata.
"""),
    ],
    "03_audience_rating_classification.ipynb": [
        md("""
# Task 3: Audience Rating Classification

Multi-class classification of maturity rating from genres, format, duration, and release year:
$$y \\in \\{ \\text{TV-MA}, \\text{TV-14}, \\text{TV-PG}, \\text{R}, \\text{PG-13}, \\text{TV-Y7}, \\text{TV-Y}, \\text{PG}, \\text{TV-G} \\}$$
"""),
        code(SETUP + "\nprint(df['rating'].value_counts().head(10))"),
        md("""
## 1. Model Training & Comparison

- **Random Forest** with balanced class weights to counter class imbalance.
- **Decision Tree** as an interpretable baseline.
"""),
        code("""from src.classifier import AudienceRatingClassifier

rf_rating = AudienceRatingClassifier(model_type='rf')
rf_metrics = rf_rating.train(df)
dt_rating = AudienceRatingClassifier(model_type='dt')
dt_metrics = dt_rating.train(df)

pd.DataFrame([
    {"Model": name, "Accuracy": m['accuracy'], "Weighted F1": m['weighted_f1'], "Macro F1": m['macro_f1']}
    for name, m in [("Random Forest", rf_metrics), ("Decision Tree", dt_metrics)]
])"""),
        md("## 2. Per-Class Breakdown (Random Forest)"),
        code("""report = rf_metrics['classification_report']
pd.DataFrame({c: report[c] for c in rf_metrics['classes']}).T.round(3)"""),
        md("## 3. Inference Demo"),
        code("""samples = [
    {"listed_in": "Kids' TV, Anime Series", "type": "TV Show", "duration": "1 Season", "release_year": 2021},
    {"listed_in": "Stand-Up Comedy", "type": "Movie", "duration": "65 min", "release_year": 2020},
    {"listed_in": "Horror Movies, Thrillers", "type": "Movie", "duration": "98 min", "release_year": 2018},
]
for s in samples:
    res = rf_rating.predict(s)
    top3 = ", ".join(f"{k}: {v:.1%}" for k, v in list(res['probabilities'].items())[:3])
    print(f"{s['listed_in']} ({s['type']}) -> {res['prediction']}  [{top3}]")"""),
        md("""
## 4. Findings

- Specialised genres (kids' content, stand-up, horror) produce the most confident predictions.
- Broad genres such as dramas and comedies spread across TV-14 / TV-MA / R; plot synopses or
  transcripts would be needed to separate them further.
"""),
    ],
    "04_content_segmentation.ipynb": [
        md("""
# Task 4: Content Segmentation & Clustering

Unsupervised grouping of the catalog:
1. Build a feature space from genres, format, rating, release year, and duration.
2. Choose $K$ with the **elbow method** and **silhouette scores**.
3. Fit **K-Means** at the best $K$ and project titles to 2D/3D with **PCA**.
4. Profile the resulting clusters.
"""),
        code(SETUP),
        md("## 1. Choosing K"),
        code("""from src.clustering import MovieClusterer

metrics = MovieClusterer().compute_elbow_metrics(df, k_range=range(2, 9))
pd.DataFrame({
    'K': metrics['k_values'],
    'Inertia': np.round(metrics['inertias'], 1),
    'Silhouette': metrics['silhouettes'],
})"""),
        md("## 2. PCA Explained Variance"),
        code("""from src.clustering import build_clusterer

best_k = metrics['k_values'][int(np.argmax(metrics['silhouettes']))]
print(f"Best K by silhouette: {best_k}")
clusterer = build_clusterer(df, n_clusters=best_k)
for i, var in enumerate(clusterer.pca_3d.explained_variance_ratio_, 1):
    print(f"Component {i}: {var:.2%}")"""),
        md("## 3. Cluster Profiles"),
        code("""summary = clusterer.get_cluster_summary()
summary[['cluster_label', 'count', 'dominant_type', 'dominant_rating', 'avg_release_year', 'sample_titles']]"""),
        md("""
## 4. Takeaways

- Format and maturity rating dominate the principal components, so clusters largely separate along those lines.
- Cluster profiles give a quick map of where the catalog is dense and where it is thin.
"""),
    ],
}


def generate_all_notebooks() -> None:
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    for filename, cells in NOTEBOOKS.items():
        print(f"Executing {filename}...")
        nb = new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        }
        NotebookClient(
            nb, timeout=600, kernel_name="python3",
            resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
        ).execute()
        nbformat.write(nb, NOTEBOOKS_DIR / filename)
    print("All notebooks generated and executed.")


if __name__ == "__main__":
    generate_all_notebooks()
