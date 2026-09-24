# 🎞️ Movie ML Analytics

[![CI](https://github.com/hashimrazix-beep/netflix-ml-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/hashimrazix-beep/netflix-ml-analytics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Four machine learning pipelines on a catalog of 8,790 movies and TV shows, with an animated Streamlit dashboard for exploring them.

| Page | What it does | How |
| :--- | :--- | :--- |
| **Discover** | Recommends similar titles | TF-IDF over whole-entity tokens (genres, directors, countries, rating) ranked by cosine similarity |
| **Format** | Predicts Movie vs TV Show | Random Forest / Logistic Regression on format-neutral genres, rating, release year |
| **Rating** | Predicts the maturity rating (9 classes) | Class-balanced Random Forest on genres, format, duration, release year |
| **Segments** | Groups the catalog into segments | K-Means with elbow/silhouette selection and 2D/3D PCA maps |

---

## Results

All numbers come from a stratified 80/20 split and are reproduced in the executed notebooks.

| Task | Model | Metric | Score |
| :--- | :--- | :--- | :--- |
| Recommendations | TF-IDF + cosine | Top-5 share of a query genre (200 random titles) | 98.8% |
| Format | Random Forest | Accuracy / ROC-AUC | 84.4% / 0.944 |
| Format | Logistic Regression | Accuracy / ROC-AUC | 84.4% / 0.924 |
| Rating | Random Forest (balanced) | Accuracy / Weighted F1 | 47.2% / 0.482 |
| Rating | Decision Tree | Accuracy / Weighted F1 | 43.5% / 0.448 |
| Segments | K-Means | Best silhouette | 0.365 at K = 3 |

### A note on the format classifier

The catalog's raw genre labels name the format ("TV Dramas" vs "Dramas", "Docuseries" vs "Documentaries"). Every raw label belongs to only one format, so a model trained on them reads the answer straight from the input and scores a meaningless ~100%. The pipeline maps each label to a **format-neutral genre** (`src/data_loader.py`, `NEUTRAL_GENRE_MAP`) before training. The 84% accuracy shown above is an honest estimate. The majority-class baseline is 70%.

---

## Project layout

```
├── app.py                     # Streamlit dashboard
├── src/
│   ├── data_loader.py         # Loading, cleaning, duration parsing, neutral genres
│   ├── recommender.py         # MovieRecommender
│   ├── classifier.py          # ContentTypeClassifier, AudienceRatingClassifier
│   └── clustering.py          # MovieClusterer (K-Means + PCA + Plotly)
├── notebooks/                 # One executed notebook per task
├── scripts/generate_notebooks.py  # Rebuilds and re-executes the notebooks
├── tests/                     # pytest suite
├── data/Dataset.csv           # Catalog data
└── .streamlit/config.toml     # Theme
```

---

## Quick start

```bash
git clone https://github.com/hashimrazix-beep/netflix-ml-analytics.git
cd netflix-ml-analytics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. The models train the first time you open each page, which takes a few seconds, and are cached after that.

### Development

```bash
pip install -r requirements-dev.txt
pytest -q                              # run the test suite
python scripts/generate_notebooks.py   # rebuild + execute all notebooks
```

Each module can also be run on its own, e.g. `python src/recommender.py`.

---

## Dashboard

The dashboard uses a dark "Midnight Marquee" palette: deep ink-violet with lavender, coral, and gold accents. It features:
- Pill navigation along the top, with no sidebar.
- A slowly drifting aurora background and a shimmering gradient wordmark.
- Cards and results that rise in one after another as a page loads, and lift on hover.
- Similarity and probability bars that animate as they fill.
- Support for `prefers-reduced-motion`, which turns all animation off.

---

## License

MIT. See [LICENSE](LICENSE).

## Author

**Hashim Razi** · [@hashimrazix-beep](https://github.com/hashimrazix-beep)
