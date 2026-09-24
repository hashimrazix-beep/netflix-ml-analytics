# 🎬 Netflix ML Analytics & Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![GitHub License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-netflix--ml--analytics-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hashimrazix-beep/netflix-ml-analytics)

---

## 📌 Executive Overview
**Netflix ML Analytics** is an end-to-end, production-grade Machine Learning and Intelligence system engineered on the Netflix global catalog dataset (8,790 titles across 10 metadata attributes).

The platform features four distinct machine learning pipelines—ranging from natural language processing and content-based recommendation to supervised multi-class prediction and unsupervised audience segmentation—wrapped in a high-performance, dark-themed Streamlit dashboard.

---

## 🌟 Core Machine Learning Pipelines

```
                             ┌──────────────────────────────────────┐
                             │          Netflix Dataset             │
                             │  (8,790 Titles / 10 Attributes)      │
                             └──────────────────┬───────────────────┘
                                                │
                 ┌──────────────────────────────┼─────────────────────────────┐
                 │                              │                             │
                 ▼                              ▼                             ▼
       ┌──────────────────┐           ┌──────────────────┐          ┌──────────────────┐
       │     Task 1       │           │   Tasks 2 & 3    │          │     Task 4       │
       │  Recommendation  │           │  Classification  │          │   Segmentation   │
       │      System      │           │     Engines      │          │   & Clustering   │
       └─────────┬────────┘           └─────────┬────────┘          └─────────┬────────┘
                 │                              │                             │
      ┌──────────┴──────────┐        ┌──────────┴──────────┐       ┌──────────┴──────────┐
      │ • TF-IDF Vectorizer │        │ • Format Predictor  │       │ • K-Means (K=3 to 8)│
      │ • Bi-gram Soup      │        │   (Movie vs TV Show)│       │ • PCA 2D/3D Proj.   │
      │ • Cosine Similarity │        │ • Rating Classifier │       │ • Elbow & Silhouette│
      │ • Fast Top-K Ranker │        │   (Multi-Class Cert)│       │ • Cluster Profiling │
      └─────────────────────┘        └─────────────────────┘       └─────────────────────┘
```

### 1. 🎬 Task 1: Content-Based Recommendation Engine
- **Objective:** Surface the top $K$ most relevant titles for any queried movie or TV show.
- **Techniques:** Natural Language Processing (NLP) metadata soup creation combining `listed_in` (genres), `director`, `country`, `title`, and `rating`.
- **Vector Space:** Sublinear TF-IDF vectorization with n-gram range `(1, 2)` yielding an $8,000$-dimensional feature representation.
- **Retrieval:** Directional Cosine Similarity computation with sub-second ranking and cross-media format filtering (Movie vs. TV Show).

### 2. 📽️ Task 2: Content Type Predictor (Binary Classification)
- **Objective:** Predict whether an incoming title is a `Movie` or `TV Show` based on metadata attributes.
- **Algorithms:** Regularized Logistic Regression vs. Random Forest Classifier (120 estimators, depth $d=14$).
- **Features:** Genre TF-IDF vectors, One-Hot encoded maturity certificates, and scaled release years.
- **Performance:** **100% Validation Accuracy**, **1.000 F1-Score**, and **1.000 ROC-AUC**, effectively detecting distinct format markers.

### 3. 🔞 Task 3: Audience Rating Classifier (Multi-Class Classification)
- **Objective:** Classify content maturity ratings across 9 standard certificates (`TV-MA`, `TV-14`, `TV-PG`, `R`, `PG-13`, `TV-Y7`, `TV-Y`, `PG`, `TV-G`).
- **Algorithms:** Random Forest Classifier with balanced class weighting to address real-world rating imbalances.
- **Features:** Text genre vectors, content format flags, numerical duration values, duration unit indicators, and production release years.
- **Inference:** Outputs top predicted certificate alongside probability confidence distributions.

### 4. 📊 Task 4: Content Segmentation & Clustering (Unsupervised ML)
- **Objective:** Discover latent viewer cohorts and content groupings across the entire catalog.
- **Optimization:** Elbow Method (Inertia curve) and Silhouette Score analysis evaluated across $K \in [2, 8]$ clusters.
- **Dimensionality Reduction:** 2D and 3D Principal Component Analysis (PCA) capturing primary variance dimensions (Format Separation, Maturity, Regional/Specialty content).
- **Interactive Visualizations:** Dark-themed Plotly scatter plots with hover metadata tooltips and cluster profile summaries.

---

## 🏗️ Repository Architecture

```
netflix-ml-analytics/
├── data/
│   └── Dataset.csv                         # 8,790 Netflix catalog entries
├── notebooks/
│   ├── 01_recommendation_engine.ipynb      # NLP TF-IDF & Cosine Similarity
│   ├── 02_content_type_prediction.ipynb    # Binary classification (Movie vs TV Show)
│   ├── 03_audience_rating_classification.ipynb # Multi-class rating prediction
│   └── 04_content_segmentation.ipynb       # K-Means clustering & PCA 2D/3D
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── data_loader.py                      # Data ingestion, parsing & feature engineering
│   ├── recommender.py                      # Content-based recommendation engine
│   ├── classifier.py                       # Supervised classification pipelines
│   └── clustering.py                       # K-Means & PCA Plotly visualizations
├── scripts/
│   └── generate_notebooks.py               # Automated notebook generator with outputs
├── app.py                                  # Production dark-themed Streamlit dashboard
├── requirements.txt                        # Production dependencies
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation
```

---

## 🖥️ Sleek Modern Streamlit Dashboard (`app.py`)

The application provides a modern, dark-themed user interface (`#0E1117` background with `#E50914` Netflix crimson accents):
- **Executive KPI Header:** Live metrics tracking Total Titles (8,790), Movies (6,126), TV Shows (2,664), Production Countries (749), and Top Genre.
- **Module 1 (Recommendation Engine):** Autocomplete search dropdown, dynamic recommendation count slider ($K=3$ to $12$), format filter, interactive poster placeholder cards with match percentage badges, and full metadata breakdowns.
- **Module 2 (Content Type Predictor):** Interactive multi-select genre picker, rating dropdown, release year slider, real-time prediction hero card, confidence meter, and validation confusion matrix.
- **Module 3 (Rating Classifier):** Content maturity certificate predictor with runtime sliders, certificate descriptions (e.g. Parental Guidance vs. Restricted), confidence bar charts, and multi-class confusion matrix.
- **Module 4 (Content Segmentation):** Interactive 2D/3D Plotly scatter plots with cluster count sliders ($K=3$ to $8$), cluster archetype summary tables, and Elbow/Silhouette analysis curves.

---

## ⚡ Quick Start & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.13)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/hashimrazix-beep/netflix-ml-analytics.git
cd netflix-ml-analytics
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` to interact with the dashboard.

---

## 🧪 Running Notebooks & Testing Modules

You can execute and verify individual modules directly via terminal:

```bash
# Test Data Ingestion & Cleaning
python src/data_loader.py

# Test Content Recommender
python src/recommender.py

# Test Classification Models
python src/classifier.py

# Test Unsupervised Clustering
python src/clustering.py
```

To run or inspect the executed Jupyter Notebooks:
```bash
jupyter notebook notebooks/
```

---

## 📊 Benchmark & Evaluation Results

| Pipeline | Model | Primary Metric | Score | Key Features |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1: Recommender** | TF-IDF + Cosine Similarity | Top-5 Relevance | **0.94+** Intra-genre match | `listed_in`, `director`, `country`, `title` |
| **Task 2: Content Type** | Random Forest Classifier | Validation Accuracy | **100.0%** | `listed_in`, `rating`, `release_year` |
| **Task 2: Content Type** | Logistic Regression | Validation Accuracy | **99.9%** | `listed_in`, `rating`, `release_year` |
| **Task 3: Audience Rating**| Random Forest (Balanced) | Weighted F1 | **0.47+** | `listed_in`, `type`, `duration`, `release_year` |
| **Task 4: Segmentation** | K-Means ($K=5$) + PCA | Silhouette Score | **Optimal at $K=5$** | Multi-dimensional metadata embeddings |

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author & Contributions
Developed with precision by **Hashim Razi** as part of the Machine Learning Internship Program.  
- GitHub: [@hashimrazix-beep](https://github.com/hashimrazix-beep)  
- Inquiries & Feedback: Welcome via Pull Requests or Issues!
