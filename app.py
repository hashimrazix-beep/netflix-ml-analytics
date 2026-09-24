"""
Netflix ML Analytics - Production Streamlit Dashboard
Decluttered, high-end, modern intelligence platform for the Netflix catalog.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import get_preprocessed_data, parse_duration
from src.recommender import NetflixRecommender, build_recommender
from src.classifier import (
    ContentTypeClassifier,
    AudienceRatingClassifier,
    train_content_type_model,
    train_rating_classifier,
)
from src.clustering import NetflixClusterer, build_clusterer


# ==============================================================================
# Page Configuration & Clean Minimalist Styling
# ==============================================================================

st.set_page_config(
    page_title="Netflix ML Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DECLUTTERED_CSS = """
<style>
    /* Global Clean Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
    }

    /* Minimalist Top Brand Bar */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 24px;
    }
    .brand-logo {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: #E50914;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-sub {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        letter-spacing: 0px;
    }
    .brand-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(229, 9, 20, 0.1);
        color: #FF5A5F;
        border: 1px solid rgba(229, 9, 20, 0.25);
    }

    /* Sleek Cards */
    .clean-card {
        background: #111622;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: border-color 0.2s ease;
    }
    .clean-card:hover {
        border-color: #334155;
    }

    /* Recommendation Item Card */
    .movie-item {
        background: #111622;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 18px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .movie-item:hover {
        transform: translateY(-2px);
        border-color: #E50914;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    }
    .match-pill {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .type-pill {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
        background: #1E293B;
        color: #94A3B8;
    }
    .genre-tag {
        font-size: 0.75rem;
        color: #94A3B8;
        line-height: 1.4;
    }

    /* Prediction Result Box */
    .pred-hero {
        background: linear-gradient(145deg, #131A29 0%, #111622 100%);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 22px 24px;
        text-align: center;
        margin-top: 8px;
    }
    .pred-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 6px 0;
    }
    .pred-confidence {
        font-size: 0.95rem;
        color: #94A3B8;
    }

    /* Form container */
    div[data-testid="stForm"] {
        background: #111622;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 20px;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #0A0D13 !important;
        border-right: 1px solid #1E293B !important;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #E50914 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: background-color 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #F40612 !important;
    }

    /* Compact Metrics */
    div[data-testid="stMetric"] {
        background: #111622;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 10px 14px;
    }
</style>
"""
st.markdown(DECLUTTERED_CSS, unsafe_allow_html=True)


# ==============================================================================
# Cached Resources
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    return get_preprocessed_data()


@st.cache_resource(show_spinner=False)
def get_recommender(df: pd.DataFrame) -> NetflixRecommender:
    return build_recommender(df)


@st.cache_resource(show_spinner=False)
def get_type_model() -> ContentTypeClassifier:
    return train_content_type_model("rf")


@st.cache_resource(show_spinner=False)
def get_rating_model() -> AudienceRatingClassifier:
    return train_rating_classifier("rf")


@st.cache_resource(show_spinner=False)
def get_clusterer(df: pd.DataFrame, k: int) -> NetflixClusterer:
    return build_clusterer(df, n_clusters=k)


# Load data
df = load_catalog()
total_titles = len(df)
movie_count = int((df["type"] == "Movie").sum())
tv_count = int((df["type"] == "TV Show").sum())

# ==============================================================================
# Header & Navigation
# ==============================================================================

# Top Brand Bar
st.markdown(f"""
<div class="brand-bar">
    <div>
        <div class="brand-logo">NETFLIX <span class="brand-sub">ML ANALYTICS</span></div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
        <span class="brand-badge">Catalog: {total_titles:,} Titles</span>
        <span class="brand-badge" style="background: rgba(59, 130, 246, 0.1); color: #60A5FA; border-color: rgba(59, 130, 246, 0.25);">
            {movie_count:,} Movies • {tv_count:,} Series
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### Modules")
    active_tab = st.radio(
        "Select Pipeline",
        options=[
            "Recommendations",
            "Format Classifier",
            "Rating Classifier",
            "Content Segmentation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Catalog Quick Stats")
    st.caption(f"**Total Records:** {total_titles:,}")
    st.caption(f"**Unique Genres:** {df['primary_genre'].nunique()}")
    st.caption(f"**Countries:** {df['country'].nunique()}")
    st.caption(f"**Year Range:** {int(df['release_year'].min())} – {int(df['release_year'].max())}")


# ==============================================================================
# TAB 1: RECOMMENDATIONS
# ==============================================================================

if active_tab == "Recommendations":
    st.subheader("Content Recommendation Engine")
    st.caption("TF-IDF metadata embedding with Cosine Similarity ranking.")

    recommender = get_recommender(df)

    # Search & Filter Row
    all_titles = sorted(df["title"].unique().tolist())
    def_title = "Dick Johnson Is Dead" if "Dick Johnson Is Dead" in all_titles else all_titles[0]
    def_idx = all_titles.index(def_title)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        selected_title = st.selectbox("Search or Select Title:", all_titles, index=def_idx)
    with col2:
        top_k = st.slider("Results:", min_value=3, max_value=9, value=6, step=1)
    with col3:
        filter_type = st.selectbox("Filter Format:", ["All", "Movie", "TV Show"])
        type_arg = None if filter_type == "All" else filter_type

    # Current Item Summary Pill Card
    current = df[df["title"] == selected_title].iloc[0]
    st.markdown(f"""
    <div class="clean-card" style="border-left: 3px solid #E50914;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <strong style="color: #FFFFFF; font-size: 1.05rem;">{current['title']}</strong>
                <span style="color: #64748B; font-size: 0.85rem; margin-left: 8px;">({current['release_year']})</span>
            </div>
            <div style="display: flex; gap: 6px;">
                <span class="type-pill">{current['type']}</span>
                <span class="type-pill">{current['rating']}</span>
                <span class="type-pill">{current['duration']}</span>
            </div>
        </div>
        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px;">
            <strong>Genres:</strong> {current['listed_in']} &nbsp;|&nbsp; 
            <strong>Director:</strong> {current['director']} &nbsp;|&nbsp; 
            <strong>Origin:</strong> {current['country']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Get Recommendations
    recs = recommender.get_recommendations(
        title=selected_title,
        top_n=top_k,
        content_type_filter=type_arg,
    )

    st.markdown("#### Similar Titles")
    grid_cols = st.columns(3)
    for i, (_, row) in enumerate(recs.iterrows()):
        col = grid_cols[i % 3]
        match_score = int(row["similarity_score"] * 100)
        with col:
            st.markdown(f"""
            <div class="movie-item">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <span class="match-pill">{match_score}% Match</span>
                        <span class="type-pill">{row['type']}</span>
                    </div>
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 4px;">
                        {row['title']}
                    </div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 8px;">
                        {row['release_year']} • {row['duration']} • {row['rating']}
                    </div>
                    <div class="genre-tag">
                        {row['listed_in']}
                    </div>
                </div>
                <div style="font-size: 0.72rem; color: #475569; margin-top: 12px; border-top: 1px solid #1E293B; padding-top: 6px;">
                    {row['director']} ({row['country']})
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")


# ==============================================================================
# TAB 2: FORMAT CLASSIFIER
# ==============================================================================

elif active_tab == "Format Classifier":
    st.subheader("Content Format Predictor")
    st.caption("Supervised binary classification (Movie vs. TV Show).")

    type_clf = get_type_model()
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("##### Title Metadata")
        with st.form("type_form"):
            unique_genres = sorted(list({
                g.strip()
                for sublist in df["listed_in"].str.split(",")
                for g in sublist
                if g.strip()
            }))
            
            chosen_genres = st.multiselect(
                "Genres:",
                options=unique_genres,
                default=["Documentaries"] if "Documentaries" in unique_genres else [unique_genres[0]],
            )

            chosen_rating = st.selectbox(
                "Certificate Rating:",
                ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y", "PG", "TV-G", "NR"],
            )

            chosen_year = st.slider("Release Year:", 1960, 2026, 2021)
            submit_type = st.form_submit_button("Predict Format")

    with col_r:
        st.markdown("##### Prediction Output")
        if submit_type:
            genre_str = ", ".join(chosen_genres) if chosen_genres else "Dramas"
            res = type_clf.predict({
                "listed_in": genre_str,
                "rating": chosen_rating,
                "release_year": chosen_year,
            })

            label = res["prediction"]
            conf = res["confidence"] * 100
            accent = "#E50914" if label == "Movie" else "#38BDF8"

            st.markdown(f"""
            <div class="pred-hero" style="border-top: 3px solid {accent};">
                <span class="type-pill" style="font-size: 0.8rem; padding: 4px 10px;">Predicted Format</span>
                <div class="pred-title" style="color: {accent};">{label}</div>
                <div class="pred-confidence">Confidence: <strong>{conf:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            for cls_name, prob in res["probabilities"].items():
                col_n, col_p = st.columns([2, 5])
                with col_n:
                    st.caption(f"**{cls_name}**")
                with col_p:
                    st.progress(float(prob))
        else:
            st.info("Select metadata attributes and click **Predict Format**.")

    # Performance Strip
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Validation Accuracy", f"{type_clf.metrics['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{type_clf.metrics['precision']*100:.1f}%")
    m3.metric("Recall", f"{type_clf.metrics['recall']*100:.1f}%")
    m4.metric("F1-Score", f"{type_clf.metrics['f1_score']*100:.1f}%")


# ==============================================================================
# TAB 3: RATING CLASSIFIER
# ==============================================================================

elif active_tab == "Rating Classifier":
    st.subheader("Audience Maturity Rating Classifier")
    st.caption("Multi-class classification across 9 viewer age ratings.")

    rating_clf = get_rating_model()
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("##### Content Specs")
        with st.form("rating_form"):
            in_type = st.radio("Format:", ["Movie", "TV Show"], horizontal=True)

            all_g = sorted(list({
                g.strip()
                for sublist in df["listed_in"].str.split(",")
                for g in sublist
                if g.strip()
            }))
            in_genres = st.multiselect(
                "Genres:",
                options=all_g,
                default=["Children & Family Movies", "Comedies"] if "Children & Family Movies" in all_g else [all_g[0]],
            )

            if in_type == "Movie":
                in_dur = st.slider("Duration (Minutes):", 40, 220, 95, step=5)
                unit_val = "min"
            else:
                in_dur = st.slider("Duration (Seasons):", 1, 12, 1, step=1)
                unit_val = "season"

            in_year = st.slider("Release Year:", 1960, 2026, 2021)
            submit_rating = st.form_submit_button("Predict Maturity Rating")

    with col_r:
        st.markdown("##### Prediction Output")
        if submit_rating:
            g_str = ", ".join(in_genres) if in_genres else "Dramas"
            res_r = rating_clf.predict({
                "listed_in": g_str,
                "type": in_type,
                "duration_num": in_dur,
                "duration_unit": unit_val,
                "release_year": in_year,
            })

            cert = res_r["prediction"]
            conf = res_r["confidence"] * 100

            labels_map = {
                "TV-MA": "Mature Audience Only (18+)",
                "R": "Restricted (Under 17 requires adult)",
                "TV-14": "Parents Strongly Cautioned (14+)",
                "PG-13": "Parents Strongly Cautioned (13+)",
                "TV-PG": "Parental Guidance Recommended",
                "PG": "Parental Guidance Suggested",
                "TV-Y7": "Suitable for Children 7+",
                "TV-Y": "All Children",
                "TV-G": "General Audience",
            }
            desc = labels_map.get(cert, "Standard Rating")

            st.markdown(f"""
            <div class="pred-hero" style="border-top: 3px solid #E50914;">
                <span class="type-pill" style="font-size: 0.8rem; padding: 4px 10px;">Predicted Certificate</span>
                <div class="pred-title" style="color: #E50914;">{cert}</div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 6px;">{desc}</div>
                <div class="pred-confidence">Confidence: <strong>{conf:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            # Clean top probabilities chart
            top_items = list(res_r["probabilities"].items())[:4]
            chart_df = pd.DataFrame({
                "Rating": [k for k, _ in top_items],
                "Probability": [v * 100 for _, v in top_items],
            })

            fig = go.Figure(go.Bar(
                x=chart_df["Probability"],
                y=chart_df["Rating"],
                orientation="h",
                marker=dict(color="#E50914"),
                text=[f"{p:.1f}%" for p in chart_df["Probability"]],
                textposition="auto",
            ))
            fig.update_layout(
                paper_bgcolor="#111622",
                plot_bgcolor="#111622",
                font=dict(color="#94A3B8", size=11),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(title="Probability (%)", range=[0, 100]),
                height=180,
                margin=dict(l=30, r=20, t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Input content duration and genres, then click **Predict Maturity Rating**.")

    # Performance Strip
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Multi-Class Accuracy", f"{rating_clf.metrics['accuracy']*100:.1f}%")
    m2.metric("Weighted Precision", f"{rating_clf.metrics['weighted_precision']*100:.1f}%")
    m3.metric("Weighted Recall", f"{rating_clf.metrics['weighted_recall']*100:.1f}%")
    m4.metric("Weighted F1", f"{rating_clf.metrics['weighted_f1']*100:.1f}%")


# ==============================================================================
# TAB 4: CONTENT SEGMENTATION
# ==============================================================================

elif active_tab == "Content Segmentation":
    st.subheader("Unsupervised Catalog Clustering")
    st.caption("K-Means feature clustering with Principal Component Analysis (PCA).")

    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        k_val = st.slider("Cluster Count (K):", 3, 7, 5)
    with ctrl2:
        mode_val = st.radio("Visualization:", ["2D Projection", "3D Space"], horizontal=True)

    clusterer = get_clusterer(df, k_val)

    if mode_val == "2D Projection":
        fig_2d = clusterer.create_2d_plot()
        fig_2d.update_layout(
            paper_bgcolor="#0B0E14",
            plot_bgcolor="#111622",
            height=480,
            margin=dict(l=20, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_2d, use_container_width=True)
    else:
        fig_3d = clusterer.create_3d_plot()
        fig_3d.update_layout(
            paper_bgcolor="#0B0E14",
            height=540,
            margin=dict(l=10, r=10, t=20, b=20),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("##### Cluster Distribution & Archetypes")
    summary = clusterer.get_cluster_summary()
    st.dataframe(
        summary[["cluster_label", "count", "dominant_type", "dominant_rating", "avg_release_year", "sample_titles"]],
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================================
# Clean Footer
# ==============================================================================

st.markdown("""
<div style="border-top: 1px solid rgba(255, 255, 255, 0.08); margin-top: 40px; padding-top: 16px; font-size: 0.75rem; color: #475569; text-align: center;">
    Netflix ML Analytics • Production Machine Learning System
</div>
""", unsafe_allow_html=True)
