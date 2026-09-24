"""
Netflix ML Analytics - Production Streamlit Dashboard
High-performance, dark-themed analytics & predictive dashboard for Netflix content intelligence.
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
# Page Configuration & Custom CSS Styling
# ==============================================================================

st.set_page_config(
    page_title="Netflix ML Analytics | AI Intelligence Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# High-end Dark Mode Netflix-Themed Custom CSS
CUSTOM_CSS = """
<style>
    /* Global Styles */
    .stApp {
        background-color: #0E1117;
        color: #E6EDF3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Top Header Bar */
    .main-header {
        background: linear-gradient(135deg, #161B22 0%, #21262D 100%);
        border-left: 5px solid #E50914;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-title span {
        color: #E50914;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #8B949E;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 28px;
    }
    .kpi-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #E50914;
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8B949E;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F0F6FC;
    }

    /* Content Cards */
    .content-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .content-card:hover {
        border-color: #58A6FF;
    }
    
    /* Recommendation Card */
    .rec-card {
        background: linear-gradient(180deg, #1C2128 0%, #161B22 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .rec-card:hover {
        transform: translateY(-4px);
        border-color: #E50914;
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.25);
    }
    .rec-poster-placeholder {
        width: 100%;
        height: 140px;
        background: linear-gradient(135deg, #21262D 0%, #30363D 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.8rem;
        margin-bottom: 14px;
        border: 1px dashed #484F58;
    }
    .rec-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        background: rgba(229, 9, 20, 0.15);
        color: #FF7B72;
        border: 1px solid rgba(229, 9, 20, 0.3);
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .rec-match-badge {
        position: absolute;
        top: 12px;
        right: 12px;
        background: #238636;
        color: #FFFFFF;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }

    /* Custom form styling */
    .stTextInput > div > div > input, .stSelectbox > div > div {
        background-color: #161B22 !important;
        color: #E6EDF3 !important;
        border-color: #30363D !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0E14 !important;
        border-right: 1px solid #21262D !important;
    }
    
    /* Result Hero Banner */
    .hero-result {
        background: linear-gradient(135deg, #1C2128 0%, #21262D 100%);
        border: 2px solid #238636;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# Cached Resource Loaders
# ==============================================================================

@st.cache_data(show_spinner="Loading and caching preprocessed Netflix data...")
def load_data() -> pd.DataFrame:
    return get_preprocessed_data()


@st.cache_resource(show_spinner="Initializing NLP Recommendation Engine...")
def get_cached_recommender(df: pd.DataFrame) -> NetflixRecommender:
    return build_recommender(df)


@st.cache_resource(show_spinner="Training Content Type Classifier (Random Forest)...")
def get_cached_type_model() -> ContentTypeClassifier:
    return train_content_type_model("rf")


@st.cache_resource(show_spinner="Training Rating Classifier (Random Forest)...")
def get_cached_rating_model() -> AudienceRatingClassifier:
    return train_rating_classifier("rf")


@st.cache_resource(show_spinner="Fitting K-Means Content Segmentation (K=5)...")
def get_cached_clusterer(df: pd.DataFrame, k: int) -> NetflixClusterer:
    return build_clusterer(df, n_clusters=k)


# Load dataset
data = load_data()

# ==============================================================================
# Top Header & Global KPIs
# ==============================================================================

st.markdown("""
<div class="main-header">
    <div class="main-title">🎬 <span>Netflix</span> ML Analytics Engine</div>
    <div class="sub-title">Production Machine Learning Pipelines & Unsupervised Audience Intelligence</div>
</div>
""", unsafe_allow_html=True)

# Metric Summary Row
kpi_cols = st.columns(5)
total_titles = len(data)
total_movies = (data["type"] == "Movie").sum()
total_tv = (data["type"] == "TV Show").sum()
total_countries = data["country"].nunique()
top_genre = data["primary_genre"].mode().iloc[0]

with kpi_cols[0]:
    st.metric(label="Total Catalog Titles", value=f"{total_titles:,}")
with kpi_cols[1]:
    st.metric(label="Movies", value=f"{total_movies:,}", delta=f"{total_movies/total_titles*100:.0f}%")
with kpi_cols[2]:
    st.metric(label="TV Shows", value=f"{total_tv:,}", delta=f"{total_tv/total_titles*100:.0f}%")
with kpi_cols[3]:
    st.metric(label="Global Countries", value=f"{total_countries}")
with kpi_cols[4]:
    st.metric(label="Top Genre", value=top_genre)


# ==============================================================================
# Sidebar Navigation
# ==============================================================================

with st.sidebar:
    st.markdown("### 📌 Navigation")
    menu_selection = st.radio(
        "Select Machine Learning Module:",
        options=[
            "🎬 Recommendation Engine",
            "📽️ Content Type Predictor",
            "🔞 Rating Classifier",
            "📊 Content Segmentation",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.success("🟢 Pipelines Online & Ready")
    st.info(f"📁 Dataset: `data/Dataset.csv` ({total_titles} entries)")

    st.markdown("---")
    st.markdown("### 💻 Architecture")
    st.caption("Engineered with **Scikit-Learn**, **TF-IDF NLP**, **Random Forest**, **K-Means**, **Plotly**, and **Streamlit**.")


# ==============================================================================
# MODULE 1: Recommendation Engine
# ==============================================================================

if menu_selection == "🎬 Recommendation Engine":
    st.markdown("## 🎬 Content-Based Recommendation Engine")
    st.markdown("Discover the most similar titles powered by **TF-IDF** metadata vectorization and **Cosine Similarity**.")

    recommender = get_cached_recommender(data)

    col_search1, col_search2, col_search3 = st.columns([3, 1, 1])
    
    # Title list for dropdown
    all_titles = sorted(data["title"].unique().tolist())
    default_title = "Dick Johnson Is Dead" if "Dick Johnson Is Dead" in all_titles else all_titles[0]
    default_idx = all_titles.index(default_title)

    with col_search1:
        selected_title = st.selectbox(
            "Select or Type a Title to Match:",
            options=all_titles,
            index=default_idx,
            help="Choose any title from the Netflix catalog."
        )

    with col_search2:
        top_k = st.slider("Recommendations (K):", min_value=3, max_value=12, value=6, step=1)

    with col_search3:
        type_filter = st.selectbox(
            "Filter Content Format:",
            options=["All Formats", "Movie", "TV Show"],
            index=0,
        )
        content_type_arg = None if type_filter == "All Formats" else type_filter

    # Target Title Metadata Card
    target_row = data[data["title"] == selected_title].iloc[0]
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid #E50914;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h3 style="margin: 0; color: #FFFFFF;">🎯 Selected: {target_row['title']}</h3>
                <p style="margin: 4px 0 0 0; color: #8B949E; font-size: 0.9rem;">
                    <strong>Director:</strong> {target_row['director']} | 
                    <strong>Country:</strong> {target_row['country']} | 
                    <strong>Year:</strong> {target_row['release_year']}
                </p>
            </div>
            <div style="margin-top: 8px;">
                <span class="rec-badge">{target_row['type']}</span>
                <span class="rec-badge" style="background: rgba(88, 166, 255, 0.15); color: #58A6FF; border-color: rgba(88, 166, 255, 0.3);">{target_row['rating']}</span>
                <span class="rec-badge" style="background: rgba(35, 134, 54, 0.15); color: #3FB950; border-color: rgba(35, 134, 54, 0.3);">{target_row['duration']}</span>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 0.85rem; color: #C9D1D9;">
            <strong>Genres:</strong> {target_row['listed_in']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Generate Recommendations
    with st.spinner("Calculating multi-dimensional cosine similarity..."):
        recs = recommender.get_recommendations(
            title=selected_title,
            top_n=top_k,
            content_type_filter=content_type_arg,
        )

    st.markdown(f"### 🌟 Top {len(recs)} Recommendations")

    # Render recommendations in dynamic grid (3 columns per row)
    grid_cols = st.columns(3)
    for idx, (_, row) in enumerate(recs.iterrows()):
        col_target = grid_cols[idx % 3]
        match_pct = int(row["similarity_score"] * 100)
        icon = "🎬" if row["type"] == "Movie" else "📺"

        with col_target:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-match-badge">{match_pct}% Match</div>
                <div class="rec-poster-placeholder">{icon}</div>
                <div style="flex-grow: 1;">
                    <h4 style="margin: 0 0 6px 0; color: #FFFFFF; font-size: 1.05rem;">{row['title']}</h4>
                    <p style="margin: 0 0 10px 0; font-size: 0.8rem; color: #8B949E;">
                        {row['release_year']} • {row['duration']} • {row['country']}
                    </p>
                    <div style="margin-bottom: 8px;">
                        <span class="rec-badge">{row['type']}</span>
                        <span class="rec-badge" style="background: rgba(88, 166, 255, 0.15); color: #58A6FF; border-color: rgba(88, 166, 255, 0.3);">{row['rating']}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #8B949E; line-height: 1.3;">
                        <strong>Genres:</strong> {row['listed_in']}
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.75rem; color: #6E7681; border-top: 1px solid #30363D; padding-top: 8px;">
                    <strong>Director:</strong> {row['director']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")  # Vertical spacing

    # Expandable raw table
    with st.expander("🔍 View Raw Similarity Breakdown Table"):
        st.dataframe(recs, use_container_width=True)


# ==============================================================================
# MODULE 2: Content Type Predictor
# ==============================================================================

elif menu_selection == "📽️ Content Type Predictor":
    st.markdown("## 📽️ Content Type Prediction Model")
    st.markdown("Supervised classification pipeline predicting whether an entry is a **Movie** or **TV Show** based on metadata attributes.")

    type_clf = get_cached_type_model()

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📝 Enter Content Attributes")
        with st.form("type_prediction_form"):
            all_genres = sorted(list({
                g.strip()
                for sublist in data["listed_in"].str.split(",")
                for g in sublist
                if g.strip()
            }))
            
            selected_genres = st.multiselect(
                "Select Genre Tags:",
                options=all_genres,
                default=["Documentaries"] if "Documentaries" in all_genres else [all_genres[0]],
                help="Select one or more genre categories."
            )

            rating_choice = st.selectbox(
                "Audience Rating / Certificate:",
                options=["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y", "PG", "TV-G", "NR"],
                index=0,
            )

            release_yr = st.slider(
                "Release Year:",
                min_value=1950,
                max_value=2026,
                value=2021,
                step=1,
            )

            submit_type = st.form_submit_button("🚀 Predict Content Type", use_container_width=True)

    with col_right:
        st.markdown("### 🔮 Prediction & Model Confidence")
        if submit_type:
            genre_str = ", ".join(selected_genres) if selected_genres else "Dramas"
            pred_input = {
                "listed_in": genre_str,
                "rating": rating_choice,
                "release_year": release_yr,
            }

            pred_result = type_clf.predict(pred_input)
            predicted_class = pred_result["prediction"]
            confidence = pred_result["confidence"]
            probs = pred_result["probabilities"]

            icon = "🎬" if predicted_class == "Movie" else "📺"
            accent_color = "#E50914" if predicted_class == "Movie" else "#58A6FF"

            st.markdown(f"""
            <div class="hero-result" style="border-color: {accent_color};">
                <div style="font-size: 3rem;">{icon}</div>
                <h2 style="margin: 8px 0; color: #FFFFFF;">Predicted: <span style="color: {accent_color};">{predicted_class}</span></h2>
                <p style="color: #8B949E; margin: 0; font-size: 1.1rem;">
                    Model Confidence: <strong>{confidence*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Probability Distribution:")
            for cls_name, prob in probs.items():
                st.write(f"**{cls_name}**: `{prob*100:.1f}%`")
                st.progress(float(prob))

        else:
            st.info("👈 Configure features on the left and click **Predict Content Type** to run inference.")

    # Model Evaluation Metrics
    st.markdown("---")
    st.markdown("### 📈 Model Evaluation Metrics & Benchmark")
    eval_cols = st.columns(4)
    with eval_cols[0]:
        st.metric(label="Validation Accuracy", value=f"{type_clf.metrics['accuracy']*100:.2f}%")
    with eval_cols[1]:
        st.metric(label="Weighted Precision", value=f"{type_clf.metrics['precision']*100:.2f}%")
    with eval_cols[2]:
        st.metric(label="Weighted Recall", value=f"{type_clf.metrics['recall']*100:.2f}%")
    with eval_cols[3]:
        st.metric(label="F1-Score", value=f"{type_clf.metrics['f1_score']*100:.2f}%")

    with st.expander("📊 View Confusion Matrix Plot"):
        cm = np.array(type_clf.metrics["confusion_matrix"])
        classes = type_clf.metrics["classes"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Format", y="Actual Format", color="Sample Count"),
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="Reds",
            template="plotly_dark",
        )
        fig_cm.update_layout(
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3"),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        st.plotly_chart(fig_cm, use_container_width=True)


# ==============================================================================
# MODULE 3: Audience Rating Classifier
# ==============================================================================

elif menu_selection == "🔞 Rating Classifier":
    st.markdown("## 🔞 Audience Rating Multi-Class Classifier")
    st.markdown("Predict the appropriate content maturity rating (`TV-MA`, `PG-13`, `R`, `TV-PG`, etc.) from genre, runtime, and media format.")

    rating_clf = get_cached_rating_model()

    col_r_left, col_r_right = st.columns([1, 1], gap="large")

    with col_r_left:
        st.markdown("### 📝 Enter Content Specifications")
        with st.form("rating_prediction_form"):
            input_type = st.radio("Content Format:", options=["Movie", "TV Show"], horizontal=True)

            all_genres_list = sorted(list({
                g.strip()
                for sublist in data["listed_in"].str.split(",")
                for g in sublist
                if g.strip()
            }))

            genre_input = st.multiselect(
                "Content Categories / Genres:",
                options=all_genres_list,
                default=["Children & Family Movies", "Comedies"] if "Children & Family Movies" in all_genres_list else [all_genres_list[0]],
            )

            if input_type == "Movie":
                duration_val = st.slider("Movie Duration (Minutes):", min_value=30, max_value=240, value=95, step=5)
                unit_val = "min"
            else:
                duration_val = st.slider("TV Show Duration (Seasons):", min_value=1, max_value=15, value=1, step=1)
                unit_val = "season"

            release_y = st.slider("Production Year:", min_value=1950, max_value=2026, value=2021, step=1)

            submit_rating = st.form_submit_button("🎯 Predict Audience Rating", use_container_width=True)

    with col_r_right:
        st.markdown("### 🎯 Predicted Maturity Classification")
        if submit_rating:
            joined_genres = ", ".join(genre_input) if genre_input else "Dramas"
            pred_rating_input = {
                "listed_in": joined_genres,
                "type": input_type,
                "duration_num": duration_val,
                "duration_unit": unit_val,
                "release_year": release_y,
            }

            res_rating = rating_clf.predict(pred_rating_input)
            pred_cert = res_rating["prediction"]
            cert_conf = res_rating["confidence"]
            all_prob = res_rating["probabilities"]

            # Descriptive certificate definitions
            cert_info = {
                "TV-MA": "Mature Audience Only (Ages 17+ / Unsuitable for children)",
                "R": "Restricted (Under 17 requires accompanying parent)",
                "TV-14": "Parents Strongly Cautioned (May be unsuitable for ages under 14)",
                "PG-13": "Parents Strongly Cautioned (Some material may be inappropriate for children under 13)",
                "TV-PG": "Parental Guidance Suggested (May contain material parents find unsuitable for younger children)",
                "PG": "Parental Guidance Suggested",
                "TV-Y7": "Suitable for Children age 7 and older",
                "TV-Y": "Designed to be appropriate for all children",
                "TV-G": "General Audience (Suitable for all ages)",
            }
            desc = cert_info.get(pred_cert, "Standard content rating certificate.")

            st.markdown(f"""
            <div class="hero-result" style="border-color: #E50914;">
                <span class="rec-badge" style="font-size: 1.4rem; padding: 6px 14px; background: rgba(229, 9, 20, 0.2);">{pred_cert}</span>
                <h3 style="margin: 12px 0 6px 0; color: #FFFFFF;">Target Rating: <span style="color: #E50914;">{pred_cert}</span></h3>
                <p style="color: #8B949E; margin: 0 0 10px 0; font-size: 0.95rem;">{desc}</p>
                <div style="font-size: 1.1rem; color: #3FB950;">
                    Model Confidence: <strong>{cert_conf*100:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top 3 probabilities chart
            top_classes = list(all_prob.keys())[:4]
            top_scores = [all_prob[c] * 100 for c in top_classes]

            fig_probs = go.Figure(go.Bar(
                x=top_scores,
                y=top_classes,
                orientation="h",
                marker=dict(color=["#E50914", "#58A6FF", "#3FB950", "#D29922"][:len(top_classes)]),
                text=[f"{s:.1f}%" for s in top_scores],
                textposition="auto",
            ))
            fig_probs.update_layout(
                title="Top Predicted Class Probabilities",
                paper_bgcolor="#161B22",
                plot_bgcolor="#161B22",
                font=dict(color="#E6EDF3"),
                xaxis=dict(title="Probability (%)", range=[0, 100]),
                yaxis=dict(autorange="reversed"),
                height=260,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_probs, use_container_width=True)

        else:
            st.info("👈 Enter title genre and runtime parameters on the left to classify audience rating.")

    # Rating Model Benchmark Section
    st.markdown("---")
    st.markdown("### 📊 Benchmark Metrics & Confusion Matrix")
    m_cols = st.columns(4)
    with m_cols[0]:
        st.metric(label="Overall Accuracy", value=f"{rating_clf.metrics['accuracy']*100:.2f}%")
    with m_cols[1]:
        st.metric(label="Weighted Precision", value=f"{rating_clf.metrics['weighted_precision']*100:.2f}%")
    with m_cols[2]:
        st.metric(label="Weighted Recall", value=f"{rating_clf.metrics['weighted_recall']*100:.2f}%")
    with m_cols[3]:
        st.metric(label="Weighted F1-Score", value=f"{rating_clf.metrics['weighted_f1']*100:.2f}%")

    with st.expander("🔍 View Normalized Multi-Class Confusion Matrix"):
        cm_r = np.array(rating_clf.metrics["confusion_matrix"])
        r_classes = rating_clf.metrics["classes"]
        fig_cm_r = px.imshow(
            cm_r,
            labels=dict(x="Predicted Rating", y="True Rating", color="Count"),
            x=r_classes,
            y=r_classes,
            text_auto=True,
            color_continuous_scale="Purples",
            template="plotly_dark",
        )
        fig_cm_r.update_layout(
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3"),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        st.plotly_chart(fig_cm_r, use_container_width=True)


# ==============================================================================
# MODULE 4: Content Segmentation & Clustering
# ==============================================================================

elif menu_selection == "📊 Content Segmentation":
    st.markdown("## 📊 Unsupervised Content Segmentation & Clustering")
    st.markdown("Cluster Netflix's catalog into distinct viewer cohorts and genres using **K-Means Clustering** with **PCA 2D/3D Dimensionality Reduction**.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        k_clusters = st.slider("Select Cluster Count (K):", min_value=3, max_value=8, value=5, step=1)
    with col_ctrl2:
        view_mode = st.radio("Visualization Mode:", options=["2D PCA Projection", "3D Interactive Space"], horizontal=True)

    clusterer = get_cached_clusterer(data, k_clusters)

    # Plot Visualizations
    if view_mode == "2D PCA Projection":
        fig_scatter = clusterer.create_2d_plot()
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        fig_scatter_3d = clusterer.create_3d_plot()
        st.plotly_chart(fig_scatter_3d, use_container_width=True)

    # Cluster Summary Cards & Table
    st.markdown("### 📋 Content Cluster Archetypes & Characteristics")
    cluster_summary = clusterer.get_cluster_summary()
    st.dataframe(
        cluster_summary[[
            "cluster_label", "count", "dominant_type", "dominant_rating", "avg_release_year", "sample_titles"
        ]],
        use_container_width=True,
    )

    # Elbow & Silhouette Analysis Expander
    with st.expander("📈 Optimal K Analysis (Elbow Method & Silhouette Scores)"):
        with st.spinner("Computing Inertia and Silhouette metrics across K=2 to 8..."):
            elbow_data = clusterer.compute_elbow_metrics(data, k_range=range(2, 9))
            k_vals = elbow_data["k_values"]
            inertias = elbow_data["inertias"]
            silhouettes = elbow_data["silhouettes"]

            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=k_vals,
                y=inertias,
                mode="lines+markers",
                name="Inertia (Sum of Squared Distances)",
                line=dict(color="#E50914", width=3),
                marker=dict(size=8),
            ))
            fig_elbow.update_layout(
                title="Elbow Method Curve (Optimal K Selection)",
                xaxis=dict(title="Cluster Count (K)"),
                yaxis=dict(title="Inertia"),
                paper_bgcolor="#161B22",
                plot_bgcolor="#161B22",
                font=dict(color="#E6EDF3"),
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig_elbow, use_container_width=True)

            fig_sil = go.Figure()
            fig_sil.add_trace(go.Bar(
                x=k_vals,
                y=silhouettes,
                name="Silhouette Score",
                marker=dict(color="#58A6FF"),
                text=[f"{s:.3f}" for s in silhouettes],
                textposition="auto",
            ))
            fig_sil.update_layout(
                title="Silhouette Score Analysis across K",
                xaxis=dict(title="Cluster Count (K)"),
                yaxis=dict(title="Silhouette Coefficient"),
                paper_bgcolor="#161B22",
                plot_bgcolor="#161B22",
                font=dict(color="#E6EDF3"),
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig_sil, use_container_width=True)


# ==============================================================================
# Footer
# ==============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8B949E; font-size: 0.85rem; padding: 10px 0;">
    Netflix ML Analytics Platform • Built with Python, Scikit-Learn, Streamlit, Plotly
</div>
""", unsafe_allow_html=True)
