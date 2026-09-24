"""
Movie ML Analytics - Streamlit Dashboard
Recommendations, format & rating prediction, and catalog segmentation.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.data_loader import get_preprocessed_data
from src.recommender import MovieRecommender, build_recommender
from src.classifier import (
    ContentTypeClassifier,
    AudienceRatingClassifier,
    train_content_type_model,
    train_rating_classifier,
)
from src.clustering import MovieClusterer, build_clusterer


st.set_page_config(
    page_title="Movie ML Analytics",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==============================================================================
# Theme: "Midnight Marquee" - ink violet base, lavender / gold / coral / teal accents
# ==============================================================================

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg: #0B0917;
  --surface: rgba(24, 20, 48, 0.72);
  --surface-solid: #16132B;
  --line: rgba(167, 139, 250, 0.14);
  --line-strong: rgba(167, 139, 250, 0.38);
  --text: #F4F1FF;
  --muted: #9C95C9;
  --faint: #6B6594;
  --violet: #A78BFA;
  --gold: #FBBF24;
  --coral: #FB7185;
  --teal: #2DD4BF;
  --grad: linear-gradient(100deg, #A78BFA 0%, #FB7185 50%, #FBBF24 100%);
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
}

html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family: 'Outfit', system-ui, sans-serif !important;
}

/* ---------- Canvas & drifting aurora ---------- */
.stApp { background: var(--bg); color: var(--text); }
.stApp::before {
  content: "";
  position: fixed; inset: -20%;
  background:
    radial-gradient(40% 35% at 15% 10%, rgba(167, 139, 250, 0.20), transparent 70%),
    radial-gradient(35% 30% at 85% 15%, rgba(251, 113, 133, 0.14), transparent 70%),
    radial-gradient(40% 35% at 60% 95%, rgba(45, 212, 191, 0.10), transparent 70%);
  animation: aurora 22s ease-in-out infinite alternate;
  pointer-events: none; z-index: 0;
}
@keyframes aurora {
  0%   { transform: translate3d(0, 0, 0) rotate(0deg); }
  50%  { transform: translate3d(3%, -2%, 0) rotate(4deg); }
  100% { transform: translate3d(-3%, 2%, 0) rotate(-3deg); }
}

/* ---------- Hide Streamlit chrome ---------- */
header[data-testid="stHeader"], footer, #MainMenu,
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"],
div[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 2.4rem !important; max-width: 1180px !important; position: relative; z-index: 1; }

/* ---------- Entrance animations ---------- */
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes pop  { from { opacity: 0; transform: scale(0.94); } to { opacity: 1; transform: none; } }
@keyframes grow { from { width: 0; } }
@keyframes shimmer { to { background-position: 200% center; } }
@keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.55); } 50% { box-shadow: 0 0 0 6px rgba(45, 212, 191, 0); } }

div[data-testid="stMainBlockContainer"] div[data-testid="stVerticalBlock"] > div {
  animation: rise 0.6s var(--ease) both;
}

/* ---------- Hero ---------- */
.hero { text-align: center; margin-bottom: 1.6rem; }
.wordmark {
  font-size: clamp(2.2rem, 5vw, 3.4rem); font-weight: 800; letter-spacing: -0.035em; line-height: 1.05;
  background: var(--grad); background-size: 200% auto;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: shimmer 6s linear infinite;
}
.tagline { color: var(--muted); font-size: 1.02rem; margin-top: 0.45rem; font-weight: 400; }
.stats { display: inline-flex; gap: 1.4rem; margin-top: 1rem; color: var(--faint); font-size: 0.86rem; flex-wrap: wrap; justify-content: center; }
.stats b { color: var(--text); font-weight: 600; }
.live { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--teal); margin-right: 6px; vertical-align: middle; animation: pulse 2.4s infinite; }

/* ---------- Navigation (segmented control) ---------- */
div[data-testid="stButtonGroup"] > div {
  background: var(--surface); border: 1px solid var(--line); border-radius: 999px; padding: 5px; gap: 4px;
  backdrop-filter: blur(12px);
}
div[data-testid="stButtonGroup"] button {
  border: none !important; border-radius: 999px !important; background: transparent !important;
  color: var(--muted) !important; padding: 0.45rem 1.15rem !important; font-weight: 500 !important;
  transition: color 0.25s var(--ease), background 0.35s var(--ease), transform 0.25s var(--ease) !important;
}
div[data-testid="stButtonGroup"] button:hover { color: var(--text) !important; transform: translateY(-1px); }
div[data-testid="stButtonGroup"] button[aria-checked="true"],
div[data-testid="stButtonGroup"] button[kind$="Active"] {
  background: var(--grad) !important; background-size: 200% auto !important; color: #140F26 !important; font-weight: 700 !important;
  box-shadow: 0 6px 24px -6px rgba(251, 113, 133, 0.55);
  animation: shimmer 5s linear infinite;
}
div[data-testid="stButtonGroup"] button p { font-size: 0.92rem !important; color: inherit !important; }

/* ---------- Section heading ---------- */
.section { margin: 2rem 0 1.1rem; }
.section h2 { font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; padding: 0; color: var(--text); }
.section p { color: var(--muted); margin: 0.25rem 0 0; font-size: 0.95rem; }
.label { color: var(--faint); font-size: 0.74rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; margin: 1.4rem 0 0.7rem; }

/* ---------- Cards ---------- */
.card {
  position: relative; background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.1rem 1.2rem; backdrop-filter: blur(12px); margin-bottom: 1rem;
  animation: rise 0.7s var(--ease) both;
  transition: transform 0.35s var(--ease), border-color 0.35s var(--ease), box-shadow 0.35s var(--ease);
}
.card:hover { transform: translateY(-4px); border-color: var(--line-strong); box-shadow: 0 18px 40px -18px rgba(167, 139, 250, 0.45); }
.card .title { font-weight: 600; font-size: 1.02rem; color: var(--text); line-height: 1.3; margin: 0.55rem 0 0.2rem; }
.card .meta { color: var(--faint); font-size: 0.8rem; }
.card .genres { color: var(--muted); font-size: 0.8rem; margin-top: 0.55rem; line-height: 1.45; }
.card-top { display: flex; justify-content: space-between; align-items: center; }
.pill { font-size: 0.7rem; font-weight: 600; padding: 0.18rem 0.6rem; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.score { font-size: 0.8rem; font-weight: 700; color: var(--gold); }
.bar { height: 3px; border-radius: 3px; background: rgba(255, 255, 255, 0.06); margin-top: 0.85rem; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 3px; background: var(--grad); animation: grow 1.1s var(--ease) both; }

.focus { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; border-left: 3px solid var(--violet); }
.focus .title { margin: 0; font-size: 1.15rem; }

/* ---------- Prediction ---------- */
.result { text-align: center; padding: 1.8rem 1.2rem; animation: pop 0.55s var(--ease) both; }
.result .big {
  font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; margin: 0.35rem 0;
  background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;
}
.result .sub { color: var(--muted); font-size: 0.92rem; }
.prob { display: grid; grid-template-columns: 4.5rem 1fr 3.2rem; align-items: center; gap: 0.7rem; margin: 0.55rem 0; font-size: 0.85rem; color: var(--muted); }
.prob .bar { margin: 0; height: 6px; }
.prob b { color: var(--text); font-weight: 600; text-align: right; }
.placeholder { color: var(--faint); text-align: center; padding: 3rem 1rem; border: 1px dashed var(--line); border-radius: 16px; }

/* ---------- Model stats strip ---------- */
.strip { display: flex; gap: 2.2rem; flex-wrap: wrap; justify-content: center; margin-top: 1.4rem; padding-top: 1.2rem; border-top: 1px solid var(--line); }
.strip div { text-align: center; }
.strip b { display: block; font-size: 1.35rem; font-weight: 700; color: var(--text); }
.strip span { font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--faint); }

/* ---------- Native widgets ---------- */
div[data-testid="stForm"] { background: var(--surface); border: 1px solid var(--line) !important; border-radius: 16px; padding: 1.3rem; backdrop-filter: blur(12px); }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: rgba(11, 9, 23, 0.6) !important; border-color: var(--line) !important; border-radius: 10px !important; transition: border-color 0.25s var(--ease); }
div[data-baseweb="select"] > div:hover { border-color: var(--line-strong) !important; }
span[data-baseweb="tag"] { background: rgba(167, 139, 250, 0.18) !important; color: var(--text) !important; border-radius: 8px !important; }
div[data-testid="stSlider"] div[role="slider"] { box-shadow: 0 0 0 4px rgba(167, 139, 250, 0.2) !important; }
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  width: 100%; background: var(--grad) !important; background-size: 200% auto !important; color: #140F26 !important;
  font-weight: 700 !important; border: none !important; border-radius: 12px !important; padding: 0.6rem 1rem !important;
  transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease), background-position 0.6s var(--ease) !important;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-2px); background-position: right center !important; box-shadow: 0 10px 28px -10px rgba(251, 113, 133, 0.7);
}
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
label, .stRadio label p, div[data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-weight: 500 !important; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


def render(markup: str) -> None:
    """Render HTML without leading indentation or blank lines, which Markdown would turn into code blocks."""
    lines = (line.strip() for line in markup.splitlines())
    st.markdown("\n".join(line for line in lines if line), unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def section(title: str, subtitle: str) -> None:
    render(f'<div class="section"><h2>{esc(title)}</h2><p>{esc(subtitle)}</p></div>')


def prob_bars(probabilities: dict, limit: int = 5) -> str:
    rows = []
    for i, (name, p) in enumerate(list(probabilities.items())[:limit]):
        rows.append(
            f'<div class="prob"><span>{esc(name)}</span>'
            f'<div class="bar"><span style="width:{p * 100:.1f}%; animation-delay:{0.15 + i * 0.08:.2f}s"></span></div>'
            f'<b>{p * 100:.1f}%</b></div>'
        )
    return "".join(rows)


def stats_strip(items: list[tuple[str, float]]) -> None:
    cells = "".join(f"<div><b>{v * 100:.1f}%</b><span>{esc(k)}</span></div>" for k, v in items)
    render(f'<div class="strip">{cells}</div>')


# ==============================================================================
# Cached resources
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    return get_preprocessed_data()


@st.cache_resource(show_spinner=False)
def get_recommender(df: pd.DataFrame) -> MovieRecommender:
    return build_recommender(df)


@st.cache_resource(show_spinner=False)
def get_type_model() -> ContentTypeClassifier:
    return train_content_type_model("rf")


@st.cache_resource(show_spinner=False)
def get_rating_model() -> AudienceRatingClassifier:
    return train_rating_classifier("rf")


@st.cache_resource(show_spinner=False)
def get_clusterer(df: pd.DataFrame, k: int) -> MovieClusterer:
    return build_clusterer(df, n_clusters=k)


def unique_genres(series: pd.Series) -> list[str]:
    return sorted({g.strip() for s in series.dropna() for g in s.split(",") if g.strip()})


df = load_catalog()
movie_count = int((df["type"] == "Movie").sum())
show_count = int((df["type"] == "TV Show").sum())


# ==============================================================================
# Hero & navigation
# ==============================================================================

render(f"""
<div class="hero">
  <div class="wordmark">Movie ML Analytics</div>
  <div class="tagline">Discover, predict, and map a catalog of films and series with machine learning.</div>
  <div class="stats">
    <span><span class="live"></span><b>{len(df):,}</b> titles</span>
    <span><b>{movie_count:,}</b> movies</span>
    <span><b>{show_count:,}</b> series</span>
  </div>
</div>
""")

PAGES = ["Discover", "Format", "Rating", "Segments"]
_, nav_col, _ = st.columns([1, 6, 1])
with nav_col:
    page = st.segmented_control(
        "Navigation", PAGES, default=PAGES[0], key="page", label_visibility="collapsed", width="stretch",
    ) or PAGES[0]


# ==============================================================================
# Discover: recommendations
# ==============================================================================

if page == "Discover":
    section("Find something similar", "Pick a title and get look-alikes ranked by shared genres, creators, origin, and rating.")

    with st.spinner("Indexing the catalog..."):
        recommender = get_recommender(df)

    titles = sorted(df["title"].unique().tolist())
    default = titles.index("Dick Johnson Is Dead") if "Dick Johnson Is Dead" in titles else 0

    c1, c2, c3 = st.columns([4, 1.4, 1.4])
    selected = c1.selectbox("Title", titles, index=default)
    fmt = c2.selectbox("Show", ["Everything", "Movies", "TV Shows"])
    top_k = c3.selectbox("Results", [3, 6, 9], index=1)
    type_filter = {"Movies": "Movie", "TV Shows": "TV Show"}.get(fmt)

    current = df[df["title"] == selected].iloc[0]
    render(f"""
    <div class="card focus">
      <div>
        <div class="title">{esc(current['title'])}</div>
        <div class="meta">{esc(current['listed_in'])} · {esc(current['director'])} · {esc(current['country'])}</div>
      </div>
      <div style="display:flex; gap:6px;">
        <span class="pill">{esc(current['type'])}</span>
        <span class="pill">{esc(current['rating'])}</span>
        <span class="pill">{esc(current['release_year'])}</span>
      </div>
    </div>
    """)

    recs = recommender.get_recommendations(selected, top_n=top_k, content_type_filter=type_filter)
    render('<div class="label">You might also like</div>')

    cols = st.columns(3)
    for i, row in recs.iterrows():
        score = float(row["similarity_score"])
        with cols[i % 3]:
            render(f"""
            <div class="card" style="animation-delay:{i * 0.07:.2f}s">
              <div class="card-top">
                <span class="pill">{esc(row['type'])}</span>
                <span class="score">{score * 100:.0f}% match</span>
              </div>
              <div class="title">{esc(row['title'])}</div>
              <div class="meta">{esc(row['release_year'])} · {esc(row['duration'])} · {esc(row['rating'])}</div>
              <div class="genres">{esc(row['listed_in'])}</div>
              <div class="bar"><span style="width:{score * 100:.0f}%; animation-delay:{0.2 + i * 0.07:.2f}s"></span></div>
            </div>
            """)


# ==============================================================================
# Format: Movie vs TV Show
# ==============================================================================

elif page == "Format":
    section("Movie or series?", "Predict a title's format from its genres, rating, and release year.")

    with st.spinner("Training the format model..."):
        type_clf = get_type_model()

    left, right = st.columns(2, gap="large")
    with left:
        with st.form("type_form", border=False):
            genres = unique_genres(df["genres_neutral"])
            chosen = st.multiselect("Genres", genres, default=["Documentary"] if "Documentary" in genres else genres[:1])
            rating = st.selectbox("Rating", ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y", "PG", "TV-G", "NR"])
            year = st.slider("Release year", 1960, 2026, 2021)
            go_type = st.form_submit_button("Predict format")

    with right:
        if go_type:
            res = type_clf.predict({
                "listed_in": ", ".join(chosen) or "Drama",
                "rating": rating,
                "release_year": year,
            })
            render(f"""
            <div class="card result">
              <div class="label" style="margin:0">Predicted format</div>
              <div class="big">{esc(res['prediction'])}</div>
              <div class="sub">{res['confidence'] * 100:.1f}% confident</div>
            </div>
            <div class="card">{prob_bars(res['probabilities'])}</div>
            """)
        else:
            render('<div class="placeholder">Describe a title and hit <b>Predict format</b>.</div>')

    m = type_clf.metrics
    stats_strip([("Accuracy", m["accuracy"]), ("Precision", m["precision"]), ("Recall", m["recall"]), ("F1", m["f1_score"])])


# ==============================================================================
# Rating: audience certificate
# ==============================================================================

elif page == "Rating":
    section("Who is it for?", "Estimate the maturity rating from format, genres, length, and year.")

    with st.spinner("Training the rating model..."):
        rating_clf = get_rating_model()

    RATING_NOTES = {
        "TV-MA": "Mature audiences only",
        "R": "Restricted: under 17 needs an adult",
        "TV-14": "Parents strongly cautioned (14+)",
        "PG-13": "Parents strongly cautioned (13+)",
        "TV-PG": "Parental guidance suggested",
        "PG": "Parental guidance suggested",
        "TV-Y7": "Suitable for children 7+",
        "TV-Y": "Suitable for all children",
        "TV-G": "General audience",
    }

    left, right = st.columns(2, gap="large")
    with left:
        fmt = st.radio("Format", ["Movie", "TV Show"], horizontal=True)
        with st.form("rating_form", border=False):
            genres = unique_genres(df.loc[df["type"] == fmt, "listed_in"])
            preferred = ["Children & Family Movies", "Comedies"] if fmt == "Movie" else ["Kids' TV"]
            chosen = st.multiselect("Genres", genres, default=[g for g in preferred if g in genres] or genres[:1])
            if fmt == "Movie":
                length = st.slider("Runtime (minutes)", 40, 220, 95, step=5)
                unit = "min"
            else:
                length = st.slider("Seasons", 1, 12, 1)
                unit = "season"
            year = st.slider("Release year", 1960, 2026, 2021)
            go_rating = st.form_submit_button("Predict rating")

    with right:
        if go_rating:
            res = rating_clf.predict({
                "listed_in": ", ".join(chosen) or "Dramas",
                "type": fmt,
                "duration_num": length,
                "duration_unit": unit,
                "release_year": year,
            })
            render(f"""
            <div class="card result">
              <div class="label" style="margin:0">Predicted rating</div>
              <div class="big">{esc(res['prediction'])}</div>
              <div class="sub">{esc(RATING_NOTES.get(res['prediction'], ''))} · {res['confidence'] * 100:.1f}% confident</div>
            </div>
            <div class="card">{prob_bars(res['probabilities'], limit=4)}</div>
            """)
        else:
            render('<div class="placeholder">Set the details and hit <b>Predict rating</b>.</div>')

    m = rating_clf.metrics
    stats_strip([("Accuracy", m["accuracy"]), ("Weighted F1", m["weighted_f1"]), ("Macro F1", m["macro_f1"])])


# ==============================================================================
# Segments: clustering
# ==============================================================================

else:
    section("Map the catalog", "K-Means groups similar titles; PCA flattens them into a map you can explore.")

    c1, c2 = st.columns([3, 2])
    k = c1.slider("Number of segments", 3, 7, 3, help="K = 3 scores the best silhouette on this catalog.")
    view = c2.segmented_control("View", ["2D map", "3D space"], default="2D map", width="stretch") or "2D map"

    with st.spinner("Clustering titles..."):
        clusterer = get_clusterer(df, k)

    fig = clusterer.create_2d_plot() if view == "2D map" else clusterer.create_3d_plot()
    fig.update_layout(height=520 if view == "2D map" else 580)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    render('<div class="label">Segment profiles</div>')
    summary = clusterer.get_cluster_summary()
    st.dataframe(
        summary[["cluster_label", "count", "dominant_type", "dominant_rating", "avg_release_year", "sample_titles"]],
        hide_index=True,
        width="stretch",
        column_config={
            "cluster_label": "Segment",
            "count": st.column_config.NumberColumn("Titles", format="%d"),
            "dominant_type": "Format",
            "dominant_rating": "Rating",
            "avg_release_year": st.column_config.NumberColumn("Avg. year", format="%.0f"),
            "sample_titles": "Examples",
        },
    )
