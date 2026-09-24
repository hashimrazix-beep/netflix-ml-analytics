"""
Movie ML Analytics - Streamlit Dashboard
Recommendations, format & rating prediction, and catalog segmentation.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.data_loader import get_preprocessed_data, neutralize_genres
from src.recommender import MovieRecommender, build_recommender
from src.classifier import (
    ContentTypeClassifier,
    AudienceRatingClassifier,
    train_content_type_model,
    train_rating_classifier,
)
from src.clustering import MovieClusterer, build_clusterer
from src import site


# ==============================================================================
# Routing: ?page=<slug>; missing -> home, unknown -> custom 404
# ==============================================================================

PAGE = site.resolve_page(st.query_params.get("page"))
page = PAGE.label

st.set_page_config(
    page_title=site.page_title(PAGE),
    page_icon=str(PROJECT_ROOT / "static" / "favicon-32.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_URL = site.get_setting("APP_URL", st.secrets, site.DEFAULT_APP_URL).rstrip("/")
GA_ID = site.valid_ga_id(site.get_setting("GA_MEASUREMENT_ID", st.secrets))
STATIC_URL = "app/static/"


# ==============================================================================
# Theme: "Holst" - deep navy base, steel blue / sage / sand / cream accents
# ==============================================================================

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
  --navy: #0D1B2A;
  --ink: #1B263B;
  --steel: #415A77;
  --sage: #778D7A;
  --sand: #D4C4A8;
  --cream: #F4F1DE;
  --bg: var(--navy);
  --surface: rgba(27, 38, 59, 0.88);
  --surface-solid: var(--ink);
  --line: rgba(212, 196, 168, 0.12);
  --line-strong: rgba(212, 196, 168, 0.4);
  --text: var(--cream);
  --muted: var(--sand);
  --faint: #8E9DAF;
  --grad: linear-gradient(100deg, #778D7A 0%, #D4C4A8 55%, #F4F1DE 100%);
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
}

html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family: 'Outfit', system-ui, sans-serif !important;
}

/* ---------- Canvas & drifting aurora ---------- */
html, body { background: var(--bg) !important; }
.stApp, div[data-testid="stAppViewContainer"], section[data-testid="stMain"] { background: transparent !important; color: var(--text); }
#beams-bg { background: var(--bg); }
iframe[title="beams_background"] { display: none; }
div[data-testid="stElementContainer"]:has(iframe[height="0"]) { display: none; }

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
@keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(119, 141, 122, 0.6); } 50% { box-shadow: 0 0 0 6px rgba(119, 141, 122, 0); } }

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
.live { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--sage); margin-right: 6px; vertical-align: middle; animation: pulse 2.4s infinite; }

/* ---------- Navigation (segmented control) ---------- */
div[data-testid="stButtonGroup"] > div {
  background: var(--surface); border: 1px solid var(--line); border-radius: 999px; padding: 5px; gap: 4px;
 
}
div[data-testid="stButtonGroup"] button {
  border: none !important; border-radius: 999px !important; background: transparent !important;
  color: var(--muted) !important; padding: 0.45rem 1.15rem !important; font-weight: 500 !important;
  transition: color 0.25s var(--ease), background 0.35s var(--ease), transform 0.25s var(--ease) !important;
}
div[data-testid="stButtonGroup"] button:hover { color: var(--text) !important; transform: translateY(-1px); }
div[data-testid="stButtonGroup"] button[aria-checked="true"],
div[data-testid="stButtonGroup"] button[kind$="Active"] {
  background: var(--grad) !important; background-size: 200% auto !important; color: var(--navy) !important; font-weight: 700 !important;
  box-shadow: 0 6px 24px -6px rgba(212, 196, 168, 0.45);
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
  padding: 1.1rem 1.2rem; margin-bottom: 1rem;
  animation: rise 0.7s var(--ease) backwards;
  transition: transform 0.35s var(--ease), border-color 0.35s var(--ease), box-shadow 0.35s var(--ease);
}
.card:hover { border-color: var(--line-strong); box-shadow: 0 18px 40px -18px rgba(212, 196, 168, 0.3); }
.card .title { font-weight: 600; font-size: 1.02rem; color: var(--text); line-height: 1.3; margin: 0.55rem 0 0.2rem; }
.card .meta { color: var(--faint); font-size: 0.8rem; }
.card .genres { color: var(--muted); font-size: 0.8rem; margin-top: 0.55rem; line-height: 1.45; }
.card-top { display: flex; justify-content: space-between; align-items: center; }
.rec-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
@media (max-width: 900px) { .rec-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 560px) { .rec-grid { grid-template-columns: 1fr; } }
.card.rec { margin: 0; }
.rec-head { display: flex; justify-content: space-between; align-items: baseline; gap: 0.75rem; }
.card.rec .title { margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.tags { display: flex; gap: 0.4rem; flex-wrap: nowrap; overflow: hidden; margin-top: 0.2rem; }
.tags .pill { white-space: nowrap; color: var(--sand); }
.rec-grid { align-items: start; }
.card.rec summary { list-style: none; cursor: pointer; display: flex; flex-direction: column; gap: 0.45rem; outline: none; }
.card.rec summary::-webkit-details-marker { display: none; }
.more { margin-left: auto; font-size: 0.72rem; color: var(--faint); white-space: nowrap; align-self: center; }
.more::after { content: " ▾"; display: inline-block; transition: transform 0.3s var(--ease); }
details[open] .more::after { transform: rotate(180deg); }
details[open].card.rec { border-color: var(--line-strong); }
.synopsis { margin: 0.8rem 0 0; padding-top: 0.8rem; border-top: 1px solid var(--line); color: var(--sand); font-size: 0.86rem; line-height: 1.55; animation: rise 0.45s var(--ease) both; }
.pill { font-size: 0.7rem; font-weight: 600; padding: 0.18rem 0.6rem; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.score { font-size: 0.8rem; font-weight: 700; color: var(--sand); }
.bar { height: 3px; border-radius: 3px; background: rgba(255, 255, 255, 0.06); margin-top: 0.85rem; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 3px; background: var(--grad); animation: grow 1.1s var(--ease) both; }

.focus { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; border-left: 3px solid var(--sage); }
.focus .title { margin: 0; font-size: 1.15rem; }

/* ---------- Prediction ---------- */
.result { text-align: center; padding: 1.8rem 1.2rem; animation: pop 0.55s var(--ease) backwards; }
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
div[data-testid="stForm"] { background: var(--surface); border: 1px solid var(--line) !important; border-radius: 16px; padding: 1.3rem; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: rgba(13, 27, 42, 0.6) !important; border-color: var(--line) !important; border-radius: 10px !important; transition: border-color 0.25s var(--ease); }
div[data-baseweb="select"] > div:hover { border-color: var(--line-strong) !important; }
span[data-baseweb="tag"] { background: rgba(119, 141, 122, 0.35) !important; color: var(--text) !important; border-radius: 8px !important; }
div[data-testid="stSlider"] div[role="slider"] { box-shadow: 0 0 0 4px rgba(212, 196, 168, 0.25) !important; }
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  width: 100%; background: var(--grad) !important; background-size: 200% auto !important; color: var(--navy) !important;
  font-weight: 700 !important; border: none !important; border-radius: 12px !important; padding: 0.6rem 1rem !important;
  transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease), background-position 0.6s var(--ease) !important;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-2px); background-position: right center !important; box-shadow: 0 10px 28px -10px rgba(212, 196, 168, 0.55);
}
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
label, .stRadio label p, div[data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-weight: 500 !important; }

/* ---------- 1. Loaders ---------- */
div[data-testid="stElementContainer"]:has(#intro-loader),
div[data-testid="stElementContainer"]:has(.mobile-cta) { animation: none !important; transform: none !important; height: 0; margin: 0; }
#intro-loader {
  position: fixed; inset: 0; z-index: 9999; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1.1rem;
  background: var(--bg); pointer-events: none;
  animation: loader-out 0.7s var(--ease) 1.1s forwards;
}
@keyframes loader-out { to { opacity: 0; visibility: hidden; } }
.loader-ring {
  width: 46px; height: 46px; border-radius: 50%;
  background: conic-gradient(from 0deg, transparent 0 25%, var(--sage) 50%, var(--sand) 75%, var(--cream));
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 4px), #000 calc(100% - 3px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 4px), #000 calc(100% - 3px));
  animation: spin 0.9s linear infinite;
}
.loader-text { color: var(--sand); font-size: 0.8rem; letter-spacing: 0.25em; text-transform: uppercase; animation: breathe 1.6s ease-in-out infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes breathe { 50% { opacity: 0.45; } }
div[data-testid="stSpinner"] > div { color: var(--sand) !important; }
div[data-testid="stSpinner"] i, div[data-testid="stSpinner"] svg { border-color: var(--sage) transparent transparent transparent !important; color: var(--sage) !important; }

/* ---------- 2. 3D tilt + glare ---------- */
.card {
  --rx: 0deg; --ry: 0deg; --lift: 0px; --mx: 50%; --my: 50%;
  transform: perspective(900px) rotateX(var(--rx)) rotateY(var(--ry)) translateY(var(--lift));
  transform-style: preserve-3d; will-change: transform;
}
.card:hover { --lift: -4px; transform: perspective(900px) rotateX(var(--rx)) rotateY(var(--ry)) translateY(var(--lift)); }
.card::after {
  content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none; opacity: 0;
  background: radial-gradient(circle at var(--mx) var(--my), rgba(244, 241, 222, 0.10), transparent 55%);
  transition: opacity 0.35s var(--ease);
}
.card:hover::after { opacity: 1; }
.card .title, .card .score { transform: translateZ(18px); }

/* ---------- 3. Scroll reveal ---------- */
.reveal { opacity: 0; translate: 0 18px; transition: opacity 0.7s var(--ease), translate 0.7s var(--ease); }
.reveal.revealed { opacity: 1; translate: 0 0; }

/* ---------- 4. Micro-interactions ---------- */
.has-ripple { position: relative; overflow: hidden; }
.ripple {
  position: absolute; border-radius: 50%; pointer-events: none; transform: scale(0);
  background: rgba(244, 241, 222, 0.28); animation: ripple 0.6s var(--ease) forwards;
}
@keyframes ripple { to { transform: scale(1); opacity: 0; } }
.stButton > button:active, div[data-testid="stFormSubmitButton"] > button:active,
div[data-testid="stButtonGroup"] button:active { transform: scale(0.96) !important; }
.pill { transition: background 0.25s var(--ease), color 0.25s var(--ease), border-color 0.25s var(--ease); }
.card:hover .pill { border-color: var(--line-strong); }
.pill:hover { background: rgba(119, 141, 122, 0.3); color: var(--cream); }
.score { transition: transform 0.3s var(--ease); display: inline-block; }
.card:hover .score { transform: translateZ(18px) scale(1.12); }
.more { transition: color 0.25s var(--ease); }
.card:hover .more { color: var(--sand); }
.strip div { transition: transform 0.3s var(--ease); }
.strip div:hover { transform: translateY(-3px); }
.strip div:hover b { color: var(--sand); }

/* ---------- 5. Parallax ---------- */
.hero { will-change: transform, opacity; }
#beams-bg { transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1); will-change: transform; }

/* ---------- CTA, breadcrumbs, footer ---------- */
.cta-row { display: flex; gap: 0.7rem; justify-content: center; flex-wrap: wrap; margin-top: 1.2rem; }
a.cta {
  display: inline-flex; align-items: center; justify-content: center; min-height: 44px; padding: 0.6rem 1.3rem;
  border-radius: 999px; font-weight: 600; font-size: 0.95rem; text-decoration: none !important;
  transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease), background-position 0.6s var(--ease), border-color 0.25s var(--ease);
}
a.cta.primary { background: var(--grad); background-size: 200% auto; color: var(--navy) !important; box-shadow: 0 8px 26px -12px rgba(212, 196, 168, 0.6); }
a.cta.primary:hover { transform: translateY(-2px); background-position: right center; }
a.cta.ghost { color: var(--sand) !important; border: 1px solid var(--line-strong); }
a.cta.ghost:hover { border-color: var(--sand); transform: translateY(-2px); }
a.cta:active { transform: scale(0.97); }
a.cta:focus-visible, .site-footer a:focus-visible, .breadcrumbs a:focus-visible, summary:focus-visible { outline: 2px solid var(--sand); outline-offset: 3px; }
.hero.compact { margin-bottom: 1rem; }
.hero.compact a.wordmark { display: inline-block; font-size: clamp(1.6rem, 4vw, 2.2rem); text-decoration: none; color: transparent !important; }
.breadcrumbs { display: flex; gap: 0.5rem; align-items: center; font-size: 0.8rem; color: var(--faint); margin: 1.2rem 0 -0.8rem; }
.breadcrumbs a { color: var(--sand) !important; text-decoration: none; }
.breadcrumbs a:hover { text-decoration: underline; }
.breadcrumbs .sep { opacity: 0.5; }
.site-footer { margin-top: 3.5rem; padding: 1.6rem 0 1rem; border-top: 1px solid var(--line); text-align: center; }
.site-footer nav { display: flex; flex-wrap: wrap; justify-content: center; gap: 0.4rem 1.3rem; }
.site-footer a { color: var(--muted) !important; font-size: 0.85rem; text-decoration: none; }
.site-footer a:hover { color: var(--cream) !important; }
.site-footer p { color: var(--faint); font-size: 0.76rem; margin: 0.9rem 0 0; }
.inline-links { color: var(--muted); font-size: 0.9rem; margin-top: 1rem; }
.inline-links a, .prose a { color: var(--sand) !important; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); border: 0; }

/* ---------- FAQ, legal, status pages ---------- */
.faq { display: grid; gap: 0.8rem; }
.faq-item { margin: 0; }
.faq-item summary { list-style: none; cursor: pointer; display: flex; justify-content: space-between; align-items: center; gap: 1rem; font-weight: 600; }
.faq-item summary::-webkit-details-marker { display: none; }
.faq-item .more { margin-left: 0; }
.prose { max-width: 760px; }
.prose h3 { color: var(--cream); font-size: 1.05rem; font-weight: 600; margin: 1.6rem 0 0.4rem; padding: 0; }
.prose p { color: var(--sand); line-height: 1.7; font-size: 0.95rem; margin: 0; }
.status-page { max-width: 620px; margin: 2rem auto; padding: 2.6rem 1.5rem; }

/* ---------- Cookie banner ---------- */
#cookie-banner {
  position: fixed; left: 50%; bottom: 1rem; transform: translateX(-50%); z-index: 1000; width: min(640px, calc(100% - 2rem));
  display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; justify-content: space-between;
  background: var(--surface-solid); border: 1px solid var(--line-strong); border-radius: 16px; padding: 0.9rem 1.1rem;
  box-shadow: 0 20px 50px -20px rgba(0, 0, 0, 0.8); animation: rise 0.5s var(--ease) both; transition: opacity 0.35s, transform 0.35s;
}
#cookie-banner.hide { opacity: 0; transform: translate(-50%, 12px); }
#cookie-banner p { margin: 0; color: var(--sand); font-size: 0.86rem; flex: 1 1 260px; }
#cookie-banner a { color: var(--cream); }
.cookie-actions { display: flex; gap: 0.5rem; }
#cookie-banner button {
  min-height: 40px; padding: 0.45rem 1rem; border-radius: 999px; font-weight: 600; cursor: pointer;
  border: 1px solid var(--line-strong); background: transparent; color: var(--sand); transition: transform 0.2s var(--ease);
}
#cookie-banner button.primary { background: var(--sand); color: var(--navy); border-color: var(--sand); }
#cookie-banner button:active { transform: scale(0.96); }

/* ---------- Sticky mobile CTA ---------- */
.mobile-cta { display: none; }

/* ---------- Breakpoints ---------- */
@media (max-width: 900px) {
  .block-container { padding-left: 1.2rem !important; padding-right: 1.2rem !important; }
  .strip { gap: 1.4rem; }
}
@media (max-width: 640px) {
  .block-container { padding-top: 1.4rem !important; padding-bottom: 5.5rem !important; }
  .tagline { font-size: 0.95rem; }
  .stats { gap: 0.9rem; }
  .section h2 { font-size: 1.3rem; }
  .result .big { font-size: 2.1rem; }
  .focus { flex-direction: column; align-items: flex-start; }
  div[data-testid="stButtonGroup"] button { padding: 0.45rem 0.6rem !important; }
  .mobile-cta {
    display: block; position: fixed; left: 0; right: 0; bottom: 0; z-index: 900; padding: 0.7rem 1rem calc(0.7rem + env(safe-area-inset-bottom));
    background: linear-gradient(to top, var(--navy) 60%, rgba(13, 27, 42, 0)); animation: rise 0.5s var(--ease) both;
  }
  .mobile-cta a.cta { width: 100%; }
  #cookie-banner { bottom: 5rem; }
}
@media (max-width: 380px) {
  div[data-testid="stButtonGroup"] button p { font-size: 0.8rem !important; }
}

/* ---------- Cross-browser fallbacks ---------- */
html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
* { -webkit-tap-highlight-color: transparent; }
@supports not (translate: 0 0) { .reveal { opacity: 1; } }
@supports not (background-clip: text) { @supports not (-webkit-background-clip: text) { .wordmark, .result .big { color: var(--sand); background: none; } } }

@media (prefers-reduced-motion: reduce) {
  #cookie-banner, .mobile-cta { animation: none !important; }
  #intro-loader { display: none !important; }
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(
    '<div id="intro-loader" role="status" aria-label="Loading"><div class="loader-ring" aria-hidden="true"></div><div class="loader-text">Movie ML Analytics</div></div>',
    unsafe_allow_html=True,
)
# Animated beams background (runs once, attaches a canvas to the page)
def _json_ld() -> list[dict]:
    crumbs = [{"@type": "ListItem", "position": 1, "name": "Home", "item": APP_URL + "/"}]
    if PAGE.slug != site.HOME:
        crumbs.append({"@type": "ListItem", "position": 2, "name": PAGE.title, "item": site.page_url(APP_URL, PAGE)})
    data = [
        {"@context": "https://schema.org", "@type": "WebApplication", "name": site.SITE_NAME, "url": APP_URL + "/",
         "applicationCategory": "EntertainmentApplication", "operatingSystem": "Any",
         "description": site.PAGES[site.HOME].description, "offers": {"@type": "Offer", "price": "0"}},
        {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": crumbs},
    ]
    if PAGE.slug == "faq":
        data.append({"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in site.FAQ]})
    return data


_page_config = {
    "slug": PAGE.slug,
    "title": site.page_title(PAGE),
    "description": PAGE.description,
    "indexable": PAGE.indexable,
    "url": site.page_url(APP_URL, PAGE),
    "siteName": site.SITE_NAME,
    "static": f"{APP_URL}/{STATIC_URL}",
    "ogImage": f"{APP_URL}/{STATIC_URL}og-image.jpg",
    "ogImageAlt": "Movie ML Analytics: recommendations, predictions, and a map of 8,790 films and series.",
    "gaId": GA_ID,
    "jsonLd": _json_ld(),
}
# Escape "</" so page data can never close the script tag early
_config_js = "window.__PAGE__ = " + json.dumps(_page_config).replace("</", "<\\/") + ";"
_scripts = "".join((PROJECT_ROOT / "static" / f).read_text() for f in ("beams.js", "interactions.js", "site.js"))
components.html(f"<script>{_config_js}{_scripts}</script>", height=0)


def render(markup: str) -> None:
    """Render HTML without leading indentation or blank lines, which Markdown would turn into code blocks."""
    lines = (line.strip() for line in markup.splitlines())
    st.markdown("\n".join(line for line in lines if line), unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def section(title: str, subtitle: str, anchor: str = "") -> None:
    anchor_attr = f' id="{anchor}"' if anchor else ""
    render(f'<div class="section"{anchor_attr}><h2>{esc(title)}</h2><p>{esc(subtitle)}</p></div>')


def link(slug: str, text: str, cls: str = "") -> str:
    href = "./" if slug == site.HOME else f"?page={slug}"
    cls_attr = f' class="{cls}"' if cls else ""
    return f'<a href="{href}" target="_self"{cls_attr}>{esc(text)}</a>'


def rate_limited(action: str, limit: int, window: float = 60) -> bool:
    """True (and a notice is shown) when this session exceeds `limit` actions per `window` seconds."""
    if site.allow(st.session_state, action, limit, window):
        return False
    st.warning("You're going a little fast. Please wait a moment and try again.", icon="⏳")
    return True


def prose(sections: list[tuple[str, str]]) -> None:
    render('<div class="prose">' + "".join(f"<h3>{esc(h)}</h3><p>{esc(t)}</p>" for h, t in sections) + "</div>")


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


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def summarize(row, query) -> str:
    """Write a short summary of a title from its catalog metadata."""
    is_movie = row["type"] == "Movie"
    genres = [g.lower() for g in neutralize_genres(row["listed_in"]).split(", ") if g != "Other"]
    countries = [c.strip() for c in str(row["country"]).split(",") if c.strip() not in ("", "Unknown Country")]
    countries = [f"the {c}" if c.split()[0] in ("United", "Czech", "Dominican") else c for c in countries]
    directors = [d.strip() for d in str(row["director"]).split(",") if d.strip() != "Unknown Director"]

    what = f"{_join(genres[:3])} {'film' if is_movie else 'series'}" if genres else ("film" if is_movie else "series")
    text = f"A {row['release_year']} {what}"
    if countries:
        text += f" from {_join(countries[:2])}"
    if directors:
        text += f", {'directed' if is_movie else 'created'} by {_join(directors[:2])}"
    text += ". "

    length = f"Runs {row['duration']}" if is_movie else f"Spans {row['duration'].lower()}"
    note = RATING_NOTES.get(row["rating"])
    text += f"{length}, rated {row['rating']}" + (f" ({note.lower()})" if note else "") + "."
    added = pd.to_datetime(row.get("date_added"), errors="coerce")
    if pd.notna(added):
        text += f" Added to the catalog in {added:%B %Y}."

    shared = [g for g in neutralize_genres(row["listed_in"]).split(", ") if g in neutralize_genres(query["listed_in"]).split(", ")]
    shared += [d for d in directors if d in str(query["director"])]
    shared += [c for c in countries if c.removeprefix("the ") in str(query["country"])]
    if row["rating"] == query["rating"]:
        shared.append(row["rating"])
    if shared:
        text += f" Suggested because it shares {_join(shared[:4])} with {query['title']}."
    return text


def unique_genres(series: pd.Series) -> list[str]:
    return sorted({g.strip() for s in series.dropna() for g in s.split(",") if g.strip()})


df = load_catalog()
movie_count = int((df["type"] == "Movie").sum())
show_count = int((df["type"] == "TV Show").sum())


# ==============================================================================
# Hero & navigation
# ==============================================================================

# Global per-session request limit (every interaction is a rerun)
if rate_limited("rerun", 120):
    st.stop()

if PAGE.tool:
    render(f"""
    <div class="hero">
      <div class="wordmark">Movie ML Analytics</div>
      <div class="tagline">Discover, predict, and map a catalog of films and series with machine learning.</div>
      <div class="cta-row">
        {link("discover", "Find your next watch", "cta primary") if PAGE.slug != "discover" else '<a href="#find" class="cta primary">Find your next watch</a>'}
        {link("faq", "How it works", "cta ghost")}
      </div>
      <div class="stats">
        <span><span class="live"></span><b>{len(df):,}</b> titles</span>
        <span><b>{movie_count:,}</b> movies</span>
        <span><b>{show_count:,}</b> series</span>
      </div>
    </div>
    """)
else:
    render(f'<div class="hero compact">{link("discover", "Movie ML Analytics", "wordmark")}</div>')

TOOL_PAGES = [p for p in site.PAGES.values() if p.tool]
LABEL_TO_SLUG = {p.label: p.slug for p in TOOL_PAGES}


def _go_to_nav_page() -> None:
    label = st.session_state.get("nav")
    if label:
        st.query_params["page"] = LABEL_TO_SLUG[label]


_, nav_col, _ = st.columns([1, 6, 1])
with nav_col:
    st.segmented_control(
        "Navigation", [p.label for p in TOOL_PAGES], default=page if PAGE.tool else None,
        key="nav", on_change=_go_to_nav_page, label_visibility="collapsed", width="stretch",
    )

# Breadcrumbs
crumbs = [link("discover", "Home")]
if PAGE.slug != site.HOME:
    crumbs.append(f'<span aria-current="page">{esc(PAGE.title)}</span>')
else:
    crumbs.append('<span aria-current="page">Recommendations</span>')
render('<nav class="breadcrumbs" aria-label="Breadcrumb">' + '<span class="sep">/</span>'.join(crumbs) + "</nav>")


# ==============================================================================
# Discover: recommendations
# ==============================================================================

if page == "Discover":
    section("Find something similar", "Pick a title and get look-alikes ranked by shared genres, creators, origin, and rating.", anchor="find")

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

    cards = []
    for i, row in recs.iterrows():
        genres = neutralize_genres(row["listed_in"]).split(", ")[:2]
        tags = "".join(f'<span class="pill">{esc(g)}</span>' for g in genres)
        cards.append(f"""
        <details class="card rec" style="animation-delay:{i * 0.06:.2f}s">
          <summary>
            <div class="rec-head">
              <div class="title" title="{esc(row['title'])}">{esc(row['title'])}</div>
              <span class="score">{float(row['similarity_score']) * 100:.0f}%</span>
            </div>
            <div class="meta">{esc(row['type'])} · {esc(row['release_year'])} · {esc(row['duration'])} · {esc(row['rating'])}</div>
            <div class="tags">{tags}<span class="more">Summary</span></div>
          </summary>
          <p class="synopsis">{esc(summarize(row, current))}</p>
        </details>
        """)
    render('<div class="rec-grid">' + "".join(cards) + "</div>")


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
        if go_type and not rate_limited("predict", 20):
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
        if go_rating and not rate_limited("predict", 20):
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

elif page == "Segments":
    section("Map the catalog", "K-Means groups similar titles; PCA flattens them into a map you can explore.")

    c1, c2 = st.columns([3, 2])
    k = c1.slider("Number of segments", 3, 7, 3, help="K = 3 scores the best silhouette on this catalog.")
    view = c2.segmented_control("View", ["2D map", "3D space"], default="2D map", width="stretch") or "2D map"

    with st.spinner("Clustering titles..."):
        clusterer = get_clusterer(df, k)

    fig = clusterer.create_2d_plot() if view == "2D map" else clusterer.create_3d_plot()
    fig.update_layout(height=520 if view == "2D map" else 580)
    render(f'<p class="sr-only">Scatter plot of all {len(df):,} titles projected with PCA and coloured by segment; '
           'the table below summarises each segment.</p>')
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


# ==============================================================================
# FAQ (with feedback form -> thank-you page)
# ==============================================================================

elif page == "FAQ":
    section("Frequently asked questions", "How the tools work, where the data comes from, and how your data is handled.")
    render('<div class="faq">' + "".join(
        f'<details class="card faq-item"><summary><span>{esc(q)}</span><span class="more">Answer</span></summary>'
        f'<p class="synopsis">{esc(a)}</p></details>' for q, a in site.FAQ) + "</div>")
    render(f'<p class="inline-links">Still curious? Try the {link("discover", "recommender")}, '
           f'the {link("format", "format predictor")}, or read the {link("privacy", "privacy policy")}.</p>')

    render('<div class="label">Send feedback</div>')
    with st.form("feedback_form", border=False):
        topic = st.selectbox("Topic", site.FEEDBACK_TOPICS)
        message = st.text_area("Message", max_chars=site.MAX_FEEDBACK_CHARS,
                               placeholder="What worked, what didn't, what you'd like to see. Please don't include personal details.")
        sent = st.form_submit_button("Send feedback")
    if sent:
        if not message.strip():
            st.warning("Please write a message before sending.")
        elif not rate_limited("feedback", 3, 600):
            site.save_feedback(topic, message)
            st.query_params["page"] = "thanks"
            st.rerun()


# ==============================================================================
# Legal pages
# ==============================================================================

elif page == "Privacy":
    section("Privacy policy", f"Last updated {site.LAST_UPDATED}.")
    prose(site.PRIVACY)

elif page == "Terms":
    section("Terms of use", f"Last updated {site.LAST_UPDATED}.")
    prose(site.TERMS)


# ==============================================================================
# Thank-you page
# ==============================================================================

elif page == "Thank you":
    render(f"""
    <div class="card result status-page">
      <div class="big">Thank you</div>
      <div class="sub">Your feedback was received. It helps make these tools better.</div>
      <div class="cta-row">{link("discover", "Back to recommendations", "cta primary")}{link("faq", "Read the FAQ", "cta ghost")}</div>
    </div>
    """)


# ==============================================================================
# Custom 404
# ==============================================================================

else:
    render(f"""
    <div class="card result status-page">
      <div class="label" style="margin:0">Error 404</div>
      <div class="big">Lost in the credits</div>
      <div class="sub">We couldn't find that page. It may have moved, or the link may be mistyped.</div>
      <div class="cta-row">{link("discover", "Go to recommendations", "cta primary")}{link("faq", "Visit the FAQ", "cta ghost")}</div>
    </div>
    """)


# ==============================================================================
# Footer (internal links) and sticky mobile CTA
# ==============================================================================

_footer_links = "".join(link(p.slug, p.label if p.tool else p.title) for p in site.PAGES.values() if p.indexable)
render(f"""
<footer class="site-footer">
  <nav aria-label="Footer">{_footer_links}<a href="#" data-cookie-settings>Cookie settings</a>
  <a href="https://github.com/hashimrazix-beep/netflix-ml-analytics" target="_blank" rel="noopener">GitHub</a></nav>
  <p>© 2026 Hashim Razi · An independent educational project, not affiliated with any streaming service.</p>
</footer>
""")
if PAGE.slug != "discover":
    render(f'<div class="mobile-cta">{link("discover", "Find your next watch", "cta primary")}</div>')
