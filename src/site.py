"""
Site-level content and helpers for the Movie ML Analytics app:
page metadata (titles, descriptions, routes), FAQ and legal copy,
a per-session rate limiter, and feedback storage.
"""

from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import MutableMapping, Optional

SITE_NAME = "Movie ML Analytics"
DEFAULT_APP_URL = "https://movie-ml-analytics.streamlit.app"
LAST_UPDATED = "24 September 2026"
FEEDBACK_FILE = Path(__file__).resolve().parent.parent / "feedback" / "feedback.csv"
GA_ID_PATTERN = re.compile(r"^G-[A-Z0-9]{4,16}$")


@dataclass(frozen=True)
class Page:
    slug: str
    label: str
    title: str
    description: str
    tool: bool = False       # appears in the main navigation
    indexable: bool = True   # robots index/follow + listed in sitemap.xml


PAGES: dict[str, Page] = {p.slug: p for p in [
    Page("discover", "Discover", "Movie & Series Recommendations",
         "Pick any film or series and get similar titles ranked by shared genres, creators, origin, and rating.", tool=True),
    Page("format", "Format", "Movie or Series Predictor",
         "Predict whether a title is a movie or a TV series from its genres, rating, and release year.", tool=True),
    Page("rating", "Rating", "Maturity Rating Predictor",
         "Estimate a title's maturity rating from its format, genres, runtime, and release year.", tool=True),
    Page("segments", "Segments", "Catalog Segments Map",
         "Explore an interactive 2D and 3D map of 8,790 films and series grouped by K-Means clustering.", tool=True),
    Page("faq", "FAQ", "Frequently Asked Questions",
         "How the recommendations and predictions work, where the data comes from, and how your data is handled."),
    Page("privacy", "Privacy", "Privacy Policy",
         "What Movie ML Analytics collects, why, how long it is kept, and the choices you have."),
    Page("terms", "Terms", "Terms of Use",
         "The terms for using Movie ML Analytics, an educational machine learning project."),
    Page("thanks", "Thank you", "Thank You",
         "Thanks for your feedback on Movie ML Analytics.", indexable=False),
]}
HOME = "discover"
NOT_FOUND = Page("404", "Not found", "Page Not Found",
                 "The page you were looking for does not exist.", indexable=False)


def resolve_page(slug: Optional[str]) -> Page:
    """Map a ?page= value to a Page; missing means home, unknown means 404."""
    if slug is None or slug == "":
        return PAGES[HOME]
    return PAGES.get(str(slug).strip().lower(), NOT_FOUND)


def page_title(page: Page) -> str:
    return f"{page.title} | {SITE_NAME}"


def page_url(app_url: str, page: Page) -> str:
    base = app_url.rstrip("/")
    return base + "/" if page.slug == HOME else f"{base}/?page={page.slug}"


def get_setting(name: str, secrets: Optional[MutableMapping] = None, default: str = "") -> str:
    """Read a setting from Streamlit secrets first, then the environment."""
    try:
        if secrets is not None and name in secrets:
            return str(secrets[name])
    except Exception:  # no secrets file configured
        pass
    return os.environ.get(name, default)


def valid_ga_id(value: str) -> str:
    """Return the GA4 measurement ID if well-formed, else an empty string."""
    value = (value or "").strip()
    return value if GA_ID_PATTERN.match(value) else ""


# ==============================================================================
# Rate limiting (per browser session, sliding window)
# ==============================================================================

def allow(state: MutableMapping, action: str, limit: int, window: float = 60.0,
          now: Optional[float] = None) -> bool:
    """Record an attempt and return False once `limit` attempts fall within `window` seconds."""
    now = time.time() if now is None else now
    key = f"_rate_{action}"
    recent = [t for t in state.get(key, []) if now - t < window]
    if len(recent) >= limit:
        state[key] = recent
        return False
    recent.append(now)
    state[key] = recent
    return True


# ==============================================================================
# Feedback
# ==============================================================================

FEEDBACK_TOPICS = ["General feedback", "Recommendations", "Predictions", "Bug report", "Other"]
MAX_FEEDBACK_CHARS = 1000


def save_feedback(topic: str, message: str, path: Path = FEEDBACK_FILE) -> None:
    """Append a feedback message (no personal data) to a CSV file."""
    message = message.strip()[:MAX_FEEDBACK_CHARS]
    if not message:
        raise ValueError("Message is empty.")
    if topic not in FEEDBACK_TOPICS:
        topic = "Other"
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["received_utc", "topic", "message"])
        writer.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), topic, message])


# ==============================================================================
# Copy
# ==============================================================================

FAQ = [
    ("What is Movie ML Analytics?",
     "An educational project that applies machine learning to a catalog of 8,790 films and series. "
     "You can get recommendations, predict a title's format or maturity rating, and explore a map of the catalog."),
    ("How are recommendations chosen?",
     "Each title is described by its genres, directors, countries, and rating. Those are turned into TF-IDF vectors, "
     "and the titles whose vectors point in the most similar direction (cosine similarity) are suggested."),
    ("How accurate are the predictions?",
     "The format predictor is right about 84% of the time on held-out titles, against a 70% baseline from always "
     "guessing \"movie\". The maturity-rating predictor chooses between nine ratings and is right about 47% of the time."),
    ("Where does the data come from?",
     "From a public catalog dataset listing each title's type, director, country, release year, rating, duration, "
     "and genres. It contains no plot descriptions, which is why summaries are built from those details."),
    ("Why is the first load slow?",
     "The models are trained the first time each page is opened after the app starts, which takes a few seconds. "
     "After that they are cached and pages respond instantly."),
    ("Do you store my data?",
     "No account is needed and nothing you type into the tools is stored. Feedback you send is saved without your "
     "name or email. Analytics cookies are only set if you accept them. See the privacy policy for details."),
    ("Is this affiliated with a streaming service?",
     "No. This is an independent project and is not affiliated with, endorsed by, or sponsored by any streaming "
     "service or studio."),
]

PRIVACY = [
    ("Who we are",
     "Movie ML Analytics is an independent, non-commercial educational project maintained by Hashim Razi. "
     "Questions about this policy can be raised through the project's GitHub repository."),
    ("What we collect",
     "Tool inputs (titles, genres, sliders) are processed in memory to produce results and are not stored. "
     "If you send feedback, we store the topic, your message, and the time it was received, with no name, email, "
     "or IP address. If you accept analytics cookies, Google Analytics 4 collects usage data such as pages viewed, "
     "approximate location, device, and browser."),
    ("Cookies and local storage",
     "Your cookie choice is remembered in your browser's local storage. Google Analytics cookies (for example _ga) "
     "are set only after you choose Accept, and you can change your choice at any time with the Cookie settings "
     "link in the footer."),
    ("Why we collect it",
     "Feedback helps us fix problems and improve the tools. Analytics, when allowed, shows which pages are useful. "
     "We do not sell data or use it for advertising."),
    ("Hosting and third parties",
     "The app is hosted on Streamlit Community Cloud, which may keep standard server logs under its own privacy "
     "policy. Analytics, when accepted, is processed by Google under its privacy policy."),
    ("Retention",
     "Feedback is kept for up to 12 months and may be lost sooner when the app is redeployed. Analytics data is "
     "kept for the Google Analytics default retention period of 2 months."),
    ("Your choices",
     "You can decline analytics, clear your browser storage, or ask for feedback you sent to be deleted by opening "
     "an issue on the project's GitHub repository with the date and topic of your message."),
]

TERMS = [
    ("About the service",
     "Movie ML Analytics is provided free of charge for educational and personal use, as is and without warranty. "
     "It may change or go offline at any time."),
    ("Predictions are estimates",
     "Recommendations, format predictions, maturity ratings, and segments are produced by statistical models and "
     "can be wrong. Do not rely on them for decisions about what is suitable for children or other audiences; "
     "check official ratings instead."),
    ("Fair use of the app",
     "Do not attempt to overload, scrape, or disrupt the app. Requests are rate-limited per session, and excessive "
     "use may be blocked."),
    ("Content and trademarks",
     "Titles and other catalog details come from a public dataset and belong to their respective owners. "
     "This project is not affiliated with any streaming service or studio, and any names mentioned are used only "
     "to identify titles."),
    ("Source code",
     "The source code is released under the MIT License and is available on GitHub."),
    ("Feedback",
     "By sending feedback you allow us to use it to improve the project. Do not include personal information."),
    ("Changes",
     "These terms may be updated. The date at the top shows when they last changed."),
]
