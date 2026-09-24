"""
Data loading and preprocessing module for Movie ML Analytics.
Handles cleaning, missing value imputation, feature engineering, and data preparation.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np


DEFAULT_DATA_PATHS = [
    Path("data/Dataset.csv"),
    Path("Dataset.csv"),
    Path("../data/Dataset.csv"),
    Path(__file__).resolve().parent.parent / "data" / "Dataset.csv",
]


def find_data_file(custom_path: Optional[str] = None) -> Path:
    """Find the dataset file across standard project locations."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Dataset not found at provided path: {custom_path}")

    for candidate in DEFAULT_DATA_PATHS:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Dataset.csv not found in any standard locations: "
        + ", ".join(str(p) for p in DEFAULT_DATA_PATHS)
    )


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load the raw catalog dataset into a pandas DataFrame."""
    resolved_path = find_data_file(filepath)
    df = pd.read_csv(resolved_path)
    return df


def parse_duration(val: str) -> Tuple[int, str]:
    """
    Parse a duration string like '90 min' or '2 Seasons' into (number, unit).
    """
    if pd.isna(val) or not str(val).strip():
        return 0, "Unknown"
    val_str = str(val).strip()
    parts = val_str.split()
    try:
        num = int(parts[0])
        unit = "min" if "min" in parts[1].lower() else "season"
        return num, unit
    except (ValueError, IndexError):
        return 0, "Unknown"


# The catalog's genre labels are format-specific ("TV Dramas" vs "Dramas",
# "Docuseries" vs "Documentaries"), so using them raw to predict Movie vs
# TV Show leaks the target. This maps each label to a format-neutral genre.
NEUTRAL_GENRE_MAP = {
    "Action & Adventure": "Action & Adventure",
    "TV Action & Adventure": "Action & Adventure",
    "Anime Features": "Anime",
    "Anime Series": "Anime",
    "British TV Shows": "British",
    "Children & Family Movies": "Kids & Family",
    "Kids' TV": "Kids & Family",
    "Classic & Cult TV": "Classic & Cult",
    "Classic Movies": "Classic & Cult",
    "Cult Movies": "Classic & Cult",
    "Comedies": "Comedy",
    "TV Comedies": "Comedy",
    "Crime TV Shows": "Crime",
    "Documentaries": "Documentary",
    "Docuseries": "Documentary",
    "Dramas": "Drama",
    "TV Dramas": "Drama",
    "Faith & Spirituality": "Faith & Spirituality",
    "Horror Movies": "Horror",
    "TV Horror": "Horror",
    "Independent Movies": "Independent",
    "International Movies": "International",
    "International TV Shows": "International",
    "Korean TV Shows": "Korean",
    "LGBTQ Movies": "LGBTQ",
    "Music & Musicals": "Music & Musicals",
    "Reality TV": "Reality",
    "Romantic Movies": "Romance",
    "Romantic TV Shows": "Romance",
    "Sci-Fi & Fantasy": "Sci-Fi & Fantasy",
    "TV Sci-Fi & Fantasy": "Sci-Fi & Fantasy",
    "Science & Nature TV": "Science & Nature",
    "Spanish-Language TV Shows": "Spanish-Language",
    "Sports Movies": "Sports",
    "Stand-Up Comedy": "Stand-Up & Talk",
    "Stand-Up Comedy & Talk Shows": "Stand-Up & Talk",
    "Teen TV Shows": "Teen",
    "TV Mysteries": "Mystery",
    "Thrillers": "Thriller",
    "TV Thrillers": "Thriller",
    # Bare format labels carry no genre information.
    "Movies": None,
    "TV Shows": None,
}


def neutralize_genres(listed_in: str) -> str:
    """Map a comma-separated genre string to format-neutral genre labels."""
    if not isinstance(listed_in, str):
        return "Other"
    genres = []
    for raw in listed_in.split(","):
        label = raw.strip()
        neutral = NEUTRAL_GENRE_MAP.get(label, label)
        if neutral and neutral not in genres:
            genres.append(neutral)
    return ", ".join(genres) if genres else "Other"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features on the catalog dataset:
    - Handles 'Not Given' and NaN entries
    - Standardizes text columns
    - Extracts duration numeric value & unit
    - Extracts date_added year, month, and day
    - Maps genres to format-neutral labels ('genres_neutral')
    """
    cleaned = df.copy()

    # Normalize missing strings
    cleaned["director"] = cleaned["director"].replace(["Not Given", "not given", ""], np.nan)
    cleaned["country"] = cleaned["country"].replace(["Not Given", "not given", ""], np.nan)
    
    cleaned["director"] = cleaned["director"].fillna("Unknown Director")
    cleaned["country"] = cleaned["country"].fillna("Unknown Country")
    cleaned["rating"] = cleaned["rating"].fillna("Unknown")

    # Clean string fields
    for col in ["title", "director", "country", "listed_in", "type", "rating"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    # Parse duration
    parsed_duration = cleaned["duration"].apply(parse_duration)
    cleaned["duration_num"] = [p[0] for p in parsed_duration]
    cleaned["duration_unit"] = [p[1] for p in parsed_duration]

    # For movies, duration in minutes; for TV shows, seasons
    cleaned["duration_min"] = np.where(
        cleaned["duration_unit"] == "min", cleaned["duration_num"], np.nan
    )
    cleaned["duration_seasons"] = np.where(
        cleaned["duration_unit"] == "season", cleaned["duration_num"], np.nan
    )

    # Parse date_added
    if "date_added" in cleaned.columns:
        cleaned["date_added_dt"] = pd.to_datetime(cleaned["date_added"], errors="coerce")
        cleaned["year_added"] = cleaned["date_added_dt"].dt.year
        cleaned["month_added"] = cleaned["date_added_dt"].dt.month_name()
    else:
        cleaned["year_added"] = np.nan
        cleaned["month_added"] = "Unknown"

    # Fill year_added missing with release_year
    cleaned["year_added"] = cleaned["year_added"].fillna(cleaned["release_year"]).astype(int)

    # Primary genre (first listed in listed_in)
    cleaned["primary_genre"] = cleaned["listed_in"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) else "Unknown"
    )

    cleaned["genres_neutral"] = cleaned["listed_in"].apply(neutralize_genres)

    return cleaned


def get_preprocessed_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Convenience helper to load raw data and apply full cleaning pipeline."""
    raw = load_raw_data(filepath)
    return clean_data(raw)


if __name__ == "__main__":
    df = get_preprocessed_data()
    print("Preprocessed dataset successfully!")
    print(f"Shape: {df.shape}")
    print(f"Sample neutral genres: {df['genres_neutral'].iloc[1]}")
    print(f"Columns: {list(df.columns)}")
