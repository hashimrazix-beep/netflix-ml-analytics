import re

from src.data_loader import NEUTRAL_GENRE_MAP, neutralize_genres, parse_duration


def test_parse_duration():
    assert parse_duration("90 min") == (90, "min")
    assert parse_duration("2 Seasons") == (2, "season")
    assert parse_duration("") == (0, "Unknown")
    assert parse_duration(None) == (0, "Unknown")


def test_neutralize_genres_merges_format_variants():
    assert neutralize_genres("TV Dramas, Dramas") == "Drama"
    assert neutralize_genres("Docuseries") == neutralize_genres("Documentaries")
    assert neutralize_genres("TV Shows") == "Other"


def test_every_catalog_genre_is_mapped(catalog):
    raw = {g.strip() for s in catalog["listed_in"] for g in s.split(",")}
    assert raw <= set(NEUTRAL_GENRE_MAP)


def test_neutral_genres_do_not_name_the_format(catalog):
    format_words = re.compile(r"\b(TV|Movies?|Shows?|Series|Docuseries)\b")
    for genres in catalog["genres_neutral"].unique():
        assert not format_words.search(genres), genres
