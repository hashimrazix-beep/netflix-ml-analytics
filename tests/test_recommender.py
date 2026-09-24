import pytest

from src.recommender import build_recommender


@pytest.fixture(scope="module")
def recommender(catalog):
    return build_recommender(catalog)


def test_recommendations_exclude_query_and_are_ranked(recommender):
    recs = recommender.get_recommendations("Dick Johnson Is Dead", top_n=5)
    assert len(recs) == 5
    assert "Dick Johnson Is Dead" not in recs["title"].tolist()
    scores = recs["similarity_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_recommendations_share_genre(recommender):
    recs = recommender.get_recommendations("Dick Johnson Is Dead", top_n=5)
    assert all("Documentaries" in g for g in recs["listed_in"])


def test_type_filter(recommender):
    recs = recommender.get_recommendations("Midnight Mass", top_n=6, content_type_filter="Movie")
    assert set(recs["type"]) == {"Movie"}


def test_case_insensitive_and_missing_titles(recommender):
    assert recommender.find_title("midnight mass") == "Midnight Mass"
    with pytest.raises(KeyError):
        recommender.get_recommendations("zzz no such title zzz")
