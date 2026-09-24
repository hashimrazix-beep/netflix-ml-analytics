import pytest

from src.classifier import AudienceRatingClassifier, ContentTypeClassifier
from src.clustering import build_clusterer


@pytest.mark.parametrize("model_type", ["rf", "lr"])
def test_content_type_classifier_has_no_leakage(catalog, model_type):
    clf = ContentTypeClassifier(model_type=model_type)
    metrics = clf.train(catalog)
    # Well above the 70% majority baseline, but not the ~100% that leaked labels produced
    assert 0.75 < metrics["accuracy"] < 0.97
    pred = clf.predict({"listed_in": "Documentaries", "rating": "PG-13", "release_year": 2020})
    assert pred["prediction"] in {"Movie", "TV Show"}
    assert sum(pred["probabilities"].values()) == pytest.approx(1, abs=1e-3)


@pytest.mark.parametrize("model_type", ["rf", "dt", "lr"])
def test_rating_classifier_trains_and_predicts(catalog, model_type):
    clf = AudienceRatingClassifier(model_type=model_type)
    metrics = clf.train(catalog)
    assert metrics["accuracy"] > 0.3
    pred = clf.predict({"listed_in": "Kids' TV", "type": "TV Show", "duration": "1 Season", "release_year": 2021})
    assert pred["prediction"] in AudienceRatingClassifier.TOP_RATINGS
    assert list(pred["probabilities"].values()) == sorted(pred["probabilities"].values(), reverse=True)


def test_clusterer(catalog):
    clusterer = build_clusterer(catalog, n_clusters=3)
    summary = clusterer.get_cluster_summary()
    assert len(summary) == 3
    assert summary["count"].sum() == len(catalog)
    assert clusterer.create_2d_plot().data
    assert clusterer.create_3d_plot().data
