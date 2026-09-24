import csv
import xml.etree.ElementTree as ET
from pathlib import Path

from src import site

STATIC = Path(__file__).resolve().parent.parent / "static"


def test_routing_home_known_and_404():
    assert site.resolve_page(None).slug == "discover"
    assert site.resolve_page("").slug == "discover"
    assert site.resolve_page("FAQ").slug == "faq"
    assert site.resolve_page("does-not-exist") is site.NOT_FOUND


def test_titles_and_descriptions_are_unique():
    pages = list(site.PAGES.values()) + [site.NOT_FOUND]
    titles = [site.page_title(p) for p in pages]
    descriptions = [p.description for p in pages]
    assert len(set(titles)) == len(titles)
    assert len(set(descriptions)) == len(descriptions)
    # Search-result-sized descriptions on every indexable page
    assert all(50 <= len(p.description) <= 160 for p in pages if p.indexable)


def test_rate_limiter_sliding_window():
    state = {}
    assert all(site.allow(state, "x", 3, window=60, now=t) for t in (0, 1, 2))
    assert not site.allow(state, "x", 3, window=60, now=3)
    assert site.allow(state, "x", 3, window=60, now=61)  # oldest attempt expired


def test_ga_id_validation():
    assert site.valid_ga_id(" G-ABC123XYZ ") == "G-ABC123XYZ"
    assert site.valid_ga_id("G-1'); alert(1)//") == ""
    assert site.valid_ga_id("") == ""


def test_save_feedback(tmp_path):
    path = tmp_path / "fb.csv"
    site.save_feedback("Bug report", "  hello  ", path=path)
    site.save_feedback("not-a-topic", "x" * 5000, path=path)
    rows = list(csv.reader(path.open()))
    assert rows[0] == ["received_utc", "topic", "message"]
    assert rows[1][1:] == ["Bug report", "hello"]
    assert rows[2][1] == "Other" and len(rows[2][2]) == site.MAX_FEEDBACK_CHARS


def test_static_assets_exist_and_are_small():
    for name, max_kb in [("favicon-32.png", 5), ("favicon-192.png", 20), ("apple-touch-icon.png", 20),
                         ("icon-512.png", 40), ("og-image.jpg", 120)]:
        f = STATIC / name
        assert f.exists() and f.stat().st_size < max_kb * 1024, name
    urls = [e.text for e in ET.parse(STATIC / "sitemap.xml").getroot().iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    assert len(urls) == sum(p.indexable for p in site.PAGES.values())
    assert "Sitemap:" in (STATIC / "robots.txt").read_text()
