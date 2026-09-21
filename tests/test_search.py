"""Ticket #7 — FR-8, FR-11, FR-12, FR-16."""
import pytest

from shop_assistant.search import find_products, is_stale, keyword_match, latest_posts

TODAY = "2026-09-16"


@pytest.mark.parametrize("date,expected", [
    ("2026-09-10", False),
    ("2026-07-18", False),   # exactly 60 days
    ("2026-07-17", True),    # 61 days
    ("2026-06-01", True),
])
def test_is_stale_boundary(date, expected):
    assert is_stale(date, TODAY, 60) is expected


def test_keyword_match_is_script_insensitive(products):
    assert keyword_match(["Кроссовка"], products[1])
    assert keyword_match(["nike"], products[1])
    assert not keyword_match(["kurtka"], products[1])


def test_find_products_filters_are_anded(products):
    got = find_products(category="kiyim", max_price=1000000, products=products)
    assert [p.id for p in got] == [1234]
    assert find_products(category="kiyim", size="42", products=products) == []


def test_find_products_size_and_price_range(products):
    got = find_products(size="42", min_price=300000, max_price=400000, products=products)
    assert [p.id for p in got] == [1300]


def test_find_products_color(products):
    assert [p.id for p in find_products(color="qora", products=products)] == [900]


def test_find_products_marks_stale_and_keeps_link_date(products):
    got = find_products(keywords=["kurtka"], products=products)
    assert len(got) == 1 and got[0].stale is True
    assert got[0].link.startswith("https://t.me/status_dokon/") and got[0].date   # FR-11


def test_find_products_limit_and_newest_first(products):
    got = find_products(category="kiyim", limit=1, products=products)
    assert [p.id for p in got] == [1234]          # 2026-09-10 beats 2026-06-01


def test_find_products_no_filters_empty_catalog():
    assert find_products(products=[]) == []


def test_latest_posts(products):
    assert [p.id for p in latest_posts(2, products=products)] == [1300, 1234]


# ---- ticket #20 follow-up: announcements (category boshqa) never reach a customer

def _announcement(products):
    import dataclasses
    return dataclasses.replace(products[0], id=695, link="https://t.me/status_dokon/695",
                               name="Boshqa", category="boshqa", price=None, sizes=(),
                               keywords=("kiyim", "yengi kolleksiya"), body="Yengi kolleksiya")


def test_find_products_skips_boshqa(products):
    catalog = products + [_announcement(products)]
    assert all(p.category != "boshqa" for p in find_products(products=catalog))
    assert find_products(keywords=["kolleksiya"], products=catalog) == []
    assert find_products(category="boshqa", products=catalog) == []


def test_latest_posts_skips_boshqa(products):
    catalog = products + [_announcement(products)]
    assert 695 not in [p.id for p in latest_posts(10, products=catalog)]


def test_semantic_search_skips_boshqa(products, monkeypatch):
    import numpy as np
    from shop_assistant import search
    ann = _announcement(products)
    monkeypatch.setattr(search, "_by_id", {ann.id: ann, products[1].id: products[1]})
    monkeypatch.setattr(search, "_ids", [ann.id, products[1].id])
    monkeypatch.setattr(search, "_matrix", np.array([[1.0, 0.0], [0.0, 1.0]]))
    from shop_assistant import index
    monkeypatch.setattr(index, "embed", lambda texts: np.array([[1.0, 0.0]]))
    ids = [p.id for p in search.semantic_search("yengi kolleksiya")]
    assert 695 not in ids and products[1].id in ids
