"""Ticket #20 — product_name(): customer-readable names, never slogans (written by senior — do not edit)."""
import pytest

from shop_assistant.extract import product_name, strip_footer
from tests.conftest import CAPTION_DVOYKA, CAPTION_KROSSOVKA, CAPTION_KURTKA, FOOTER

BODY_KROSSOVKA = strip_footer(CAPTION_KROSSOVKA)
BODY_DVOYKA = strip_footer(CAPTION_DVOYKA)
BODY_KURTKA = strip_footer(CAPTION_KURTKA)
BODY_SLOGAN_ONLY = strip_footer("Yengi kolleksiya\nRazmer: 40 41 42 43\nNarx: 350.000" + FOOTER)

KW_KROSSOVKA = ["krossovka", "nike", "sneakers"]
KW_KURTKA = ["kurtka", "barena", "jacket"]
KW_SVITER = ["sviter", "lora piana"]

SLOGANS = [
    "Yengi kolleksiya",
    "Yangi kolleksiya",
    "New collection",
    "Unknown",
    "Okam bu modella siz uchun eng yahshilari‼",
    "Assalomu alaykum",
    "YENGI KOLLEKSIYA",
    "new collection",
]


def _words(name: str) -> list[str]:
    return name.split()


# --- good names pass through -------------------------------------------------

def test_good_name_returned_unchanged():
    item = {"name": "Krossovka Nike Air", "category": "poyabzal", "keywords": KW_KROSSOVKA}
    assert product_name(item, "poyabzal", BODY_KROSSOVKA) == "Krossovka Nike Air"


def test_good_name_whitespace_stripped():
    item = {"name": "  Qishki kurtka \n", "category": "kiyim", "keywords": KW_KURTKA}
    assert product_name(item, "kiyim", BODY_KURTKA) == "Qishki kurtka"


def test_good_name_with_brand_kept():
    item = {"name": "Sviter Lora Piana", "category": "kiyim", "keywords": KW_SVITER}
    assert product_name(item, "kiyim", BODY_KURTKA) == "Sviter Lora Piana"


# --- slogans are replaced ----------------------------------------------------

@pytest.mark.parametrize("slogan", SLOGANS)
def test_slogan_replaced_by_keyword_type(slogan):
    item = {"name": slogan, "category": "poyabzal", "keywords": KW_KROSSOVKA}
    got = product_name(item, "poyabzal", BODY_SLOGAN_ONLY)
    assert got.lower() != slogan.lower()
    assert got.startswith("Krossovka")
    assert 1 <= len(_words(got)) <= 4


@pytest.mark.parametrize("slogan", SLOGANS)
def test_slogan_replaced_by_category_when_no_keywords(slogan):
    item = {"name": slogan, "category": "kiyim", "keywords": []}
    got = product_name(item, "kiyim", BODY_SLOGAN_ONLY)
    assert got.lower() != slogan.lower()
    assert got.startswith("Kiyim")
    assert 1 <= len(_words(got)) <= 4


def test_slogan_case_insensitive_mixed_case():
    item = {"name": "yEnGi KoLlEkSiYa", "category": "kiyim", "keywords": KW_KURTKA}
    got = product_name(item, "kiyim", BODY_SLOGAN_ONLY)
    assert got.startswith("Kurtka")


@pytest.mark.parametrize("category", ["kiyim", "poyabzal", "aksessuar"])
def test_category_word_used_when_no_keywords(category):
    item = {"name": "New collection", "category": category, "keywords": []}
    got = product_name(item, category, BODY_SLOGAN_ONLY)
    assert got.split()[0].lower() == category


# --- empty / missing names ----------------------------------------------------

@pytest.mark.parametrize("item", [
    {},
    {"name": ""},
    {"name": None},
    {"name": "   "},
])
def test_empty_or_missing_name_falls_back_to_keyword(item):
    item = {**item, "keywords": KW_KROSSOVKA}
    got = product_name(item, "poyabzal", BODY_KROSSOVKA)
    assert got.strip() != ""
    assert got.startswith("Krossovka")


def test_empty_name_no_keywords_uses_category():
    got = product_name({"name": ""}, "poyabzal", BODY_SLOGAN_ONLY)
    assert got.startswith("Poyabzal")


def test_missing_keywords_key_entirely():
    got = product_name({"name": None, "category": "kiyim"}, "kiyim", BODY_SLOGAN_ONLY)
    assert got.strip() != ""
    assert got.startswith("Kiyim")


# --- URL / no-letter names ----------------------------------------------------

def test_url_name_replaced():
    item = {"name": "https://t.me/status_dokon/123", "keywords": KW_KROSSOVKA}
    got = product_name(item, "poyabzal", BODY_KROSSOVKA)
    assert "http" not in got and "t.me" not in got
    assert got.startswith("Krossovka")


@pytest.mark.parametrize("bad", ["🔥🔥🔥", "350.000", "‼‼", "👟👟 👟"])
def test_no_letter_name_replaced(bad):
    item = {"name": bad, "keywords": KW_KURTKA}
    got = product_name(item, "kiyim", BODY_KURTKA)
    assert got != bad
    assert any(c.isalpha() for c in got)
    assert got.startswith("Kurtka")


# --- shape of the fallback name ----------------------------------------------

def test_fallback_first_keyword_then_optional_brand():
    item = {"name": "Yengi kolleksiya", "keywords": KW_KURTKA}
    got = product_name(item, "kiyim", BODY_SLOGAN_ONLY)
    assert got.startswith("Kurtka")
    assert len(_words(got)) <= 4


def test_fallback_capped_at_four_words_with_long_keywords():
    item = {"name": "New collection",
            "keywords": ["sport kostyum dvoyka erkaklar uchun", "kostyum", "dvoyka", "adidas"]}
    got = product_name(item, "kiyim", BODY_DVOYKA)
    assert 1 <= len(_words(got)) <= 4
    assert got.startswith("Sport")


def test_fallback_has_no_newline():
    item = {"name": "Yengi kolleksiya", "keywords": ["krossovka\nnike", "nike"]}
    got = product_name(item, "poyabzal", BODY_SLOGAN_ONLY)
    assert "\n" not in got


def test_fallback_is_title_cased_at_first_letter():
    item = {"name": "", "keywords": ["dvoyka", "sport kostyum"]}
    got = product_name(item, "kiyim", BODY_DVOYKA)
    assert got[0].isupper()
    assert got.startswith("Dvoyka")


def test_fallback_from_lowercase_keyword_is_capitalised():
    got = product_name({"name": "Unknown", "keywords": []}, "aksessuar", BODY_SLOGAN_ONLY)
    assert got[0] == "A"
    assert got.startswith("Aksessuar")


# --- category boshqa ---------------------------------------------------------

def test_boshqa_no_keywords_uses_first_body_line():
    body = strip_footer("Qishki kurtka Barena\nRang: qora, kok\nNarx: 1.200.000" + FOOTER)
    got = product_name({"name": "Yengi kolleksiya", "keywords": []}, "boshqa", body)
    assert got.startswith("Qishki kurtka")
    assert "\n" not in got


def test_boshqa_body_line_is_slogan_falls_back_to_mahsulot():
    body = strip_footer("Yengi kolleksiya\nNarx: 350.000" + FOOTER)
    got = product_name({"name": "New collection", "keywords": []}, "boshqa", body)
    assert got == "Mahsulot"


def test_boshqa_empty_body_falls_back_to_mahsulot():
    got = product_name({"name": "", "keywords": []}, "boshqa", "")
    assert got == "Mahsulot"


def test_boshqa_body_line_without_letters_falls_back_to_mahsulot():
    body = strip_footer("🔥🔥🔥\n350.000" + FOOTER)
    got = product_name({"name": "🔥🔥🔥", "keywords": []}, "boshqa", body)
    assert got == "Mahsulot"


def test_boshqa_with_keywords_still_prefers_keyword():
    got = product_name({"name": "Unknown", "keywords": ["sumka", "gucci"]}, "boshqa", BODY_SLOGAN_ONLY)
    assert got.startswith("Sumka")
