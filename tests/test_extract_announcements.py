"""Ticket #20 follow-up — announcements never become searchable products (category boshqa)."""
from shop_assistant.extract import is_announcement, strip_footer
from shop_assistant.models import Post
from tests.conftest import FOOTER


def _post(caption: str, has_media: bool) -> Post:
    return Post(id=695, date="2026-06-09T11:04:04", link="https://t.me/status_dokon/695",
                caption=caption, has_media=has_media)


def test_bare_username_post_is_announcement():
    p = _post("@Shaxzodakh", has_media=False)
    assert is_announcement(p, strip_footer(p.caption), None, ())


def test_footer_only_caption_is_announcement():
    p = _post(FOOTER, has_media=True)
    assert is_announcement(p, strip_footer(p.caption), None, ())


def test_text_only_teaser_without_price_or_sizes_is_announcement():
    p = _post("Yengi kolleksiya\nKuz mavsumi uchun\nKutib qoling" + FOOTER, has_media=False)
    assert is_announcement(p, strip_footer(p.caption), None, ())


def test_text_only_post_with_price_is_a_product():
    p = _post("Dvoyka\nNarx: 980.000" + FOOTER, has_media=False)
    assert not is_announcement(p, strip_footer(p.caption), 980000, ())


def test_text_only_post_with_sizes_is_a_product():
    p = _post("Krossovka\nRazmer: 40 41 42" + FOOTER, has_media=False)
    assert not is_announcement(p, strip_footer(p.caption), None, ("40", "41", "42"))


def test_photo_post_without_price_is_still_a_product():
    p = _post("Kurtka Barena\nRazmer: L XL" + FOOTER, has_media=True)
    assert not is_announcement(p, strip_footer(p.caption), None, ("L", "XL"))
    assert not is_announcement(p, strip_footer(p.caption), None, ())
