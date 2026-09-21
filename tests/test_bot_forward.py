"""Ticket #19 — forward matching channel posts (pure part: link → post id extraction).
Live forwarding (photos, ≤ 10 s, deleted posts) is the manual protocol in Notion."""
from shop_assistant.bot import post_ids_in

REPLY_THREE = (
    "1. Krossovka Nike Air · 350000 · 40,41,42 · 2026-09-12 · https://t.me/status_dokon/1300\n"
    "2. Dvoyka · 980000 · M,L,XL · 2026-09-13 · https://t.me/status_dokon/1301\n"
    "3. Dvoyka · 980000 · M,L,XL,2XL · 2026-09-10 · https://t.me/status_dokon/1234\n"
    "Qaysi biri kerak? Raqamini yozing."
)

NO_RESULTS = "Kechirasiz, bunday mahsulot topilmadi."
ESCALATED = "Sizning savolingiz egaga yuborildi, tez orada javob beramiz."


def test_numbered_reply_ids_in_text_order():
    assert post_ids_in(REPLY_THREE) == [1300, 1301, 1234]


def test_duplicate_link_kept_once_at_first_position():
    text = (
        "1. Krossovka Nike Air · 350000 · https://t.me/status_dokon/1300\n"
        "2. Dvoyka · 980000 · https://t.me/status_dokon/1234\n"
        "Eslatma: Krossovka Nike Air uchun obunachilarga 320000 — https://t.me/status_dokon/1300"
    )
    assert post_ids_in(text) == [1300, 1234]


def test_at_most_five_ids():
    ids = [1300, 1301, 1234, 1302, 1305, 1310, 1311]
    text = "\n".join(f"{n}. Mahsulot · https://t.me/status_dokon/{i}" for n, i in enumerate(ids, 1))
    assert post_ids_in(text) == ids[:5]


def test_empty_and_no_links():
    assert post_ids_in("") == []
    assert post_ids_in(NO_RESULTS) == []


def test_other_channel_links_ignored():
    assert post_ids_in("Boshqa do'konda bor: https://t.me/other_shop/55") == []
    mixed = (
        "1. Kurtka · 1200000 · https://t.me/other_shop/55\n"
        "2. Krossovka Nike Air · 350000 · https://t.me/status_dokon/1300\n"
        "3. Dvoyka · 980000 · https://t.me/status_dokon/1234"
    )
    assert post_ids_in(mixed) == [1300, 1234]


def test_bare_and_https_forms_accepted():
    assert post_ids_in("Ha, bor: t.me/status_dokon/1300") == [1300]
    assert post_ids_in("Ha, bor: https://t.me/status_dokon/1300") == [1300]


def test_link_followed_by_punctuation():
    assert post_ids_in("Mana bu: https://t.me/status_dokon/1300.") == [1300]
    assert post_ids_in("Mana bu (https://t.me/status_dokon/1300)") == [1300]


def test_link_in_eskirgan_line():
    text = (
        "1. Krossovka Nike Air · 350000 · https://t.me/status_dokon/1300\n"
        "2. Dvoyka · 980000 · 2026-09-10 · https://t.me/status_dokon/1234 [eskirgan]"
    )
    assert post_ids_in(text) == [1300, 1234]


def test_ids_are_ints():
    ids = post_ids_in(REPLY_THREE)
    assert ids and all(isinstance(i, int) for i in ids)


def test_escalation_text_has_no_ids():
    assert post_ids_in(ESCALATED) == []
