"""Ticket #19 (carousel design) — pure helpers behind the one-message photo reply:
caption marking, self-contained callback data, inline keyboard rows."""
import pytest

from shop_assistant.bot import (
    CAPTION_LIMIT,
    carousel_buttons,
    carousel_caption,
    carousel_data,
    parse_carousel_data,
)

INTRO = "Ha, bor. Mana topilganlar:"
OFFER = "Narxini bilmoqchi bo'lsangiz raqamini yozing"
LINE1 = "1. Krossovka Nike Air · 350000 · 40,41,42 · 2026-09-12 · https://t.me/status_dokon/983"
LINE2 = "2. Dvoyka · 980000 · M,L,XL · 2026-09-13 · https://t.me/status_dokon/950"
LINE3 = "3. Dvoyka · 980000 · M,L,XL,2XL · 2026-09-10 · https://t.me/status_dokon/926"
REPLY_THREE = "\n".join([INTRO, LINE1, LINE2, LINE3, OFFER])
NO_RESULTS = "Kechirasiz, bunday mahsulot topilmadi."
IDS = [983, 950, 926]


def _data(button) -> bytes:
    """Callback payload of a Telethon inline button (`.type.data` since Telethon 1.45)."""
    return button.type.data


def _url(button) -> str:
    return button.type.url


# ---------------------------------------------------------------- carousel_caption

def test_caption_is_one_item_card():
    card = carousel_caption(REPLY_THREE, 0)
    lines = card.split("\n")
    assert lines[0] == "1. Krossovka Nike Air"
    assert "Narxi: 350 000 so'm" in lines
    assert "O'lcham: 40, 41, 42" in lines
    assert "Sana: 2026-09-12" in lines
    assert "2. " not in card and "3. " not in card and INTRO not in card and OFFER not in card
    assert "t.me" not in card


def test_caption_second_item():
    card = carousel_caption(REPLY_THREE, 1)
    assert card.startswith("2. Dvoyka\n")
    assert "Narxi: 980 000 so'm" in card and "O'lcham: M, L, XL" in card


def test_caption_missing_price_and_sizes():
    line = "1. Kiyim · narxi: so'rab beraman · - · 2026-08-19 · https://t.me/status_dokon/928"
    card = carousel_caption("\n".join([INTRO, line]), 0)
    assert "Narxi: so'rab beraman" in card
    assert "O'lcham" not in card


def test_caption_stale_note():
    line = ("1. Kiyim · narxi: so'rab beraman · - · 2026-06-09 · https://t.me/status_dokon/695 "
            "(Bu mahsulot eskirgan bo'lishi mumkin, egadan tasdiqlash lozim)")
    card = carousel_caption("\n".join([INTRO, line]), 0)
    assert card.startswith("1. Kiyim\n")
    assert "eskirgan" in card.lower()
    assert "(" not in card
    card2 = carousel_caption("1. Dvoyka · 980000 · M · 2026-09-10 · https://t.me/status_dokon/926 · [eskirgan]", 0)
    assert "eskirgan" in card2.lower() and "[" not in card2


def test_caption_without_numbered_line_falls_back_to_reply():
    assert carousel_caption(NO_RESULTS, 0) == NO_RESULTS
    assert carousel_caption(REPLY_THREE, 7) == REPLY_THREE[:CAPTION_LIMIT]


def test_caption_ten_not_confused_with_one():
    lines = [f"{i}. Item {i} · 100000 · M · 2026-09-0{i % 9 + 1} · https://t.me/status_dokon/{900 + i}"
             for i in range(1, 11)]
    card = carousel_caption("\n".join([INTRO, *lines]), 0)
    assert card.startswith("1. Item 1\n")
    card10 = carousel_caption("\n".join([INTRO, *lines]), 9)
    assert card10.startswith("10. Item 10\n")


# ---------------------------------------------------------------- carousel_data / parse_carousel_data

def test_data_bytes_format():
    data = carousel_data("c", 2, IDS)
    assert isinstance(data, bytes)
    assert data == b"c:2:983,950,926"


def test_data_round_trip():
    assert parse_carousel_data(carousel_data("c", 2, IDS)) == ("c", 2, IDS)


def test_data_five_ids_fit_64_bytes():
    ids = [10001, 20002, 30003, 40004, 50005]
    assert len(carousel_data("c", 4, ids)) <= 64
    assert len(carousel_data("p", 4, ids)) <= 64


@pytest.mark.parametrize("bad", [b"noop", b"c:x:1", b"z:0:1", b""])
def test_parse_malformed_is_none(bad):
    assert parse_carousel_data(bad) is None


def test_parse_keeps_id_order():
    assert parse_carousel_data(b"c:0:926,983,950") == ("c", 0, [926, 983, 950])


def test_parse_price_kind():
    assert parse_carousel_data(b"p:0:983") == ("p", 0, [983])


# ---------------------------------------------------------------- carousel_buttons

def test_buttons_two_rows_and_nav_texts():
    rows = carousel_buttons(1, IDS, ask_price=False)
    assert len(rows) == 2
    assert [b.text for b in rows[0]] == ["◀", "2/3", "▶"]


def test_buttons_nav_data():
    rows = carousel_buttons(1, IDS, ask_price=False)
    assert _data(rows[0][0]) == carousel_data("c", 0, IDS)
    assert _data(rows[0][2]) == carousel_data("c", 2, IDS)


def test_buttons_wrap_around():
    last = carousel_buttons(2, IDS, ask_price=False)
    assert _data(last[0][2]) == b"c:0:983,950,926"
    first = carousel_buttons(0, IDS, ask_price=False)
    assert _data(first[0][0]) == b"c:2:983,950,926"


def test_buttons_single_id_has_no_nav_row():
    rows = carousel_buttons(0, [983], ask_price=False)
    assert len(rows) == 1
    assert all(not hasattr(b, "data") for b in rows[0])


def test_buttons_ask_price():
    rows = carousel_buttons(1, IDS, ask_price=True)
    price = [b for b in rows[-1] if getattr(b, "text", None) == "Narxini so'rash"]
    assert len(price) == 1
    assert _data(price[0]) == carousel_data("p", 1, IDS)


def test_buttons_no_ask_price_button_when_off():
    rows = carousel_buttons(1, IDS, ask_price=False)
    assert not any(getattr(b, "text", None) == "Narxini so'rash" for row in rows for b in row)


def test_buttons_url_points_to_current_post():
    rows = carousel_buttons(1, IDS, ask_price=True)
    url_buttons = [b for b in rows[-1] if hasattr(b.type, "url")]
    assert len(url_buttons) == 1
    assert url_buttons[0].text == "Kanalda ko'rish"
    assert _url(url_buttons[0]).endswith("/status_dokon/950")
