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


# ---------------------------------------------------------------- carousel_caption

def test_caption_marks_current_line_only():
    lines = carousel_caption(REPLY_THREE, 1).split("\n")
    assert lines == [INTRO, LINE1, "▶ " + LINE2, LINE3, OFFER]
    assert lines[2].startswith("▶ 2. ")


def test_caption_current_zero_marks_first_line():
    lines = carousel_caption(REPLY_THREE, 0).split("\n")
    assert lines == [INTRO, "▶ " + LINE1, LINE2, LINE3, OFFER]


def test_caption_cut_to_limit():
    long_lines = [f"{n}. Dvoyka · 980000 · {'M,L,XL,' * 40} · https://t.me/status_dokon/{900 + n}" for n in range(1, 6)]
    reply = "\n".join([INTRO] + long_lines + [OFFER])
    assert len(reply) > CAPTION_LIMIT
    out = carousel_caption(reply, 0)
    assert len(out) <= CAPTION_LIMIT
    assert out.startswith(INTRO + "\n▶ 1. ")


def test_caption_without_numbered_lines_unchanged():
    assert carousel_caption(NO_RESULTS, 0) == NO_RESULTS


def test_caption_ten_not_confused_with_one():
    lines = [f"{n}. Dvoyka · 980000 · https://t.me/status_dokon/{900 + n}" for n in range(1, 11)]
    out = carousel_caption("\n".join(lines), 0).split("\n")
    assert out[0] == "▶ " + lines[0]
    assert out[9] == lines[9]
    assert sum(line.startswith("▶ ") for line in out) == 1


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
    assert rows[0][0].data == carousel_data("c", 0, IDS)
    assert rows[0][2].data == carousel_data("c", 2, IDS)


def test_buttons_wrap_around():
    last = carousel_buttons(2, IDS, ask_price=False)
    assert last[0][2].data == b"c:0:983,950,926"
    first = carousel_buttons(0, IDS, ask_price=False)
    assert first[0][0].data == b"c:2:983,950,926"


def test_buttons_single_id_has_no_nav_row():
    rows = carousel_buttons(0, [983], ask_price=False)
    assert len(rows) == 1
    assert all(not hasattr(b, "data") for b in rows[0])


def test_buttons_ask_price():
    rows = carousel_buttons(1, IDS, ask_price=True)
    price = [b for b in rows[-1] if getattr(b, "text", None) == "Narxini so'rash"]
    assert len(price) == 1
    assert price[0].data == carousel_data("p", 1, IDS)


def test_buttons_no_ask_price_button_when_off():
    rows = carousel_buttons(1, IDS, ask_price=False)
    assert not any(getattr(b, "text", None) == "Narxini so'rash" for row in rows for b in row)


def test_buttons_url_points_to_current_post():
    rows = carousel_buttons(1, IDS, ask_price=True)
    url_buttons = [b for b in rows[-1] if hasattr(b, "url")]
    assert len(url_buttons) == 1
    assert url_buttons[0].text == "Kanalda ko'rish"
    assert url_buttons[0].url.endswith("/status_dokon/950")
