"""Ticket #21 — numbered product replies so a customer can answer `narxi 2`."""
import dataclasses

from shop_assistant.tools import NO_RESULTS, format_products

OFFER = "Narxini bilmoqchi bo'lsangiz raqamini yozing"
ASK = "narxi: so'rab beraman"


def _lines(text: str) -> list[str]:
    return text.split("\n")


def test_lines_are_numbered_in_order(products):
    lines = _lines(format_products(products))
    assert len(lines) == len(products)
    for i, line in enumerate(lines, start=1):
        assert line.startswith(f"{i}. ")


def test_each_line_keeps_link_and_name(products):
    lines = _lines(format_products(products))
    for p, line in zip(products, lines):
        assert p.link in line
        assert p.name in line
        assert "t.me" in line


def test_no_line_contains_none(products):
    priceless = [dataclasses.replace(products[0], price=None)] + products[1:]
    for line in _lines(format_products(priceless)):
        assert "None" not in line


def test_missing_price_renders_ask_phrase(products):
    priceless = [products[0], dataclasses.replace(products[1], price=None)]
    lines = _lines(format_products(priceless))
    assert lines[1].startswith("2. ")
    assert ASK in lines[1]
    assert ASK not in lines[0]


def test_offer_line_once_when_a_price_is_missing(products):
    priceless = [dataclasses.replace(products[0], price=None), products[1]]
    out = format_products(priceless)
    lines = _lines(out)
    assert lines[-1] == OFFER
    assert out.count(OFFER) == 1
    assert len(lines) == len(priceless) + 1


def test_offer_line_once_even_with_two_priceless_items(products):
    priceless = [dataclasses.replace(p, price=None) for p in products[:2]] + [products[2]]
    out = format_products(priceless)
    assert out.count(OFFER) == 1
    assert _lines(out)[-1] == OFFER
    assert len(_lines(out)) == len(priceless) + 1


def test_no_offer_line_when_all_prices_known(products):
    assert all(p.price is not None for p in products)
    out = format_products(products)
    assert OFFER not in out
    assert ASK not in out
    assert len(_lines(out)) == len(products)


def test_single_product_still_numbered(products):
    out = format_products([products[1]])
    assert out.startswith("1. ")
    assert "\n" not in out
    assert products[1].link in out


def test_empty_list_is_no_results():
    out = format_products([])
    assert out == NO_RESULTS
    assert OFFER not in out


def test_stale_tag_stays_at_end_of_line(products):
    stale = dataclasses.replace(products[2], stale=True)
    lines = _lines(format_products([products[0], stale]))
    assert lines[1].startswith("2. ")
    assert lines[1].endswith("[eskirgan]")
    assert not lines[0].endswith("[eskirgan]")


def test_ten_products_reach_number_ten(products):
    base = products[0]
    ten = [dataclasses.replace(base, id=2000 + i, link=f"https://t.me/example_shop/{2000 + i}")
           for i in range(10)]
    lines = _lines(format_products(ten))
    assert len(lines) == 10
    assert lines[9].startswith("10. ")
    assert "https://t.me/example_shop/2009" in lines[9]
    assert lines[0].startswith("1. ") and not lines[0].startswith("10. ")
