"""Ticket #25 — listing tools report the total and page with `offset`; the agent never claims "only these".

Live bug: "menga 42 razmerli poyabzal kerak" → 5 shoes; "boshqalari bormi?" and "ha" → the model re-ran the same
find_products_tool(category=poyabzal, size=42), got the same top 5 and told the customer the catalog has only
these. The catalog had 23 size-42 shoes.

CONTRACT
- find_products_tool, semantic_search_tool, latest_posts_tool take an optional `offset: int = 0` (declared as
  a plain integer, not required). None, a negative number or a float from Gemini (5.0) are accepted:
  None / negative → 0, float → int.
- Page size = config.MAX_RESULTS (5); latest_posts_tool's `n` (clamped 1..5) is its page size.
- tools.format_products(products, start=1): numbered lines from `start` (`6. …`, `7. …`); default unchanged.
- A page with products = format_products(page, start=offset + 1) (offer line included when a shown item has
  no price), then ONE last line, the range/total line (model-facing, Uzbek Latin):
      more remain : `ko'rsatildi 6–10, jami 23; keyingilari: shu filtrlar bilan offset=10`
      last page   : `ko'rsatildi 21–23, jami 23; boshqa yo'q`
  i.e. `ko'rsatildi <first>–<last>, jami <total>` then `; keyingilari: … offset=<last>` only when
  last < total. Tests pin it with RANGE_RE (en dash or hyphen accepted); the last page has no "offset=".
- No match at all (any offset) → exactly tools.NO_RESULTS ("no results"), unchanged.
- Matches exist but offset >= total → one line starting `boshqa natija yo'q` that names the total, e.g.
  `boshqa natija yo'q (jami 23, offset=30)`; no numbered lines; never "no results".
- Totals: find_products_tool = every match of the filters (search.find_products(..., limit=None) returns all
  of them, newest first); latest_posts_tool = every sellable product (search.latest_posts(None)).
  semantic_search_tool = the candidates semantic search ranks: the tool asks
  search.semantic_search(text, max_price=..., limit=20) once (a fixed pool of 20 = 4 pages; there is no
  similarity cut-off) and pages through that list, so its total is at most 20.
- agent.last_run["tools"][i]["n_results"] for the three listing tools = number of numbered product lines
  (`<digits>. `), not the offer or range line; 0 for "no results" / "boshqa natija yo'q"; other tools unchanged.
- agent.system_prompt(lang), all three languages: a paging rule — on a request for more (boshqalari, yana,
  boshqa bormi, ещё, другие) or a yes (ha, xa, да) to an offer of more, call the same tool with the same
  filters and the next offset from the tool's last line; never say the catalog has only the shown items
  unless the tool's total says so. Pinned: "offset", those trigger words, the word "total", and a
  "never … only" rule.
- bot carousel on a second page (items numbered 6.–10.): carousel_caption(reply, current) takes the
  `current`-th numbered line of the reply (0-based position, not the number) and keeps its own number as the
  title; carousel_buttons(idx, ids, ask_price, lang="uz_latn", first=1) shows the counter
  `<first+idx>/<first+len(ids)-1>` (6/10 … 10/10; 1/n as before); callback data stays position-based
  (`c:<idx>:<ids>`); the ask-price button follows the line at that position. Numbering after a bot restart
  (_rebuild_reply numbers from 1) is out of scope.
Written by the senior — do not edit.
"""
import asyncio
import dataclasses
import re

import pytest

from shop_assistant import agent, bot, config, lang, llm, search, tools
from shop_assistant.agent import History
from shop_assistant.tools import NO_RESULTS, OFFER_PRICE, TOOLS, format_products
from tests.fake_gemini import FakeClient, call_response, dump, plain, text_response

RANGE_RE = re.compile(r"^ko'rsatildi (\d+)\s*[–-]\s*(\d+), jami (\d+)(.*)$")
NEXT_RE = re.compile(r"offset=(\d+)")
NO_MORE = "boshqa natija yo'q"
ITEM_RE = re.compile(r"^(\d+)\. ")
ID_RE = re.compile(r"t\.me/example_shop/(\d+)")
LISTING = ["find_products_tool", "semantic_search_tool", "latest_posts_tool"]


def _tool(name):
    return next(t for t in TOOLS if t.name == name)


def _items(out: str) -> list[tuple[int, int]]:
    """[(item number, post id)] of the numbered lines."""
    got = []
    for line in out.split("\n"):
        m = ITEM_RE.match(line)
        if m:
            got.append((int(m.group(1)), int(ID_RE.search(line).group(1))))
    return got


def _range(out: str) -> tuple[int, int, int, int | None]:
    """(first, last, total, next offset or None) from the LAST line."""
    last = out.split("\n")[-1]
    m = RANGE_RE.match(last)
    assert m, f"last line is not the range/total line: {last!r}"
    nxt = NEXT_RE.search(m.group(4))
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(nxt.group(1)) if nxt else None


@pytest.fixture
def catalog(monkeypatch, shoe_catalog):
    """The tools search the fixture catalog as if it were products.jsonl (no disk, no reload)."""
    monkeypatch.setattr(search, "PRODUCTS", shoe_catalog)
    monkeypatch.setattr(search, "_by_id", {p.id: p for p in shoe_catalog})
    monkeypatch.setattr(search, "reload_if_changed", lambda: False)
    return shoe_catalog


def _expected(catalog, **filters) -> list[int]:
    return [p.id for p in search.find_products(**filters, limit=None, products=catalog)]


def find(**args) -> str:
    return _tool("find_products_tool").call(args)


# ---------------------------------------------------------------- search + format_products

def test_find_products_limit_none_returns_every_match(shoe_catalog):
    got = search.find_products(category="poyabzal", size="42", limit=None, products=shoe_catalog)
    assert len(got) == 23
    assert [p.id for p in got] == list(range(2023, 2000, -1))          # newest first


def test_format_products_numbers_from_start(shoe_catalog):
    lines = format_products(shoe_catalog[:3], start=6).split("\n")
    assert [ln.split(" ", 1)[0] for ln in lines] == ["6.", "7.", "8."]
    assert shoe_catalog[0].link in lines[0]
    assert format_products(shoe_catalog[:2]).startswith("1. ")          # default unchanged


# ---------------------------------------------------------------- find_products_tool

def test_first_page_states_range_and_total(catalog):
    out = find(category="poyabzal", size="42")
    expected = _expected(catalog, category="poyabzal", size="42")
    assert _items(out) == list(zip(range(1, 6), expected[:5]))
    assert _range(out) == (1, 5, 23, 5)


def test_offset_5_returns_items_6_to_10(catalog):
    out = find(category="poyabzal", size="42", offset=5)
    expected = _expected(catalog, category="poyabzal", size="42")
    assert _items(out) == list(zip(range(6, 11), expected[5:10]))
    assert _range(out) == (6, 10, 23, 10)
    assert "\n1. " not in "\n" + out                                   # not the first page again


def test_pages_cover_every_match_once_in_order(catalog):
    seen, offset = [], 0
    for _ in range(10):
        out = find(category="poyabzal", size="42", offset=offset)
        seen += [pid for _, pid in _items(out)]
        first, last, total, nxt = _range(out)
        assert (first, total) == (offset + 1, 23)
        if nxt is None:
            break
        offset = nxt
    assert seen == _expected(catalog, category="poyabzal", size="42")


def test_last_page_has_no_next_hint(catalog):
    out = find(category="poyabzal", size="42", offset=20)
    assert [n for n, _ in _items(out)] == [21, 22, 23]
    assert _range(out) == (21, 23, 23, None)
    assert "offset=" not in out


@pytest.mark.parametrize("offset", [23, 30])
def test_offset_past_the_end_says_no_more(catalog, offset):
    out = find(category="poyabzal", size="42", offset=offset)
    assert out != NO_RESULTS
    assert out.startswith(NO_MORE)
    assert "23" in out
    assert _items(out) == []


@pytest.mark.parametrize("offset", [0, 5])
def test_no_match_is_still_no_results(catalog, offset):
    assert find(category="poyabzal", size="47", offset=offset) == NO_RESULTS


def test_offset_with_size_and_max_price_filters(catalog):
    filters = {"category": "poyabzal", "size": "42", "max_price": 600000}
    expected = _expected(catalog, **filters)
    assert len(expected) == 11                                         # priceless 2004 never passes max_price
    page2 = find(**filters, offset=5)
    assert _items(page2) == list(zip(range(6, 11), expected[5:10]))
    assert _range(page2) == (6, 10, 11, 10)
    page3 = find(**filters, offset=10)
    assert _items(page3) == [(11, expected[10])]
    assert _range(page3) == (11, 11, 11, None)


@pytest.mark.parametrize("offset", [None, -5, 0])
def test_none_or_negative_offset_is_the_first_page(catalog, offset):
    assert find(category="poyabzal", size="42", offset=offset) == find(category="poyabzal", size="42")


def test_float_offset_from_gemini_is_accepted(catalog):
    out = find(category="poyabzal", size="42", offset=5.0)
    assert [n for n, _ in _items(out)] == [6, 7, 8, 9, 10]


def test_everything_on_one_page_says_so(catalog):
    out = find(category="aksessuar")
    assert [pid for _, pid in _items(out)] == [2300]
    assert _range(out) == (1, 1, 1, None)


def test_offer_line_stays_before_the_range_line(catalog):
    out = find(category="poyabzal", size="42", offset=5)          # page 6–10 = ids 2018…2014, 2017 has no price
    lines = out.split("\n")
    assert out.count(OFFER_PRICE) == 1
    assert lines[-2] == OFFER_PRICE
    assert RANGE_RE.match(lines[-1])
    assert "narxi: so'rab beraman" in next(ln for ln in lines if "/2017" in ln)


def test_never_more_than_a_page(catalog):
    out = find(category="poyabzal", offset=3)
    assert len(_items(out)) == config.MAX_RESULTS
    assert _range(out)[:3] == (4, 8, 29)


# ---------------------------------------------------------------- semantic_search_tool

@pytest.fixture
def semantic_pool(monkeypatch, shoe_catalog):
    """Fake semantic_search: the first `size` sellable fixture products as the ranked list; records `limit`."""
    state = {"size": 12, "limits": []}
    sellable = [p for p in shoe_catalog if p.category != "boshqa"]

    def semantic_search(text, max_price=None, limit=5):
        state["limits"].append(limit)
        return sellable[:min(limit, state["size"])]
    monkeypatch.setattr(search, "semantic_search", semantic_search)
    state["ranked"] = sellable
    return state


def sem(**args) -> str:
    return _tool("semantic_search_tool").call({"text": "qishda kiyishga issiq narsa", **args})


def test_semantic_pages_through_its_pool(semantic_pool):
    ranked = [p.id for p in semantic_pool["ranked"]]
    first = sem()
    assert _items(first) == list(zip(range(1, 6), ranked[:5]))
    assert _range(first) == (1, 5, 12, 5)
    last = sem(offset=10)
    assert _items(last) == [(11, ranked[10]), (12, ranked[11])]
    assert _range(last) == (11, 12, 12, None)
    assert set(semantic_pool["limits"]) == {20}


def test_semantic_total_is_capped_at_20(semantic_pool):
    semantic_pool["size"] = 30
    assert _range(sem(offset=15)) == (16, 20, 20, None)
    out = sem(offset=20)
    assert out.startswith(NO_MORE) and "20" in out


def test_semantic_no_hits_is_no_results(semantic_pool):
    semantic_pool["size"] = 0
    assert sem() == NO_RESULTS
    assert sem(offset=5) == NO_RESULTS


# ---------------------------------------------------------------- latest_posts_tool

def latest(**args) -> str:
    return _tool("latest_posts_tool").call(args)


def test_latest_posts_pages(catalog):
    newest = [p.id for p in search.latest_posts(None, products=catalog)]
    assert len(newest) == 35
    out = latest(n=5, offset=5)
    assert _items(out) == list(zip(range(6, 11), newest[5:10]))
    assert _range(out) == (6, 10, 35, 10)
    out = latest(n=3, offset=3)
    assert _items(out) == list(zip(range(4, 7), newest[3:6]))
    assert _range(out) == (4, 6, 35, 6)


def test_latest_posts_default_page_and_clamp(catalog):
    assert _range(latest()) == (1, 5, 35, 5)
    assert len(_items(latest(n=50))) == config.MAX_RESULTS


def test_latest_posts_past_the_end(catalog):
    out = latest(offset=35)
    assert out.startswith(NO_MORE) and "35" in out and _items(out) == []


# ---------------------------------------------------------------- declarations

@pytest.mark.parametrize("name", LISTING)
def test_listing_tools_declare_an_optional_integer_offset(name):
    schema = _tool(name).input_schema
    assert schema["properties"]["offset"]["type"] == "integer"
    assert "offset" not in schema.get("required", [])
    assert "offset" in schema["properties"]["offset"]["description"].lower() or \
        "skip" in schema["properties"]["offset"]["description"].lower()


@pytest.mark.parametrize("name", ["search_faq_tool", "ask_owner"])
def test_other_tools_have_no_offset(name):
    assert "offset" not in _tool(name).input_schema["properties"]


# ---------------------------------------------------------------- system prompt

@pytest.mark.parametrize("code", ["uz_latn", "uz_cyrl", "ru"])
def test_system_prompt_has_the_paging_rule(code):
    prompt = agent.system_prompt(code)
    assert "offset" in prompt
    for word in ("boshqalari", "yana", "boshqa bormi", "ещё", "другие"):
        assert word in prompt, word
    assert re.search(r"\bha\b", prompt) and re.search(r"\bxa\b", prompt) and "да" in prompt
    assert "total" in prompt.lower()
    assert re.search(r"never[^.\n]*only", prompt, re.IGNORECASE)


# ---------------------------------------------------------------- agent plumbing (Gemini fake)

SHOES_42 = "menga 42 razmerli poyabzal kerak"
MORE = "boshqalari bormi?"


@pytest.fixture
def gemini(monkeypatch):
    def install(*script):
        fake = FakeClient(list(script))
        monkeypatch.setattr(llm, "client", lambda: fake)
        return fake
    return install


def test_boshqalari_bormi_calls_the_same_tool_with_offset_5(gemini, catalog):
    first_page = find(category="poyabzal", size="42")
    h = History()
    h.append(7001, "user", SHOES_42)
    h.append(7001, "assistant", "Mana 42 razmerli poyabzallar:\n" + "\n".join(first_page.split("\n")[:5]))

    def answer(contents):
        """Second model request: echo the numbered lines the tool returned (as a model would)."""
        result = plain(contents[-1])["parts"][0]["function_response"]["response"]["result"]
        return text_response("Yana 42 razmerli poyabzallar:\n" +
                             "\n".join(ln for ln in result.split("\n") if ITEM_RE.match(ln)))
    fake = gemini(call_response("find_products_tool", {"category": "poyabzal", "size": "42", "offset": 5}),
                  answer)
    reply = agent.run_agent(7001, MORE, h)

    sent = dump(fake.calls[0]["contents"])
    assert SHOES_42 in sent and MORE in sent                          # the earlier page is in the context
    assert "offset" in dump(fake.calls[0]["config"].system_instruction)
    tool_result = dump(fake.calls[1]["contents"])
    assert "ko'rsatildi 6" in tool_result and "jami 23" in tool_result
    assert [n for n, _ in _items(reply)] == [6, 7, 8, 9, 10]
    expected = _expected(catalog, category="poyabzal", size="42")
    assert [pid for _, pid in _items(reply)] == expected[5:10]
    assert agent.last_run["tools"] == [{"name": "find_products_tool",
                                        "input": {"category": "poyabzal", "size": "42", "offset": 5},
                                        "n_results": 5}]
    assert h.get(7001)[-1] == {"role": "assistant", "content": reply}


def test_n_results_is_zero_past_the_end(gemini, catalog):
    gemini(call_response("find_products_tool", {"category": "poyabzal", "size": "42", "offset": 25}),
           text_response("42 razmerli poyabzallarning hammasini (23 ta) ko'rsatdim."))
    agent.run_agent(7002, "yana", History())
    assert agent.last_run["tools"][0]["n_results"] == 0


def test_n_results_counts_item_lines_only(gemini, catalog):
    gemini(call_response("find_products_tool", {"category": "poyabzal", "size": "42", "offset": 20}),
           text_response("Oxirgi 3 tasi."))
    agent.run_agent(7003, "ha", History())
    assert agent.last_run["tools"][0]["n_results"] == 3


# ---------------------------------------------------------------- bot carousel on page 2

INTRO = "Yana 42 razmerli poyabzallar:"
PAGE2_IDS = [2018, 2017, 2016, 2015, 2014]
PAGE2 = "\n".join([INTRO,
                   "6. Krossovka Adidas Gazelle · 760000 · 41,42,43,44 · 2026-08-18 · https://t.me/example_shop/2018",
                   "7. Krossovka Salomon XT-6 · narxi: so'rab beraman · 40,41,42,43 · 2026-08-17 · https://t.me/example_shop/2017",
                   "8. Krossovka On Cloud 5 · 700000 · 41,42,43,44 · 2026-08-16 · https://t.me/example_shop/2016",
                   "9. Krossovka Hoka Clifton 9 · 670000 · 40,41,42,43 · 2026-08-15 · https://t.me/example_shop/2015",
                   "10. Krossovka Skechers Go Walk · 640000 · 41,42,43,44 · 2026-08-14 · https://t.me/example_shop/2014",
                   OFFER_PRICE])


def _texts(rows) -> list[str]:
    return [b.text for row in rows for b in row]


def test_caption_uses_the_position_among_numbered_lines():
    assert bot.carousel_caption(PAGE2, 0).startswith("6. Krossovka Adidas Gazelle\n")
    assert bot.carousel_caption(PAGE2, 1).startswith("7. Krossovka Salomon XT-6\n")
    assert bot.carousel_caption(PAGE2, 4).startswith("10. Krossovka Skechers Go Walk\n")
    assert bot.carousel_caption(PAGE2, 5) == PAGE2[:bot.CAPTION_LIMIT]


def test_counter_starts_at_the_first_item_number():
    rows = bot.carousel_buttons(0, PAGE2_IDS, ask_price=False, first=6)
    assert [b.text for b in rows[0]] == ["◀", "6/10", "▶"]
    assert rows[0][0].type.data == bot.carousel_data("c", 4, PAGE2_IDS)       # wraps to the last item
    assert rows[0][2].type.data == bot.carousel_data("c", 1, PAGE2_IDS)
    assert [b.text for b in bot.carousel_buttons(4, PAGE2_IDS, ask_price=False, first=6)[0]] == ["◀", "10/10", "▶"]
    assert [b.text for b in bot.carousel_buttons(1, PAGE2_IDS, ask_price=False)[0]] == ["◀", "2/5", "▶"]


class _Telegram:
    def __init__(self):
        self.files: list[dict] = []

    async def send_file(self, chat_id, file, caption=None, buttons=None, reply_to=None, **kw):
        self.files.append({"file": file, "caption": caption, "buttons": buttons})
        return type("Msg", (), {"id": 8100 + len(self.files), "photo": None})()


class _Message:
    def __init__(self):
        self.chat_id = self.sender_id = 4343
        self.message = type("Msg", (), {"id": 1})()
        self.client = _Telegram()
        self.replies: list[str] = []

    async def reply(self, text, **kw):
        self.replies.append(text)


class _Callback:
    def __init__(self, data: bytes, message_id: int):
        self.data, self.message_id = data, message_id
        self.chat_id = self.sender_id = 4343
        self.client = _Telegram()
        self.answers: list = []
        self.edits: list[dict] = []

    async def answer(self, message=None, **kw):
        self.answers.append(message)

    async def edit(self, text=None, file=None, buttons=None, **kw):
        self.edits.append({"text": text, "file": file, "buttons": buttons})
        return type("Msg", (), {"id": self.message_id, "photo": None})()


def test_second_page_carousel_numbering_counter_and_ask_price(monkeypatch):
    monkeypatch.setenv("TG_OWNER_ID", "999")
    monkeypatch.setattr(bot, "_captions", {})

    async def media_for(client, ids):
        return {i: f"photo-{i}" for i in ids}
    monkeypatch.setattr(bot, "_media_for", media_for)
    ask = lang.TEXTS["uz_latn"]["ask_price_button"]

    ev = _Message()
    asyncio.run(bot.send_reply(ev, PAGE2))
    assert ev.replies == []
    sent = ev.client.files[0]
    assert sent["file"] == "photo-2018"
    assert sent["caption"].startswith("6. Krossovka Adidas Gazelle\n")
    assert "6/10" in _texts(sent["buttons"]) and ask not in _texts(sent["buttons"])

    cb = _Callback(bot.carousel_data("c", 1, PAGE2_IDS), message_id=8101)
    asyncio.run(bot.handle_callback(cb))
    edit = cb.edits[0]
    assert edit["file"] == "photo-2017"
    assert edit["text"].startswith("7. Krossovka Salomon XT-6\n")
    assert "7/10" in _texts(edit["buttons"]) and ask in _texts(edit["buttons"])

    escalated = []

    async def fake_escalate(customer_id, question, post_ids):
        escalated.append(post_ids)
    monkeypatch.setattr(bot, "escalate", fake_escalate)
    asyncio.run(bot.handle_callback(_Callback(bot.carousel_data("p", 1, PAGE2_IDS), message_id=8101)))
    assert escalated == [[2017]]
