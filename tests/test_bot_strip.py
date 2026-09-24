"""Ticket #27 — the listing tools' paging line never reaches a customer.

Live bug (eval, 2026-09-24): "600 ming gacha sviter bormi?" → the model copied the tool's internal line
`ko'rsatildi 1–3, jami 3; boshqa yo'q` into the customer reply.

CONTRACT
- bot.strip_tool_lines(text) -> str, pure. Removes every WHOLE line that is one of the listing tools'
  model-only lines (#25, tools._page):
      range/total, more remain : `ko'rsatildi 6–10, jami 23; keyingilari: shu filtrlar bilan offset=10`
      range/total, last page   : `ko'rsatildi 21–23, jami 23; boshqa yo'q`
      past the end             : `boshqa natija yo'q (jami 23, offset=30)`
  also when the model retyped it with surrounding spaces, a hyphen instead of the en dash, or the
  Uzbek ‘ / ’ apostrophe (`ko‘rsatildi`). The pattern should live next to the formatter in tools.py and
  bot.py import it — the tests feed real tool output through the helper, so the wording cannot drift.
- Everything else is kept byte-for-byte: product lines (`6. Krossovka … · https://t.me/…`), the offer line,
  the model's own sentences — including total sentences ("Jami 23 ta shunday poyabzal bor. …",
  "Жами 23 та …", "Всего найдено 22 товара") and a sentence that merely contains the word
  ("Jami 3 ta topildi, hammasi ko'rsatildi."). A reply without tool lines comes back unchanged.
- Blank lines left behind are collapsed (never three newlines in a row, no leading/trailing blank line).
  None / "" → "". Never "" when the input had other text.
- bot.handle_customer applies it to the agent's reply before send_reply: the plain-text reply, the carousel
  caption/buttons and the reply kept in bot._captions (◀/▶ rebuild cards from it) have no tool line and
  keep the reply's own numbering (6/10 counter on a second page). Owner messages are relayed untouched.
- agent.system_prompt(lang), all three languages: one rule that mentions "ko'rsatildi" and says not to
  copy / show that line (it is for the assistant only); the customer is told the total in their language.
Written by the senior — do not edit.
"""
import asyncio
import re

import pytest

from shop_assistant import agent, bot, config, faq, search, tools
from shop_assistant.bot import strip_tool_lines
from shop_assistant.models import FaqEntry
from tests.test_bot_lang import FakeEvent, FakeTelegram

OWNER = 999
CUSTOMER = 4242
LEAKS = ("ko'rsatildi", "ko‘rsatildi", "offset=", "boshqa natija yo'q", "keyingilari:")

INTRO = "Ha, 600 ming so'mgacha sviterlar bor:"
S1 = "1. Sviter oversize · 450000 · M,L · 2026-09-20 · https://t.me/example_shop/1401"
S2 = "2. Sviter vyazaniy · narxi: so'rab beraman · L,XL · 2026-09-18 · https://t.me/example_shop/1398"
S3 = "3. Sviter bolalar · 290000 · - · 2026-09-15 · https://t.me/example_shop/1390"
OFFER = tools.OFFER_PRICE
LAST_PAGE = "ko'rsatildi 1–3, jami 3; boshqa yo'q"

P6 = [f"{n}. Krossovka model {n} · {250000 + 30000 * n} · 41,42,43 · 2026-08-{30 - n:02d} · "
      f"https://t.me/example_shop/{2030 - n}" for n in range(6, 11)]
MORE = "ko'rsatildi 6–10, jami 23; keyingilari: shu filtrlar bilan offset=10"
TOTAL_UZ = "Jami 23 ta shunday poyabzal bor. Keyingilarini ko'rsatishimni xohlaysizmi?"
TOTAL_CYRL = "Жами 23 та шундай пойабзал бор. Кейингиларини кўрсатайми?"
TOTAL_RU = "Всего найдено 22 товара. Показать следующие?"


def _no_leak(text: str) -> None:
    for bad in LEAKS:
        assert bad not in text, f"{bad!r} reached the customer: {text!r}"


# ---------------------------------------------------------------- the helper, hand-written replies

def test_last_page_line_removed_rest_kept_byte_for_byte():
    reply = "\n".join([INTRO, S1, S2, S3, OFFER, LAST_PAGE])
    assert strip_tool_lines(reply) == "\n".join([INTRO, S1, S2, S3, OFFER])


def test_more_remain_line_removed_second_page_numbering_kept():
    reply = "\n".join(["Yana 5 tasi:", *P6, MORE, TOTAL_UZ])
    assert strip_tool_lines(reply) == "\n".join(["Yana 5 tasi:", *P6, TOTAL_UZ])


def test_past_the_end_line_removed():
    reply = "boshqa natija yo'q (jami 23, offset=30)\nHammasi ko'rsatib bo'lindi. Boshqa o'lcham bilan qidiraymi?"
    assert strip_tool_lines(reply) == "Hammasi ko'rsatib bo'lindi. Boshqa o'lcham bilan qidiraymi?"


@pytest.mark.parametrize("line", [
    "ko'rsatildi 1-3, jami 3; boshqa yo'q",                                    # hyphen instead of en dash
    "  ko'rsatildi 1–3, jami 3; boshqa yo'q  ",                                # surrounding spaces
    "ko‘rsatildi 1–5, jami 23; keyingilari: shu filtrlar bilan offset=5",      # Uzbek ‘ apostrophe
    "ko’rsatildi 11–15, jami 23; keyingilari: shu filtrlar bilan offset=15",   # typographic ’
])
def test_retyped_variants_removed(line):
    assert strip_tool_lines("\n".join([INTRO, S1, line])) == "\n".join([INTRO, S1])


@pytest.mark.parametrize("sentence", [
    TOTAL_UZ,
    TOTAL_CYRL,
    TOTAL_RU,
    "Jami 3 ta topildi, hammasi ko'rsatildi.",
    "Barcha 3 ta mahsulot ko'rsatildi, boshqa yo'q.",
])
def test_model_total_sentences_kept(sentence):
    reply = "\n".join([INTRO, S1, S2, S3, OFFER, sentence])
    assert strip_tool_lines(reply) == reply


def test_reply_of_only_product_lines_unchanged():
    reply = "\n".join([S1, S2, S3, OFFER])
    assert strip_tool_lines(reply) == reply


def test_reply_with_paragraphs_and_no_tool_line_unchanged():
    reply = f"{INTRO}\n\n{S1}\n{S3}\n\nYetkazib berish bor. Qaysi biri yoqdi?"
    assert strip_tool_lines(reply) == reply


def test_blank_lines_left_behind_are_collapsed():
    reply = f"{INTRO}\n\n{S1}\n{S3}\n\n{LAST_PAGE}\n\nBoshqa filtr bilan qidiraymi?"
    assert strip_tool_lines(reply) == f"{INTRO}\n\n{S1}\n{S3}\n\nBoshqa filtr bilan qidiraymi?"


def test_trailing_tool_line_after_blank_leaves_no_trailing_blank():
    reply = f"{INTRO}\n{S1}\n{S3}\n\n{MORE}\n"
    assert strip_tool_lines(reply) == f"{INTRO}\n{S1}\n{S3}"


def test_leading_tool_line_leaves_no_leading_blank():
    reply = f"{LAST_PAGE}\n\n{TOTAL_RU}"
    assert strip_tool_lines(reply) == TOTAL_RU


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_and_none_are_safe(empty):
    assert strip_tool_lines(empty) == ""


def test_never_empty_when_other_text_remains():
    assert strip_tool_lines(f"{MORE}\nXo'p") == "Xo'p"


# ---------------------------------------------------------------- the helper vs real tool output

@pytest.fixture
def catalog(monkeypatch, shoe_catalog):
    monkeypatch.setattr(search, "PRODUCTS", shoe_catalog)
    monkeypatch.setattr(search, "_by_id", {p.id: p for p in shoe_catalog})
    monkeypatch.setattr(search, "reload_if_changed", lambda: False)
    return shoe_catalog


def _find(offset: int) -> str:
    return tools.find_products_tool.call({"category": "poyabzal", "size": "42", "offset": offset})


@pytest.mark.parametrize("offset", [0, 5, 15, 20])
def test_real_page_loses_exactly_its_last_line(catalog, offset):
    out = _find(offset)
    body, last = out.rsplit("\n", 1)
    assert last.startswith("ko'rsatildi ")              # sanity: this is the #25 range line
    assert strip_tool_lines(out) == body
    wrapped = f"{INTRO}\n{out}\n{TOTAL_UZ}"
    assert strip_tool_lines(wrapped) == f"{INTRO}\n{body}\n{TOTAL_UZ}"


def test_real_page_with_offer_line_keeps_it(catalog):
    out = _find(15)                                       # items 16–20 include the unpriced id 2004
    assert OFFER in out.split("\n")
    assert OFFER in strip_tool_lines(out).split("\n")


def test_real_past_the_end_line_removed(catalog):
    out = _find(30)
    assert out.startswith("boshqa natija yo'q")
    assert strip_tool_lines(f"{out}\n{TOTAL_UZ}") == TOTAL_UZ


def test_real_latest_posts_page(catalog):
    out = tools.latest_posts_tool.call({"n": 5, "offset": 0})
    body = out.rsplit("\n", 1)[0]
    assert strip_tool_lines(out) == body
    _no_leak(strip_tool_lines(out))


# ---------------------------------------------------------------- end to end through handle_customer

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_PATH", tmp_path / "log.jsonl")
    monkeypatch.setenv("TG_OWNER_ID", str(OWNER))
    monkeypatch.setattr(bot, "_captions", {})

    async def media_for(client, ids):
        return {i: f"photo-{i}" for i in ids}
    monkeypatch.setattr(bot, "_media_for", media_for)
    return tmp_path


@pytest.fixture
def agent_reply(monkeypatch):
    """Replace agent.run_agent with one returning `agent_reply.text`."""
    def fake_run_agent(chat_id, text, *args, **kw):
        agent.last_run = {"tools": [], "escalated": False, "usd": 0.0, "llm_calls": 1}
        return fake_run_agent.text
    fake_run_agent.text = ""
    monkeypatch.setattr(agent, "run_agent", fake_run_agent)
    monkeypatch.setattr(agent, "last_run", {"tools": [], "escalated": False, "usd": 0.0})
    return fake_run_agent


def _button_texts(rows) -> list[str]:
    return [b.text for row in rows for b in row]


def test_carousel_reply_has_no_tool_line(env, agent_reply):
    agent_reply.text = "\n".join([INTRO, S1, S2, S3, OFFER, LAST_PAGE])
    ev = FakeEvent("600 ming gacha sviter bormi?", "uz")
    asyncio.run(bot.handle_customer(ev))
    assert ev.replies == []
    assert len(ev.client.files) == 1
    sent = ev.client.files[0]
    _no_leak(sent["caption"])
    assert sent["caption"].startswith("1. Sviter oversize\n")
    texts = _button_texts(sent["buttons"])
    _no_leak(" ".join(texts))
    assert "1/3" in texts
    kept = bot._captions[7001]                           # ◀/▶ rebuild cards from this reply
    _no_leak(kept)
    assert kept == "\n".join([INTRO, S1, S2, S3, OFFER])


def test_carousel_second_page_keeps_its_numbering(env, agent_reply):
    agent_reply.text = "\n".join(["Yana 5 tasi:", *P6, MORE, TOTAL_UZ])
    ev = FakeEvent("boshqalari bormi?", "uz")
    asyncio.run(bot.handle_customer(ev))
    sent = ev.client.files[0]
    assert sent["caption"].startswith("6. Krossovka model 6\n")
    assert "6/10" in _button_texts(sent["buttons"])
    kept = bot._captions[7001]
    _no_leak(kept)
    assert [ln.split(".")[0] for ln in kept.split("\n") if ln[:1].isdigit()] == ["6", "7", "8", "9", "10"]
    assert TOTAL_UZ in kept


def test_plain_text_reply_has_no_tool_line(env, agent_reply, monkeypatch):
    async def no_media(client, ids):
        return {}
    monkeypatch.setattr(bot, "_media_for", no_media)
    agent_reply.text = "\n".join([INTRO, S1, S2, S3, OFFER, "", LAST_PAGE])
    ev = FakeEvent("600 ming gacha sviter bormi?", "uz")
    asyncio.run(bot.handle_customer(ev))
    assert ev.client.files == []
    assert ev.replies == ["\n".join([INTRO, S1, S2, S3, OFFER])]


def test_plain_text_second_page_keeps_numbering_and_total(env, agent_reply, monkeypatch):
    async def no_media(client, ids):
        return {}
    monkeypatch.setattr(bot, "_media_for", no_media)
    agent_reply.text = "\n".join(["Yana 5 tasi:", *P6, MORE, TOTAL_UZ])
    ev = FakeEvent("boshqalari bormi?", "uz")
    asyncio.run(bot.handle_customer(ev))
    assert ev.replies == ["\n".join(["Yana 5 tasi:", *P6, TOTAL_UZ])]


def test_reply_without_tool_line_reaches_customer_unchanged(env, agent_reply, monkeypatch):
    async def no_media(client, ids):
        return {}
    monkeypatch.setattr(bot, "_media_for", no_media)
    agent_reply.text = "\n".join([TOTAL_RU, S1, S3])
    ev = FakeEvent("есть свитеры?", "ru")
    asyncio.run(bot.handle_customer(ev))
    assert ev.replies == ["\n".join([TOTAL_RU, S1, S3])]


class OwnerReply:
    """Owner's NewMessage replying to an #esc message."""

    def __init__(self, text: str, quoted: str):
        self.raw_text = text
        self.sender_id = OWNER
        self.chat_id = OWNER
        self.is_private = True
        self.is_reply = True
        self._quoted = quoted
        self.replies: list[str] = []

    async def get_reply_message(self):
        return type("Msg", (), {"raw_text": self._quoted})()

    async def reply(self, text, **kw):
        self.replies.append(text)


def test_owner_relay_is_untouched(monkeypatch):
    tg = FakeTelegram()
    monkeypatch.setattr(bot, "_client", lambda: tg)
    monkeypatch.setattr(faq, "add", lambda q, a, post_ids=(), ts=None: FaqEntry("2026-09-24T10:00:00", q, a, ()))
    owner_text = "Hammasi ko'rsatildi.\nko'rsatildi 1–3, jami 3; boshqa yo'q"
    esc = bot.format_escalation(CUSTOMER, "Sviter hali bormi?", ["https://t.me/example_shop/1401"])
    asyncio.run(bot.handle_owner_reply(OwnerReply(owner_text, esc)))
    assert tg.messages == [(CUSTOMER, owner_text)]


# ---------------------------------------------------------------- system prompt

_DONT_COPY = re.compile(r"\b(?:never|do not|don't|must not|not to)\s+(?:copy|show|paste|quote|include|write)\b",
                        re.IGNORECASE)


@pytest.mark.parametrize("code", ["uz_latn", "uz_cyrl", "ru"])
def test_prompt_says_the_paging_line_is_not_for_the_customer(code):
    prompt = agent.system_prompt(code)
    rules = [ln for ln in prompt.split("\n") if "ko'rsatildi" in ln and _DONT_COPY.search(ln)]
    assert rules, f"no rule says the ko'rsatildi line must not be copied ({code})"
    assert any("total" in r.lower() for r in rules), "the same rule should say to tell the customer the total"
