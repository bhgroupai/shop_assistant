"""Ticket #24 — every message to a customer in that customer's language (FR-13a, SDD §3.7, §3.8).

Contract (lang.py contract is in its module docstring):

- bot.handle_customer resolves the language with lang.resolve(event.chat_id, text, lang_code) before
  anything else; lang_code comes from the Telegram sender (`await event.get_sender()` / `event.sender`,
  field `lang_code`); an event without a sender means lang_code None.
- '/start' (also '/start@bot') from a customer → one reply lang.TEXTS[l]["greeting"], no agent / LLM call.
- Every other message → agent.run_agent(chat_id, text, lang=l) (keyword argument).
- carousel_caption(reply, current, lang="uz_latn"), carousel_buttons(idx, ids, ask_price, lang="uz_latn"):
  labels, stale note and button texts from lang.TEXTS[lang]. The ask-price button also appears when the
  item line carries lang.TEXTS[lang]["ask_price"] (the model copies that phrase, see the system prompt).
- Error reply, escalation reply ("owner will answer"), price-callback toast: lang.TEXTS[l][...] for the
  chat's stored language (lang.get_store()); a chat with nothing stored gets uz_latn.
- A ◀ ▶ press after a restart reads the language from the store on disk.
- agent.run_agent(chat_id, text, history=None, lang="uz_latn"): the system instruction carries exactly one
  of "Answer in Russian" / "Answer in Uzbek, Latin script" / "Answer in Uzbek, Cyrillic script", and the
  phrases the model must copy (TEXTS[lang]["ask_price"], TEXTS[lang]["offer_price"]) in that language.
- Owner-facing texts (#esc header, OWNER_HINT_*) stay Uzbek.
Written by the senior — do not edit.
"""
import asyncio
import json

import pytest

from shop_assistant import agent, bot, config, lang, llm
from shop_assistant.agent import History
from tests.fake_gemini import FakeClient, dump, text_response

OWNER = 999
CUSTOMER = 4242
IDS = [983, 950]
ANSWER_LINES = {"ru": "Answer in Russian",
                "uz_latn": "Answer in Uzbek, Latin script",
                "uz_cyrl": "Answer in Uzbek, Cyrillic script"}


def T(code: str, key: str) -> str:
    return lang.TEXTS[code][key]


# ---------------------------------------------------------------- fakes

class Sender:
    def __init__(self, lang_code):
        self.lang_code = lang_code


class FakeTelegram:
    """Telethon client stand-in: records send_file / send_message."""

    def __init__(self):
        self.files: list[dict] = []
        self.messages: list[tuple] = []

    async def send_file(self, chat_id, file, caption=None, buttons=None, reply_to=None, **kw):
        self.files.append({"chat_id": chat_id, "file": file, "caption": caption, "buttons": buttons})
        return type("Msg", (), {"id": 7000 + len(self.files), "photo": None})()

    async def send_message(self, to, text, **kw):
        self.messages.append((to, text))


class FakeEvent:
    """Telethon NewMessage stand-in for a private customer chat."""

    def __init__(self, text: str, lang_code: str | None = "uz", chat_id: int = CUSTOMER):
        self.raw_text = text
        self.sender_id = chat_id
        self.chat_id = chat_id
        self.is_private = True
        self.is_reply = False
        self.message = type("Msg", (), {"id": 1})()
        self.sender = Sender(lang_code)
        self.client = FakeTelegram()
        self.replies: list[str] = []

    async def get_sender(self):
        return self.sender

    async def reply(self, text, **kwargs):
        self.replies.append(text)


class FakeCallback:
    """Telethon CallbackQuery stand-in."""

    def __init__(self, data: bytes, sender_id: int = CUSTOMER):
        self.data = data
        self.sender_id = sender_id
        self.chat_id = sender_id
        self.message_id = 555
        self.client = FakeTelegram()
        self.answers: list = []
        self.edits: list[dict] = []

    async def answer(self, message=None, **kw):
        self.answers.append(message)

    async def edit(self, text=None, file=None, buttons=None, **kw):
        self.edits.append({"text": text, "file": file, "buttons": buttons})
        return type("Msg", (), {"id": self.message_id, "photo": None})()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temp log + language store, owner id, no carousel cache, fake media for every post."""
    monkeypatch.setattr(config, "LOG_PATH", tmp_path / "log.jsonl")
    monkeypatch.setattr(config, "LANGUAGES_PATH", tmp_path / "languages.json")
    monkeypatch.setenv("TG_OWNER_ID", str(OWNER))
    monkeypatch.setattr(bot, "_captions", {})

    async def media_for(client, ids):
        return {i: f"photo-{i}" for i in ids}
    monkeypatch.setattr(bot, "_media_for", media_for)
    return tmp_path


def _stored(env, mapping: dict) -> None:
    """Languages already on disk, as after a restart (nothing in memory yet)."""
    (env / "languages.json").write_text(json.dumps({str(k): v for k, v in mapping.items()}), encoding="utf-8")


@pytest.fixture
def agent_calls(monkeypatch):
    """Replace agent.run_agent; records (text, lang) and answers `agent_calls.reply`."""
    class Calls(list):
        fake = None
    calls = Calls()

    def fake_run_agent(chat_id, text, *args, **kw):
        calls.append((text, kw.get("lang")))
        agent.last_run = {"tools": [], "escalated": False, "usd": 0.0, "llm_calls": 1}
        return fake_run_agent.reply
    fake_run_agent.reply = "Qaysi mahsulot kerak?"
    monkeypatch.setattr(agent, "run_agent", fake_run_agent)
    monkeypatch.setattr(agent, "last_run", {"tools": [], "escalated": False, "usd": 0.0})
    calls.fake = fake_run_agent
    return calls


def _texts(rows) -> list[str]:
    return [b.text for row in rows for b in row]


# ---------------------------------------------------------------- /start

@pytest.mark.parametrize("lang_code, expected", [("ru", "ru"), ("uz", "uz_latn"), ("en", "uz_latn"), (None, "uz_latn")])
def test_start_is_answered_with_localized_greeting_and_no_agent(env, agent_calls, lang_code, expected):
    greeting = T(expected, "greeting")
    ev = FakeEvent("/start", lang_code)
    asyncio.run(bot.route(ev, OWNER))
    assert ev.replies == [greeting]
    assert agent_calls == []


def test_start_with_bot_suffix_is_also_a_greeting(env, agent_calls):
    greeting = T("ru", "greeting")
    ev = FakeEvent("/start@example_shop_bot", "ru")
    asyncio.run(bot.handle_customer(ev))
    assert ev.replies == [greeting]
    assert agent_calls == []


def test_stored_language_wins_over_lang_code_on_later_start(env, agent_calls):
    """Acceptance: Telegram in Russian, customer writes Uzbek → next /start is Uzbek."""
    greeting = T("uz_latn", "greeting")
    asyncio.run(bot.handle_customer(FakeEvent("krossovka 42 bormi?", "ru")))
    assert agent_calls == [("krossovka 42 bormi?", "uz_latn")]
    ev = FakeEvent("/start", "ru")
    asyncio.run(bot.handle_customer(ev))
    assert ev.replies == [greeting]


def test_start_after_restart_reads_the_store(env, agent_calls):
    greeting = T("ru", "greeting")
    _stored(env, {CUSTOMER: "ru"})
    ev = FakeEvent("/start", "uz")
    asyncio.run(bot.handle_customer(ev))
    assert ev.replies == [greeting]


# ---------------------------------------------------------------- language passed to the agent

@pytest.mark.parametrize("text, expected", [("туфли 43 размер есть?", "ru"),
                                            ("кроссовка борми", "uz_cyrl"),
                                            ("dvoyka narxi qancha?", "uz_latn")])
def test_run_agent_receives_the_detected_language(env, agent_calls, text, expected):
    assert lang.detect(text) == expected
    asyncio.run(bot.handle_customer(FakeEvent(text, "uz" if expected == "ru" else "ru")))
    assert agent_calls == [(text, expected)]


def test_short_message_keeps_the_stored_language(env, agent_calls):
    lang.get_store().set(CUSTOMER, "ru")
    asyncio.run(bot.handle_customer(FakeEvent("2", "uz")))
    assert agent_calls == [("2", "ru")]


# ---------------------------------------------------------------- carousel texts

REPLY_TWO = ("1. Krossovka Nike Air · 350000 · 40,41,42 · 2026-09-12 · https://t.me/example_shop/983\n"
             "2. Dvoyka · narxi: so'rab beraman · M,L · 2026-09-13 · https://t.me/example_shop/950 · [eskirgan]")


@pytest.mark.parametrize("code", ["uz_latn", "uz_cyrl", "ru"])
def test_buttons_in_chat_language(code):
    ask, view = T(code, "ask_price_button"), T(code, "view_in_channel_button")
    rows = bot.carousel_buttons(1, IDS, ask_price=True, lang=code)
    assert [b.text for b in rows[-1]] == [ask, view]
    assert rows[-1][1].type.url.endswith("/example_shop/950")
    assert [b.text for b in rows[0]] == ["◀", "2/2", "▶"]


@pytest.mark.parametrize("code", ["uz_cyrl", "ru"])
def test_caption_labels_and_stale_note_in_chat_language(code):
    price, stale = T(code, "price_label"), T(code, "stale_note")
    card = bot.carousel_caption(REPLY_TWO, 0, lang=code)
    assert card.startswith("1. Krossovka Nike Air\n")
    assert price in card and "350 000" in card
    assert "Narxi:" not in card and "O'lcham:" not in card and "Sana:" not in card
    stale_card = bot.carousel_caption(REPLY_TWO, 1, lang=code)
    assert stale in stale_card
    assert bot.STALE_NOTE not in stale_card
    assert "so'rab" not in stale_card


def test_caption_default_stays_uzbek_latin():
    stale = T("uz_latn", "stale_note")
    card = bot.carousel_caption(REPLY_TWO, 1)
    assert "Narxi: so'rab beraman" in card and stale in card


def test_customer_reply_carousel_in_russian(env, agent_calls):
    ask, view, price = T("ru", "ask_price_button"), T("ru", "view_in_channel_button"), T("ru", "price_label")
    agent_calls.fake.reply = (
        "Вот что нашлось:\n"
        f"1. Кроссовки Nike Air · {T('ru', 'ask_price')} · 40,41,42 · 2026-09-12 · https://t.me/example_shop/983\n"
        "2. Двойка · 980000 · M,L · 2026-09-13 · https://t.me/example_shop/950\n"
        f"{T('ru', 'offer_price')}")
    ev = FakeEvent("есть кроссовки 42 размера?", "uz")
    asyncio.run(bot.handle_customer(ev))
    assert agent_calls == [("есть кроссовки 42 размера?", "ru")]
    assert len(ev.client.files) == 1
    sent = ev.client.files[0]
    assert price in sent["caption"] and "Narxi:" not in sent["caption"]
    assert _texts(sent["buttons"])[-2:] == [ask, view]


# ---------------------------------------------------------------- fixed replies

def test_error_reply_in_chat_language(env, agent_calls, monkeypatch):
    error = T("ru", "error_reply")

    def broken(*a, **kw):
        raise RuntimeError("tool crashed")
    monkeypatch.setattr(agent, "run_agent", broken)
    ev = FakeEvent("сколько стоит двойка?", "uz")
    asyncio.run(bot.handle_customer(ev))
    assert ev.replies == [error]


@pytest.mark.parametrize("stored, expected", [("ru", "ru"), ("uz_cyrl", "uz_cyrl"), (None, "uz_latn")])
def test_escalation_reply_in_customer_language(env, monkeypatch, stored, expected):
    reply = T(expected, "escalated_reply")
    if stored:
        _stored(env, {CUSTOMER: stored})
    client = FakeTelegram()
    monkeypatch.setattr(bot, "_client", lambda: client)
    asyncio.run(bot.escalate(CUSTOMER, "Qishki kurtka hali bormi?", [900]))
    to_owner = [t for to, t in client.messages if to == OWNER]
    to_customer = [t for to, t in client.messages if to == CUSTOMER]
    assert len(to_owner) == 1 and to_owner[0].startswith(f"{bot.ESC_PREFIX} {CUSTOMER}\n")   # owner side unchanged
    assert to_customer == [reply]


def test_owner_hints_stay_uzbek():
    assert bot.OWNER_HINT_NOT_REPLY == "Mijozga javob berish uchun #esc xabariga reply qiling."
    assert bot.OWNER_HINT_NOT_ESC == "Bu #esc xabari emas."


# ---------------------------------------------------------------- callbacks after a restart

def test_price_callback_toast_in_stored_language(env, monkeypatch):
    toast = T("ru", "callback_toast")
    _stored(env, {CUSTOMER: "ru"})
    escalated: list = []

    async def fake_escalate(customer_id, question, post_ids):
        escalated.append((customer_id, post_ids))
    monkeypatch.setattr(bot, "escalate", fake_escalate)
    ev = FakeCallback(b"p:1:983,950")
    asyncio.run(bot.handle_callback(ev))
    assert escalated == [(CUSTOMER, [950])]
    assert ev.answers == [toast]


def test_navigation_after_restart_uses_stored_language(env, monkeypatch):
    ask, view, price = (T("uz_cyrl", "ask_price_button"), T("uz_cyrl", "view_in_channel_button"),
                        T("uz_cyrl", "price_label"))
    _stored(env, {CUSTOMER: "uz_cyrl"})
    monkeypatch.setattr(bot, "_rebuild_reply", lambda ids: REPLY_TWO)   # catalog rebuild after restart
    ev = FakeCallback(b"c:1:983,950")
    asyncio.run(bot.handle_callback(ev))
    assert len(ev.edits) == 1
    edit = ev.edits[0]
    assert edit["text"].startswith("2. Dvoyka\n")
    assert price in edit["text"] and T("uz_cyrl", "stale_note") in edit["text"]
    assert _texts(edit["buttons"])[-2:] == [ask, view]


# ---------------------------------------------------------------- agent system prompt

@pytest.fixture
def gemini(monkeypatch):
    fake = FakeClient([text_response("Assalomu alaykum!")])
    monkeypatch.setattr(llm, "client", lambda: fake)
    return fake


def _in(phrase: str, blob: str) -> bool:
    return json.dumps(phrase, ensure_ascii=False)[1:-1] in blob


def _instruction(fake) -> str:
    return dump(fake.calls[0]["config"].system_instruction)


@pytest.mark.parametrize("code", ["ru", "uz_latn", "uz_cyrl"])
def test_system_prompt_has_one_explicit_answer_line(gemini, code):
    agent.run_agent(601, "Salom", History(), lang=code)
    blob = _instruction(gemini)
    assert _in(ANSWER_LINES[code], blob)
    for other, line in ANSWER_LINES.items():
        if other != code:
            assert not _in(line, blob), other
    assert "exactly as they wrote" not in blob


@pytest.mark.parametrize("code", ["ru", "uz_latn", "uz_cyrl"])
def test_system_prompt_fixed_phrases_in_that_language(gemini, code):
    ask, offer = T(code, "ask_price"), T(code, "offer_price")
    agent.run_agent(602, "Salom", History(), lang=code)
    blob = _instruction(gemini)
    assert _in(ask, blob) and _in(offer, blob)


def test_run_agent_default_language_is_uzbek_latin(gemini):
    agent.run_agent(603, "Salom", History())
    assert _in(ANSWER_LINES["uz_latn"], _instruction(gemini))
