"""Ticket #23.7 — FAQ store: owner answers are saved, embedded and found again (SDD §2.4, §3.6-3.8, FR-22).
Fake Gemini client only, no network (tests/fake_gemini.py). Written by the senior — do not edit.

CONTRACT (read before implementing):

Files (paths read from config AT CALL TIME; tests point them at a tmp dir, see conftest `_faq_files`):
* config.FAQ_PATH (data/faq.jsonl): one JSON object per line {"ts", "question", "answer", "post_ids"}
  (post_ids a list of ints), append-only, UTF-8. Parent dir created when missing.
* config.FAQ_EMBEDDINGS_PATH (data/faq_embeddings.npy): float32[n, config.EMBED_DIM]; row i is the
  embedding of the QUESTION of line i of faq.jsonl (lines 0..n-1 are embedded, the rest are "pending").
* NEW config.FAQ_META_PATH = config.DATA_DIR / "faq_embeddings_meta.json": {"model", "dim", "n"} of that
  matrix — the same model-name guard as the product matrix (search._load_matrix).

shop_assistant/faq.py (new module; reuses index.embed and search.cosine_top_k by calling them —
index.py / search.py are not changed by this ticket):
* THRESHOLD: float in (0.3, 0.95) — minimum cosine(question, query) for a hit.
* load() -> list[FaqEntry] in file order (post_ids as a tuple). Missing file -> [].
* add(question, answer, post_ids=(), ts=None) -> FaqEntry. Appends the line FIRST (ts default: now,
  ISO 8601), then embeds every pending entry: ONE index.embed([...], kind="document") call with
  retry=False (it runs inside the bot; never time.sleep), rewrites the .npy and the meta
  {"model": config.EMBED_MODEL, "dim": config.EMBED_DIM, "n": <rows>}.
  Any embedding failure (429, 5xx, timeout, bad key) -> logged, NEVER raised; the line stays saved and is
  embedded by the next add()/search() whose embedding succeeds.
* search(query, k=3) -> list[FaqEntry], best first, at most k, only entries with cosine >= THRESHOLD.
  First tries to embed pending entries (failure -> carry on with the rows that exist). The query goes
  through index.embed([...], kind="query"); failure -> [] and a WARNING, never raises.
  No embedded entries -> [] and NO embedding request at all.
  Meta missing while an .npy exists, or meta model != config.EMBED_MODEL, or dim != config.EMBED_DIM ->
  exactly one ERROR on the "shop_assistant.faq" logger per search, [] and NO request; add() still saves the
  line but never appends new-model rows to an old-model matrix (vectors of two models are never mixed).
* reindex() -> int: re-embeds ALL entries with the current model (retry=True allowed, offline) and rewrites
  matrix + meta; the fix after a model change. `python -m shop_assistant.faq` runs it.

tools.search_faq_tool(text) (name, docstring and schema unchanged — tests/tool_declarations.json):
* calls faq.search(text, ...) through the module attribute (`from shop_assistant import faq`; tests
  monkeypatch faq.search), not search.search_faq. Hits -> one line per entry
  "<question> — <owner_said>: <answer>", where owner_said = lang.TEXTS[l]["owner_said"] and l is the
  language stored for the current chat (lang.get_store().get(agent.current_chat_id.get()), default uz_latn).
  uz_latn value is exactly "egasi aytgan". Never the product format: no numbering "1. ", no " · ", no t.me
  link, even when the entry has post_ids. No hits -> tools.NO_FAQ.
* lang.TEXTS gains the key "owner_said" in every language (tests/test_lang.py KEYS).

agent.system_prompt(lang) for every language contains the literal rule "search_faq_tool before ask_owner"
(delivery / payment / shop questions: look in the FAQ first, escalate only when it has no answer) and
mentions lang.TEXTS[lang]["owner_said"] (FAQ answers are presented as what the owner said, never as
catalog data).

bot.handle_owner_reply: after the owner's text was sent to the customer, calls
faq.add(question, answer, post_ids) with question = the quoted #esc message without its header line and
without its t.me link lines (stripped), answer = the owner's text, post_ids = bot.post_ids_in(quoted text).
faq.add is looked up on the module at call time (tests monkeypatch it); it may run via asyncio.to_thread.
A faq.add exception is logged and never blocks the relay (customer message and "✓ yuborildi" still go out).
Not a reply / not an #esc message -> no faq.add.
"""
import asyncio
import datetime
import json
import logging
import subprocess
import sys
import time

import numpy as np
import pytest
from google.genai import types

from shop_assistant import agent, bot, config, faq, lang, llm, tools
from shop_assistant.agent import History
from shop_assistant.models import FaqEntry
from tests.fake_gemini import (FakeClient, call_response, dump, embed_config, invalid_key_error, quota_error,
                               server_error, text_response, timeout_error)

OWNER = 999
CUSTOMER = 4242
QUESTION = "dastavka Samarqandga qancha?"
ANSWER = "Samarqandga dastavka 35 ming so'm, 2 kunda yetkazamiz"
PAY_Q = "Click yoki Payme bilan to'lasa bo'ladimi?"
PAY_A = "Ha, Click va Payme ikkalasi ham bor, karta raqamini yozib beraman"
HOURS_Q = "Do'kon soat nechada ochiladi?"
HOURS_A = "Har kuni 9:00 dan 20:00 gacha, yakshanba ham ishlaymiz"
UNRELATED = "qishki kurtka qora rangi bormi 48 razmer"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """The bot path must never sleep-retry; any time.sleep in these tests is recorded."""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    return sleeps


@pytest.fixture
def gemini(monkeypatch):
    """Install a FakeClient as llm.client(); returns a factory so a test can script embed answers."""
    holder = {}

    def install(script=(), embed_script=None) -> FakeClient:
        holder["fake"] = FakeClient(script, embed_script)
        return holder["fake"]

    monkeypatch.setattr(llm, "client", lambda: holder["fake"])
    install()
    return install


def _task_types(fake: FakeClient) -> list[str]:
    return [embed_config(c["config"]).get("task_type") for c in fake.embed_calls]


def _lines() -> list[dict]:
    return [json.loads(ln) for ln in config.FAQ_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _matrix() -> np.ndarray:
    return np.load(config.FAQ_EMBEDDINGS_PATH)


def _meta() -> dict:
    return json.loads(config.FAQ_META_PATH.read_text(encoding="utf-8"))


def _vector_response(vec) -> types.EmbedContentResponse:
    return types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[float(x) for x in vec])])


# ---------------------------------------------------------------- storage

def test_empty_store_has_no_entries_and_search_sends_no_request(gemini):
    fake = gemini()
    assert not config.FAQ_PATH.exists()
    assert faq.load() == []
    assert faq.search(QUESTION) == []
    assert fake.embed_calls == []


def test_add_appends_one_json_line_with_the_four_fields(gemini):
    gemini()
    entry = faq.add(QUESTION, ANSWER, [1234], ts="2026-09-23T10:15:00")
    assert entry == FaqEntry(ts="2026-09-23T10:15:00", question=QUESTION, answer=ANSWER, post_ids=(1234,))
    assert _lines() == [{"ts": "2026-09-23T10:15:00", "question": QUESTION, "answer": ANSWER, "post_ids": [1234]}]
    faq.add(PAY_Q, PAY_A, [])
    assert [ln["question"] for ln in _lines()] == [QUESTION, PAY_Q]


def test_add_default_ts_is_now_iso(gemini):
    gemini()
    before = datetime.datetime.now() - datetime.timedelta(minutes=1)
    entry = faq.add(HOURS_Q, HOURS_A)
    ts = datetime.datetime.fromisoformat(entry.ts)
    assert before <= ts <= datetime.datetime.now() + datetime.timedelta(minutes=1)
    assert _lines()[0]["post_ids"] == []


def test_load_round_trips_cyrillic_and_keeps_file_order(gemini):
    gemini()
    faq.add(QUESTION, ANSWER, [1234], ts="2026-09-23T10:15:00")
    faq.add("Доставка в Ташкент сколько стоит?", "По Ташкенту 25 000 сум, на следующий день", [],
            ts="2026-09-23T11:00:00")
    entries = faq.load()
    assert [e.question for e in entries] == [QUESTION, "Доставка в Ташкент сколько стоит?"]
    assert entries[1].answer == "По Ташкенту 25 000 сум, на следующий день"
    assert entries[0].post_ids == (1234,)


def test_add_embeds_the_question_as_a_document_and_writes_meta(gemini):
    fake = gemini()
    faq.add(QUESTION, ANSWER, [1234])
    assert _task_types(fake) == ["RETRIEVAL_DOCUMENT"]
    assert fake.embed_calls[0]["model"] == config.EMBED_MODEL
    assert len(fake.embed_calls[0]["contents"]) == 1
    m = _matrix()
    assert m.shape == (1, config.EMBED_DIM)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-4)   # index.embed rows, L2-normalised
    assert _meta() == {"model": config.EMBED_MODEL, "dim": config.EMBED_DIM, "n": 1}
    faq.add(PAY_Q, PAY_A)
    assert _matrix().shape == (2, config.EMBED_DIM)
    assert _meta()["n"] == 2
    assert len(fake.embed_calls[1]["contents"]) == 1   # only the new (pending) entry is embedded


def test_meta_path_is_in_config():
    out = subprocess.run([sys.executable, "-c",
                          "from shop_assistant import config; print(config.FAQ_META_PATH.relative_to(config.DATA_DIR))"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "faq_embeddings_meta.json"


# ---------------------------------------------------------------- search

def test_same_question_finds_the_owner_answer_with_a_query_embedding(gemini):
    fake = gemini()
    faq.add(QUESTION, ANSWER, [1234])
    faq.add(PAY_Q, PAY_A)
    faq.add(HOURS_Q, HOURS_A)
    n_before = len(fake.embed_calls)
    hits = faq.search(QUESTION)
    assert [h.answer for h in hits] == [ANSWER]
    assert _task_types(fake)[n_before:] == ["RETRIEVAL_QUERY"]


def test_unrelated_query_is_below_the_threshold(gemini):
    gemini()
    faq.add(QUESTION, ANSWER, [1234])
    faq.add(PAY_Q, PAY_A)
    assert faq.search(UNRELATED) == []


def test_threshold_is_a_cosine_cut(gemini):
    assert 0.3 < faq.THRESHOLD < 0.95
    dim = config.EMBED_DIM
    e1, e2 = np.zeros(dim), np.zeros(dim)
    e1[0], e2[1] = 1.0, 1.0

    def query_at(c):
        return _vector_response(c * e1 + np.sqrt(1 - c * c) * e2)

    above, below = min(faq.THRESHOLD + 0.05, 1.0), faq.THRESHOLD - 0.05
    gemini(embed_script=[_vector_response(3.0 * e1), query_at(above), query_at(below)])
    faq.add(QUESTION, ANSWER, [1234])
    assert [h.question for h in faq.search("Samarqandga yetkazib berish narxi")] == [QUESTION]
    assert faq.search("Samarqandga yetkazib berish narxi") == []


def test_search_returns_best_first_and_at_most_k(gemini):
    dim = config.EMBED_DIM
    rng = np.random.default_rng(7)
    base = rng.normal(size=dim)
    base /= np.linalg.norm(base)
    noise = [rng.normal(size=dim) for _ in range(3)]
    docs = [base + s * n / np.linalg.norm(n) for s, n in zip((0.30, 0.05, 0.15), noise)]
    gemini(embed_script=[_vector_response(d) for d in docs] + [_vector_response(base), _vector_response(base)])
    faq.add("Samarqandga dastavka bormi?", "Bor, 35 ming", ts="2026-09-20T09:00:00")
    faq.add(QUESTION, ANSWER, [1234], ts="2026-09-21T09:00:00")
    faq.add("Samarqand viloyatiga yetkazish qancha turadi?", "35 ming, tumanlarga 45 ming", ts="2026-09-22T09:00:00")
    assert [h.answer for h in faq.search(QUESTION, k=3)] == [ANSWER, "35 ming, tumanlarga 45 ming", "Bor, 35 ming"]
    assert [h.answer for h in faq.search(QUESTION, k=1)] == [ANSWER]


@pytest.mark.parametrize("error", [quota_error, server_error, timeout_error, invalid_key_error])
def test_query_embedding_failure_gives_empty_and_a_warning(gemini, caplog, error):
    gemini()
    faq.add(QUESTION, ANSWER, [1234])
    gemini(embed_script=[error()])
    with caplog.at_level(logging.WARNING, logger="shop_assistant.faq"):
        assert faq.search(QUESTION) == []
    assert any(r.levelno == logging.WARNING and r.name == "shop_assistant.faq" for r in caplog.records)


# ---------------------------------------------------------------- failure -> recovery (acceptance 4)

@pytest.mark.parametrize("error", [quota_error, server_error, timeout_error, invalid_key_error])
def test_embedding_failure_never_raises_and_the_entry_is_saved(gemini, error, _no_sleep):
    fake = gemini(embed_script=[error()])
    entry = faq.add(QUESTION, ANSWER, [1234])
    assert entry.answer == ANSWER
    assert _lines()[0]["answer"] == ANSWER
    assert not config.FAQ_EMBEDDINGS_PATH.exists() or _matrix().shape[0] == 0
    assert len(fake.embed_calls) == 1   # retry=False: one attempt, no sleeping in the bot
    assert _no_sleep == []


def test_pending_entry_is_embedded_by_the_next_successful_add(gemini):
    fake = gemini(embed_script=[quota_error()])
    faq.add(QUESTION, ANSWER, [1234])
    faq.add(PAY_Q, PAY_A)
    assert _matrix().shape == (2, config.EMBED_DIM)
    assert _meta()["n"] == 2
    assert len(fake.embed_calls[-1]["contents"]) == 2
    assert [h.answer for h in faq.search(QUESTION)] == [ANSWER]


def test_pending_entry_is_embedded_by_the_next_search(gemini):
    fake = gemini(embed_script=[quota_error()])
    faq.add(QUESTION, ANSWER, [1234])
    assert [h.answer for h in faq.search(QUESTION)] == [ANSWER]
    assert _task_types(fake) == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert _meta()["n"] == 1


def test_search_still_works_when_pending_embedding_fails(gemini):
    gemini()
    faq.add(QUESTION, ANSWER, [1234])
    gemini(embed_script=[quota_error(), quota_error()])   # add's embedding fails, then search's pending one
    faq.add(PAY_Q, PAY_A)                                  # saved, not embedded
    assert [h.answer for h in faq.search(QUESTION)] == [ANSWER]   # the query itself gets through
    assert _meta()["n"] == 1
    assert len(faq.load()) == 2


# ---------------------------------------------------------------- model guard (acceptance 3)

def _old_model_store():
    """A store built by another embedding model: 1 entry, 1024-d matrix, meta naming that model."""
    config.FAQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.FAQ_PATH.write_text(json.dumps({"ts": "2026-09-01T12:00:00", "question": QUESTION,
                                           "answer": ANSWER, "post_ids": [1234]}, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    np.save(config.FAQ_EMBEDDINGS_PATH, np.ones((1, 1024), dtype=np.float32) / 32.0)
    config.FAQ_META_PATH.write_text(json.dumps({"model": "bge-m3", "dim": 1024, "n": 1}), encoding="utf-8")


def test_model_mismatch_turns_faq_search_off_with_one_error(gemini, caplog):
    fake = gemini()
    _old_model_store()
    with caplog.at_level(logging.WARNING, logger="shop_assistant.faq"):
        assert faq.search(QUESTION) == []
    errors_ = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == "shop_assistant.faq"]
    assert len(errors_) == 1
    assert "bge-m3" in errors_[0].getMessage() or "1024" in errors_[0].getMessage()
    assert fake.embed_calls == []


def test_npy_without_meta_is_a_mismatch(gemini, caplog):
    fake = gemini()
    _old_model_store()
    config.FAQ_META_PATH.unlink()
    with caplog.at_level(logging.ERROR, logger="shop_assistant.faq"):
        assert faq.search(QUESTION) == []
    assert len([r for r in caplog.records if r.levelno == logging.ERROR and r.name == "shop_assistant.faq"]) == 1
    assert fake.embed_calls == []


def test_add_on_a_mismatched_store_saves_but_never_mixes_models(gemini):
    gemini()
    _old_model_store()
    faq.add(PAY_Q, PAY_A)
    assert [ln["question"] for ln in _lines()] == [QUESTION, PAY_Q]
    assert _matrix().shape == (1, 1024)
    assert _meta()["model"] == "bge-m3"


def test_reindex_rebuilds_with_the_current_model(gemini):
    fake = gemini()
    _old_model_store()
    faq.add(PAY_Q, PAY_A)
    assert faq.reindex() == 2
    assert _matrix().shape == (2, config.EMBED_DIM)
    assert _meta() == {"model": config.EMBED_MODEL, "dim": config.EMBED_DIM, "n": 2}
    assert set(_task_types(fake)) == {"RETRIEVAL_DOCUMENT"}
    assert [h.answer for h in faq.search(QUESTION)] == [ANSWER]


# ---------------------------------------------------------------- search_faq_tool (acceptance 2)

def _hits(monkeypatch, entries):
    calls = []

    def fake_search(query, *args, **kw):
        calls.append(query)
        return entries
    monkeypatch.setattr(faq, "search", fake_search)
    return calls


def _call_tool(text: str) -> str:
    tool = next(t for t in tools.TOOLS if t.name == "search_faq_tool")
    return tool.call({"text": text})


def test_tool_uses_faq_search_and_labels_owner_answers(monkeypatch):
    calls = _hits(monkeypatch, [FaqEntry("2026-09-23T10:15:00", QUESTION, ANSWER, (1234,))])
    out = _call_tool(QUESTION)
    assert calls == [QUESTION]
    assert lang.TEXTS["uz_latn"]["owner_said"] == "egasi aytgan"
    assert out == f"{QUESTION} — egasi aytgan: {ANSWER}"


def test_tool_output_is_never_product_data(monkeypatch):
    _hits(monkeypatch, [FaqEntry("2026-09-23T10:15:00", QUESTION, ANSWER, (1234,)),
                        FaqEntry("2026-09-23T11:00:00", PAY_Q, "Ha, Click va Payme\nikkalasi ham bor", ())])
    out = _call_tool(QUESTION)
    lines = out.splitlines()
    assert len(lines) == 2
    for line in lines:
        assert "egasi aytgan" in line
        assert " · " not in line and "t.me/" not in line
        assert not line[:1].isdigit()
    assert out != tools.NO_FAQ


def test_tool_label_follows_the_chat_language(monkeypatch):
    _hits(monkeypatch, [FaqEntry("2026-09-23T10:15:00", QUESTION, ANSWER, (1234,))])
    lang.get_store().set(CUSTOMER, "ru")
    token = agent.current_chat_id.set(CUSTOMER)
    try:
        out = _call_tool("Сколько доставка в Самарканд?")
    finally:
        agent.current_chat_id.reset(token)
    assert lang.TEXTS["ru"]["owner_said"] in out
    assert "egasi aytgan" not in out


def test_tool_without_hits_says_no_faq(monkeypatch):
    _hits(monkeypatch, [])
    assert _call_tool(UNRELATED) == tools.NO_FAQ


def test_tool_on_an_empty_store(gemini):
    assert _call_tool(QUESTION) == tools.NO_FAQ


# ---------------------------------------------------------------- system prompt

@pytest.mark.parametrize("code", lang.LANGS)
def test_prompt_checks_the_faq_before_escalating(code):
    prompt = agent.system_prompt(code)
    assert "search_faq_tool before ask_owner" in prompt
    assert lang.TEXTS[code]["owner_said"] in prompt


# ---------------------------------------------------------------- bot relay

class FakeTelegram:
    def __init__(self):
        self.messages: list[tuple] = []

    async def send_message(self, to, text, **kw):
        self.messages.append((to, text))


class OwnerReply:
    """Owner's NewMessage replying to an #esc message."""

    def __init__(self, text: str, quoted: str | None, is_reply: bool = True):
        self.raw_text = text
        self.sender_id = OWNER
        self.chat_id = OWNER
        self.is_private = True
        self.is_reply = is_reply
        self._quoted = quoted
        self.replies: list[str] = []

    async def get_reply_message(self):
        return None if self._quoted is None else type("Msg", (), {"raw_text": self._quoted})()

    async def reply(self, text, **kw):
        self.replies.append(text)


@pytest.fixture
def telegram(monkeypatch):
    tg = FakeTelegram()
    monkeypatch.setattr(bot, "_client", lambda: tg)
    return tg


@pytest.fixture
def faq_adds(monkeypatch, telegram):
    """Replace faq.add; records its arguments and whether the customer already had the answer."""
    calls: list[dict] = []

    def fake_add(question, answer, post_ids=(), ts=None):
        calls.append({"question": question, "answer": answer, "post_ids": list(post_ids),
                      "relayed_before": (CUSTOMER, answer) in telegram.messages})
        return FaqEntry("2026-09-23T10:15:00", question, answer, tuple(post_ids))
    monkeypatch.setattr(faq, "add", fake_add)
    return calls


ESC = bot.format_escalation(CUSTOMER, QUESTION, ["https://t.me/example_shop/1234"])


def test_relay_saves_the_answer_after_sending_it(telegram, faq_adds):
    event = OwnerReply(ANSWER, ESC)
    asyncio.run(bot.handle_owner_reply(event))
    assert telegram.messages == [(CUSTOMER, ANSWER)]
    assert event.replies == [bot.OWNER_SENT]
    assert faq_adds == [{"question": QUESTION, "answer": ANSWER, "post_ids": [1234], "relayed_before": True}]


def test_relay_without_links_saves_empty_post_ids(telegram, faq_adds):
    esc = bot.format_escalation(CUSTOMER, HOURS_Q, [])
    asyncio.run(bot.handle_owner_reply(OwnerReply(HOURS_A, esc)))
    assert faq_adds[0]["question"] == HOURS_Q
    assert faq_adds[0]["post_ids"] == []


def test_faq_failure_never_blocks_the_relay(telegram, monkeypatch, caplog):
    def broken_add(*a, **kw):
        raise OSError("No space left on device")
    monkeypatch.setattr(faq, "add", broken_add)
    event = OwnerReply(ANSWER, ESC)
    asyncio.run(bot.handle_owner_reply(event))   # must not raise
    assert telegram.messages == [(CUSTOMER, ANSWER)]
    assert event.replies == [bot.OWNER_SENT]


def test_relay_survives_an_embedding_outage_and_keeps_the_line(telegram, gemini):
    gemini(embed_script=[quota_error()])
    event = OwnerReply(ANSWER, ESC)
    asyncio.run(bot.handle_owner_reply(event))
    assert telegram.messages == [(CUSTOMER, ANSWER)]
    assert event.replies == [bot.OWNER_SENT]
    assert _lines()[0]["question"] == QUESTION and _lines()[0]["post_ids"] == [1234]


@pytest.mark.parametrize("event", [OwnerReply(ANSWER, None, is_reply=False), OwnerReply(ANSWER, "salom")])
def test_no_faq_entry_without_an_escalation(telegram, faq_adds, event):
    asyncio.run(bot.handle_owner_reply(event))
    assert faq_adds == []
    assert telegram.messages == []


# ---------------------------------------------------------------- acceptance 1, end to end with fakes

def test_owner_answer_is_given_to_the_next_customer_without_escalation(telegram, gemini, monkeypatch):
    fake = gemini(script=[call_response("search_faq_tool", {"text": QUESTION}),
                          text_response(f"Egasi aytgan: {ANSWER}")])
    asyncio.run(bot.handle_owner_reply(OwnerReply(ANSWER, ESC)))       # 1st customer's question answered

    escalations = []
    monkeypatch.setattr(bot, "escalate_sync", lambda q, ids: escalations.append((q, ids)))
    reply = agent.run_agent(5151, QUESTION, history=History())         # 2nd customer asks the same
    assert reply == f"Egasi aytgan: {ANSWER}"
    assert [t["name"] for t in agent.last_run["tools"]] == ["search_faq_tool"]
    assert agent.last_run["tools"][0]["n_results"] == 1
    assert agent.last_run["escalated"] is False and escalations == []
    tool_result = dump(fake.calls[1]["contents"])
    assert ANSWER in tool_result and "egasi aytgan" in tool_result
