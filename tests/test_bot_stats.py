"""Ticket #17 — /stats and /reindex owner commands (FR-26, SDD §2.5, §2.6, §3.8).

Contract (all in shop_assistant/bot.py unless noted):

- config.GEMINI_DAILY_LIMIT: int — free daily Gemini request limit shown by /stats.
- log.jsonl records (log_turn) gain an int field `llm_calls`: Gemini requests made in that turn.
  log_turn(..., llm_calls: int = 1) writes it; handle_customer passes llm_calls_for(agent.last_run).
- llm_calls_for(last_run) -> int: last_run["llm_calls"] if it is an int (a later agent may expose it),
  else 1 + len(last_run["tools"]) + number of tools named "semantic_search_tool" (query embedding).
- Ingestion log: config.DATA_DIR / bot.INGEST_LOG_NAME ("gemini_ingest.jsonl"), one JSON line
  {"ts": ISO, ...} per Gemini request made by fetch/extract/index. Written by the ingestion side
  (not this ticket); absent file → 0.
- compute_stats(today, log_path=None, state_path=None, ingest_log_path=None) -> dict with keys
  indexed_posts (len(search.PRODUCTS)), last_index_at (state.json "last_index_at" or None),
  questions_today, escalations_today, gemini_today, gemini_limit. A line counts for `today` when
  its ts date (first 10 chars) equals today.isoformat(). Lines without llm_calls count 1.
  Missing / empty files → zeros / None. Blank lines are skipped.
- format_stats(stats) -> str with lines 'Indexed posts: N', 'Last index: <ts or ->',
  'Questions today: N', 'Escalations today: N', 'Gemini: N / LIMIT today'.
- async handle_stats(event), async handle_reindex(event) (search.reload() then reply),
  async route(event, owner) — the NewMessage dispatcher run() registers; owner-only commands.
"""
import asyncio
import dataclasses
import json
from datetime import date

import pytest

from shop_assistant import bot, config, search

TODAY = date(2026, 9, 23)
OWNER = 999
CUSTOMER = 4242


# ---------------------------------------------------------------- helpers

def _line(ts: str, question: str, escalated: bool = False, llm_calls: int | None = None) -> str:
    rec = {"ts": ts, "chat_id": CUSTOMER, "question": question, "tools": [],
           "answer": "Ha, bor: https://t.me/example_shop/1300", "escalated": escalated,
           "ms": 1200, "usd": 0.0}
    if llm_calls is not None:
        rec["llm_calls"] = llm_calls
    return json.dumps(rec, ensure_ascii=False)


def _write(path, lines: list[str]) -> None:
    path.write_text("".join(l + "\n" for l in lines), encoding="utf-8")


def _product_line(p) -> str:
    return json.dumps(dataclasses.asdict(p), ensure_ascii=False)


class FakeEvent:
    """Minimal Telethon NewMessage stand-in: text, sender, privacy, reply recorder."""

    def __init__(self, text: str, sender_id: int, is_private: bool = True):
        self.raw_text = text
        self.sender_id = sender_id
        self.chat_id = sender_id
        self.is_private = is_private
        self.is_reply = False
        self.message = type("Msg", (), {"id": 1})()
        self.client = None
        self.replies: list[str] = []

    async def reply(self, text, **kwargs):
        self.replies.append(text)


@pytest.fixture
def products_list(products):
    """conftest products reachable as search.PRODUCTS; restored afterwards."""
    saved = (search.PRODUCTS, search._by_id, search._matrix, search._ids)
    search.PRODUCTS = list(products)
    search._by_id = {p.id: p for p in products}
    yield products
    search.PRODUCTS, search._by_id, search._matrix, search._ids = saved


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "LOG_PATH", tmp_path / "log.jsonl")
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(config, "GEMINI_DAILY_LIMIT", 1500)
    return tmp_path


# ---------------------------------------------------------------- config / llm_calls

def test_gemini_daily_limit_is_positive_int():
    assert isinstance(config.GEMINI_DAILY_LIMIT, int) and config.GEMINI_DAILY_LIMIT > 0


def test_llm_calls_derived_from_tools():
    tools = [{"name": "find_products_tool", "input": {"keywords": ["krossovka"]}, "n_results": 0},
             {"name": "semantic_search_tool", "input": {"text": "krossovka"}, "n_results": 2}]
    # 3 model requests (two tool rounds + final answer) + 1 query embedding
    assert bot.llm_calls_for({"tools": tools, "escalated": False, "usd": 0.0}) == 4
    assert bot.llm_calls_for({"tools": [], "escalated": False, "usd": 0.0}) == 1
    assert bot.llm_calls_for({}) == 1


def test_llm_calls_explicit_count_wins():
    tools = [{"name": "find_products_tool", "input": {"size": "42"}, "n_results": 1}]
    assert bot.llm_calls_for({"tools": tools, "llm_calls": 5}) == 5


def test_log_turn_writes_llm_calls(paths):
    rec = bot.log_turn(CUSTOMER, "krossovka 42", [], "Ha, bor: https://t.me/example_shop/1300",
                       False, 1200, 0.0, llm_calls=3)
    assert rec["llm_calls"] == 3
    assert json.loads(config.LOG_PATH.read_text().splitlines()[0])["llm_calls"] == 3


def test_handle_customer_logs_llm_calls(paths, monkeypatch):
    from shop_assistant import agent

    def fake_run_agent(chat_id, text):
        agent.last_run = {"tools": [{"name": "find_products_tool", "input": {"size": "42"},
                                     "n_results": 1}], "escalated": False, "usd": 0.0}
        return "Ha, 42 razmer bor."

    monkeypatch.setattr(agent, "run_agent", fake_run_agent)
    monkeypatch.setattr(agent, "last_run", agent.last_run)
    ev = FakeEvent("krossovka 42 bormi?", CUSTOMER)
    asyncio.run(bot.handle_customer(ev))
    assert ev.replies == ["Ha, 42 razmer bor."]
    rec = json.loads(config.LOG_PATH.read_text().splitlines()[-1])
    assert rec["llm_calls"] == 2


# ---------------------------------------------------------------- compute_stats

def test_counts_today_only_and_midnight_boundary(paths, products_list):
    _write(config.LOG_PATH, [
        _line("2026-09-22T23:59:59", "dvoyka bormi?", escalated=True, llm_calls=4),   # yesterday
        _line("2026-09-23T00:00:00", "krossovka 42", llm_calls=2),
        _line("2026-09-23T12:30:00", "kurtka hali bormi?", escalated=True, llm_calls=3),
        _line("2026-09-23T23:59:59", "yangi nima bor?", llm_calls=1),
        _line("2026-09-24T00:00:00", "narxi 2", escalated=True, llm_calls=2),          # tomorrow
    ])
    s = bot.compute_stats(TODAY)
    assert s["questions_today"] == 3
    assert s["escalations_today"] == 1
    assert s["gemini_today"] == 6
    assert s["gemini_limit"] == 1500
    assert s["indexed_posts"] == 3


def test_lines_without_llm_calls_count_one(paths, products_list):
    _write(config.LOG_PATH, [
        _line("2026-09-23T09:00:00", "krossovka 42"),                 # old format
        _line("2026-09-23T09:05:00", "dvoyka bormi?"),                # old format
        _line("2026-09-23T09:10:00", "kurtka narxi?", llm_calls=4),
    ])
    assert bot.compute_stats(TODAY)["gemini_today"] == 6


def test_blank_lines_skipped(paths, products_list):
    config.LOG_PATH.write_text(_line("2026-09-23T09:00:00", "krossovka 42", llm_calls=2) + "\n\n",
                               encoding="utf-8")
    s = bot.compute_stats(TODAY)
    assert s["questions_today"] == 1 and s["gemini_today"] == 2


def test_missing_log_state_and_ingest(paths, products_list):
    s = bot.compute_stats(TODAY)
    assert s["questions_today"] == 0
    assert s["escalations_today"] == 0
    assert s["gemini_today"] == 0
    assert s["last_index_at"] is None
    assert s["indexed_posts"] == 3


def test_empty_log(paths, products_list):
    config.LOG_PATH.write_text("", encoding="utf-8")
    s = bot.compute_stats(TODAY)
    assert (s["questions_today"], s["escalations_today"], s["gemini_today"]) == (0, 0, 0)


def test_state_last_index_at(paths, products_list):
    config.STATE_PATH.write_text(json.dumps({"last_post_id": 1301,
                                             "last_index_at": "2026-09-23T06:00:00"}),
                                 encoding="utf-8")
    assert bot.compute_stats(TODAY)["last_index_at"] == "2026-09-23T06:00:00"


def test_state_without_last_index_at(paths, products_list):
    config.STATE_PATH.write_text(json.dumps({"last_post_id": 1301}), encoding="utf-8")
    assert bot.compute_stats(TODAY)["last_index_at"] is None


def test_ingest_log_today_lines_added(paths, products_list):
    _write(config.LOG_PATH, [_line("2026-09-23T10:00:00", "krossovka 42", llm_calls=2)])
    _write(paths / bot.INGEST_LOG_NAME, [
        json.dumps({"ts": "2026-09-22T23:00:00", "kind": "extract"}),   # yesterday
        json.dumps({"ts": "2026-09-23T03:00:00", "kind": "extract"}),
        json.dumps({"ts": "2026-09-23T03:00:05", "kind": "extract"}),
        json.dumps({"ts": "2026-09-23T03:01:00", "kind": "embed"}),
    ])
    assert bot.compute_stats(TODAY)["gemini_today"] == 2 + 3


def test_explicit_paths(tmp_path, products_list, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_DAILY_LIMIT", 1500)
    log_p, state_p, ingest_p = tmp_path / "l.jsonl", tmp_path / "s.json", tmp_path / "i.jsonl"
    _write(log_p, [_line("2026-09-23T10:00:00", "kurtka hali bormi?", escalated=True, llm_calls=3)])
    state_p.write_text(json.dumps({"last_post_id": 1301, "last_index_at": "2026-09-22T21:00:00"}),
                       encoding="utf-8")
    _write(ingest_p, [json.dumps({"ts": "2026-09-23T04:00:00", "kind": "embed"})])
    s = bot.compute_stats(TODAY, log_path=log_p, state_path=state_p, ingest_log_path=ingest_p)
    assert s == {"indexed_posts": 3, "last_index_at": "2026-09-22T21:00:00",
                 "questions_today": 1, "escalations_today": 1,
                 "gemini_today": 4, "gemini_limit": 1500}


# ---------------------------------------------------------------- format_stats

def test_format_stats_lines():
    text = bot.format_stats({"indexed_posts": 3, "last_index_at": "2026-09-23T06:00:00",
                             "questions_today": 12, "escalations_today": 2,
                             "gemini_today": 37, "gemini_limit": 1500})
    lines = text.split("\n")
    assert "Indexed posts: 3" in lines
    assert "Last index: 2026-09-23T06:00:00" in lines
    assert "Questions today: 12" in lines
    assert "Escalations today: 2" in lines
    assert "Gemini: 37 / 1500 today" in lines


def test_format_stats_never_indexed():
    text = bot.format_stats({"indexed_posts": 0, "last_index_at": None, "questions_today": 0,
                             "escalations_today": 0, "gemini_today": 0, "gemini_limit": 250})
    assert "Last index: -" in text.split("\n")
    assert "Gemini: 0 / 250 today" in text


# ---------------------------------------------------------------- handlers + gating

def test_owner_stats_replies_with_todays_counts(paths, products_list):
    today = date.today().isoformat()
    _write(config.LOG_PATH, [_line(f"{today}T10:00:00", "krossovka 42", llm_calls=2),
                             _line(f"{today}T11:00:00", "kurtka hali bormi?", escalated=True)])
    ev = FakeEvent("/stats", OWNER)
    asyncio.run(bot.route(ev, OWNER))
    assert len(ev.replies) == 1
    assert "Gemini: 3 / 1500 today" in ev.replies[0]
    assert "Questions today: 2" in ev.replies[0]
    assert "Escalations today: 1" in ev.replies[0]


def _recorders(monkeypatch) -> list[str]:
    calls: list[str] = []
    for name in ("handle_customer", "handle_stats", "handle_reindex", "handle_owner_reply"):
        async def rec(event, _n=name):
            calls.append(_n)
        monkeypatch.setattr(bot, name, rec)
    return calls


@pytest.mark.parametrize("text", ["/stats", "/reindex"])
def test_customer_command_goes_to_agent(monkeypatch, text):
    calls = _recorders(monkeypatch)
    asyncio.run(bot.route(FakeEvent(text, CUSTOMER), OWNER))
    assert calls == ["handle_customer"]


@pytest.mark.parametrize("text, handler", [("/stats", "handle_stats"),
                                           ("/reindex", "handle_reindex"),
                                           ("salom", "handle_owner_reply")])
def test_owner_routing(monkeypatch, text, handler):
    calls = _recorders(monkeypatch)
    asyncio.run(bot.route(FakeEvent(text, OWNER), OWNER))
    assert calls == [handler]


def test_group_messages_ignored(monkeypatch):
    calls = _recorders(monkeypatch)
    asyncio.run(bot.route(FakeEvent("/stats", CUSTOMER, is_private=False), OWNER))
    asyncio.run(bot.route(FakeEvent("/stats", OWNER, is_private=False), OWNER))
    assert calls == []


def test_customer_stats_gets_normal_reply(paths, products_list, monkeypatch):
    from shop_assistant import agent
    seen: list[str] = []

    def fake_run_agent(chat_id, text):
        seen.append(text)
        return "Qaysi mahsulot kerak? Nomini yozing."

    monkeypatch.setattr(agent, "run_agent", fake_run_agent)
    monkeypatch.setattr(agent, "last_run", {"tools": [], "escalated": False, "usd": 0.0})
    ev = FakeEvent("/stats", CUSTOMER)
    asyncio.run(bot.route(ev, OWNER))
    assert seen == ["/stats"]
    assert ev.replies == ["Qaysi mahsulot kerak? Nomini yozing."]
    assert "Gemini:" not in ev.replies[0]


# ---------------------------------------------------------------- /reindex

def test_reindex_calls_search_reload(monkeypatch, products_list):
    called: list[bool] = []
    monkeypatch.setattr(search, "reload", lambda: called.append(True))
    ev = FakeEvent("/reindex", OWNER)
    asyncio.run(bot.handle_reindex(ev))
    assert called == [True]
    assert len(ev.replies) == 1


def test_reindex_picks_up_new_product(tmp_path, monkeypatch, products_list):
    dvoyka, krossovka, kurtka = products_list
    prod_path = tmp_path / "products.jsonl"
    monkeypatch.setattr(config, "PRODUCTS_PATH", prod_path)
    monkeypatch.setattr(config, "EMBEDDINGS_PATH", tmp_path / "embeddings.npy")
    monkeypatch.setattr(config, "EMBEDDINGS_IDS_PATH", tmp_path / "embeddings_ids.json")
    _write(prod_path, [_product_line(dvoyka), _product_line(krossovka)])
    search.reload()
    assert search.find_products(keywords=["kurtka"]) == []

    # out-of-band ingest appends a product; the running bot must see it after /reindex
    with open(prod_path, "a", encoding="utf-8") as f:
        f.write(_product_line(kurtka) + "\n")
    ev = FakeEvent("/reindex", OWNER)
    asyncio.run(bot.route(ev, OWNER))

    assert [p.id for p in search.find_products(keywords=["kurtka"])] == [900]
    assert len(search.PRODUCTS) == 3
    assert len(ev.replies) == 1 and "3" in ev.replies[0]
