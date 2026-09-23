"""Ticket #10 — FR-17 history (pure part); ticket #23 — run_agent on Gemini through the llm seam
(fake client, no network; the seam is documented at the top of tests/test_llm.py)."""
import logging
import time

import pytest

from shop_assistant import agent, config, llm, search
from shop_assistant.agent import History
from shop_assistant.tools import TOOLS
from tests.fake_gemini import (FakeClient, call_response, dump, empty_response, invalid_key_error, plain,
                               quota_error, server_error, text_response, timeout_error)

QUESTION = "Krossovka bormi? 42 razmer"
ANSWER = ("1. Dvoyka · 980000 · M,L,XL,2XL,3XL · 2026-09-10 · https://t.me/example_shop/1234\n"
          "2. Krossovka Nike Air · 350000 · 40,41,42,43 · 2026-09-12 · https://t.me/example_shop/1300")


def test_history_keeps_last_10_turns():
    h = History(max_turns=10)
    for i in range(12):
        h.append(7, "user", f"q{i}")
        h.append(7, "assistant", f"a{i}")
    msgs = h.get(7)
    assert len(msgs) == 20
    assert msgs[0] == {"role": "user", "content": "q2"}
    assert msgs[-1] == {"role": "assistant", "content": "a11"}


def test_history_is_per_customer():
    h = History()
    h.append(1, "user", "krossovka bormi")
    assert h.get(2) == []


def test_history_clear():
    h = History()
    h.append(1, "user", "x")
    h.clear(1)
    assert h.get(1) == []


# --- ticket #23: run_agent on Gemini ------------------------------------------

@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Ticket #26: agent retries 5xx / timeouts once after time.sleep; never wait for real in tests."""
    monkeypatch.setattr(time, "sleep", lambda s: None)


@pytest.fixture
def gemini(monkeypatch):
    """Install a FakeClient with the given script behind llm.client(); returns it."""
    def install(*script):
        fake = FakeClient(list(script))
        monkeypatch.setattr(llm, "client", lambda: fake)
        return fake
    return install


@pytest.fixture
def catalog(monkeypatch, products):
    calls = []

    def find_products(**kw):
        calls.append(kw)
        return products[:2]
    monkeypatch.setattr(search, "find_products", find_products)
    monkeypatch.setattr(search, "latest_posts", lambda n: products[:1])
    return calls


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_function_call_loop_runs_the_tool_and_returns_the_model_text(gemini, catalog):
    fake = gemini(call_response("find_products_tool", {"keywords": ["krossovka"], "size": "42"}),
                  text_response(ANSWER))
    reply = agent.run_agent(501, QUESTION, History())
    assert reply == ANSWER
    assert catalog and catalog[0]["keywords"] == ["krossovka"] and catalog[0]["size"] == "42"
    assert len(fake.calls) == 2
    second = dump(fake.calls[1]["contents"])
    assert "function_call" in second and "function_response" in second      # model turn + tool result sent back
    assert "find_products_tool" in second and "Krossovka Nike Air" in second


def test_request_config_disables_automatic_calling_and_declares_all_tools(gemini, catalog):
    fake = gemini(text_response("Assalomu alaykum! Qanday mahsulot qidiryapsiz?"))
    agent.run_agent(502, "Salom", History())
    call = fake.calls[0]
    assert call["model"] == getattr(config, "GEMINI_MODEL", None)
    cfg = plain(call["config"])
    assert cfg["automatic_function_calling"]["disable"] is True
    assert cfg["http_options"]["timeout"] == llm.AGENT_TIMEOUT_S * 1000
    assert agent.SYSTEM_PROMPT[:60] in dump(cfg["system_instruction"])
    declared = [f["name"] for tool in cfg["tools"] for f in tool["function_declarations"]]
    assert declared == [t.name for t in TOOLS]


def test_last_run_records_tools_and_results(gemini, catalog):
    gemini(call_response("find_products_tool", {"keywords": ["krossovka"]}), text_response(ANSWER))
    agent.run_agent(503, QUESTION, History())
    assert agent.last_run["tools"] == [{"name": "find_products_tool", "input": {"keywords": ["krossovka"]},
                                        "n_results": 2}]
    assert agent.last_run["escalated"] is False


def test_ask_owner_marks_the_turn_escalated(gemini, monkeypatch):
    from shop_assistant import bot
    forwarded = []
    monkeypatch.setattr(bot, "escalate_sync", lambda q, ids: forwarded.append((q, ids)))
    gemini(call_response("ask_owner", {"question": "Qishki kurtka hali bormi?", "post_ids": [900]}),
           text_response("Savolingizni do'kon egasiga yubordim, tez orada javob beradi."))
    reply = agent.run_agent(504, "Qishki kurtka hali bormi?", History())
    assert forwarded == [("Qishki kurtka hali bormi?", [900])]
    assert agent.last_run["escalated"] is True
    assert "egasi" in reply


def test_loop_stops_after_max_iterations(gemini, catalog):
    fake = gemini(call_response("latest_posts_tool", {"n": 1}))      # the model never stops calling tools
    reply = agent.run_agent(505, "Yangi tovarlar bormi?", History())
    assert config.MAX_ITERATIONS == 8
    assert len(fake.calls) == config.MAX_ITERATIONS
    assert reply == agent.APOLOGY
    assert 7 <= len(agent.last_run["tools"]) <= 8


def test_history_format_is_kept_and_sent_to_the_model(gemini, catalog):
    h = History()
    h.append(506, "user", "Dvoyka bormi?")
    h.append(506, "assistant", "1. Dvoyka · 980000 · M,L,XL · https://t.me/example_shop/1234")
    fake = gemini(text_response("Ha, 42 razmer bor."))
    reply = agent.run_agent(506, QUESTION, h)
    sent = dump(fake.calls[0]["contents"])
    assert sent.index("Dvoyka bormi?") < sent.index("t.me/example_shop/1234") < sent.index(QUESTION)
    assert '"model"' in sent                                  # assistant turns go to Gemini as role "model"
    assert h.get(506)[-2:] == [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": reply}]


@pytest.mark.parametrize("error", [quota_error, server_error, timeout_error, invalid_key_error],
                         ids=["429", "503", "timeout", "invalid-key"])
def test_api_errors_give_polite_reply_and_one_error_log(gemini, catalog, caplog, error):
    """503 / timeout fail twice here (the fake repeats its last entry): since #26 that is one retry,
    one WARNING, then still exactly one ERROR (tests/test_agent_retry.py covers the retry itself)."""
    h = History()
    gemini(error())
    reply = agent.run_agent(507, QUESTION, h)
    assert reply == agent.APOLOGY
    for raw in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "timed out", "API key", "429", "503"):
        assert raw not in reply
    assert len(_errors(caplog)) == 1
    assert h.get(507) == []                                   # failed turn is not stored
    assert agent.last_run["tools"] == []


def test_error_after_a_tool_call_still_gives_polite_reply(gemini, catalog, caplog):
    gemini(call_response("find_products_tool", {"keywords": ["krossovka"]}), quota_error())
    reply = agent.run_agent(508, QUESTION, History())
    assert reply == agent.APOLOGY
    assert len(_errors(caplog)) == 1


def test_missing_api_key_gives_polite_reply(monkeypatch, caplog):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_client", None)
    reply = agent.run_agent(509, QUESTION, History())
    assert reply == agent.APOLOGY
    assert len(_errors(caplog)) == 1


def test_empty_model_response_gives_polite_reply(gemini, catalog):
    gemini(empty_response())
    assert agent.run_agent(510, QUESTION, History()) == agent.APOLOGY


def test_every_turn_logs_its_own_error(gemini, catalog, caplog):
    h = History()
    gemini(quota_error())
    agent.run_agent(511, QUESTION, h)
    agent.run_agent(511, "Kurtka bormi?", h)
    assert len(_errors(caplog)) == 2
