"""Ticket #26 — agent.py: shorter Gemini timeout + one retry on 5xx / timeout (fake client, no network).

Contract (decision by Sanjarbek: retry once on the same model, no backup model):
* `llm.AGENT_TIMEOUT_S` <= 8 s, still sent in every agent request as `http_options.timeout` (ms).
* Each agent `generate_content` request that fails with a server error (500 / 502 / 503 / 504) or a
  client timeout (httpx timeout, e.g. `httpx.ReadTimeout` / `httpx.ConnectTimeout`) is retried ONCE,
  with the same model and the same contents, after a pause of 0.5–1 s taken with `time.sleep`
  (agent does `import time` ... `time.sleep(...)`, so tests patch `time.sleep`; no real waiting).
* 429 (quota) and 400 (invalid key / bad request) are never retried: one request, no sleep, the
  polite reply in the chat's language, exactly as before.
* Logging (logger `shop_assistant.agent`): the first failure -> exactly one WARNING; if the retry also
  fails -> the existing single ERROR line and the polite reply. run_agent never raises.
* `agent.last_run["llm_calls"]` counts every request sent, retries included.
* Extraction / embedding retry policies are unchanged (their own tests cover them).
"""
import logging
import time

import httpx
import pytest
from google.genai import errors

from shop_assistant import agent, llm, search
from shop_assistant.agent import History
from shop_assistant.lang import TEXTS
from tests.fake_gemini import (FakeClient, call_response, dump, invalid_key_error, plain, quota_error,
                               server_error, text_response, timeout_error)

QUESTION = "menga 42 razmerli poyabzal kerak"
ANSWER = ("1. Krossovka Nike Air · 350000 · 40,41,42,43 · 2026-09-12 · https://t.me/example_shop/1300\n"
          "Yana ko'rsataymi?")


def _server(code: int) -> errors.ServerError:
    status = {500: "INTERNAL", 502: "BAD_GATEWAY", 503: "UNAVAILABLE", 504: "DEADLINE_EXCEEDED"}[code]
    return errors.ServerError(code, {"error": {"code": code, "status": status,
                                               "message": "The service is currently unavailable."}})


def _gateway_timeout() -> errors.ServerError:
    return _server(504)


def _connect_timeout():
    return httpx.ConnectTimeout("timed out while connecting")


RETRYABLE = {
    "500": lambda: _server(500),
    "502": lambda: _server(502),
    "503": server_error,
    "504": _gateway_timeout,
    "read-timeout": timeout_error,
    "connect-timeout": _connect_timeout,
}


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Every time.sleep call is recorded instead of waiting."""
    got: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: got.append(s))
    return got


@pytest.fixture
def gemini(monkeypatch):
    """Install a FakeClient with the given script behind llm.client(); returns it.
    The script's last entry repeats (see tests/fake_gemini.py)."""
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
        return products[1:2]
    monkeypatch.setattr(search, "find_products", find_products)
    return calls


def _agent_records(caplog, level):
    return [r for r in caplog.records if r.name == agent.__name__ and r.levelno == level]


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


# --- timeout -----------------------------------------------------------------

def test_agent_timeout_is_at_most_8_seconds():
    assert 0 < llm.AGENT_TIMEOUT_S <= 8
    assert llm.EXTRACT_TIMEOUT_S > llm.AGENT_TIMEOUT_S        # extraction timeout unchanged


def test_shorter_timeout_is_still_sent_in_every_request_including_the_retry(gemini, catalog):
    fake = gemini(_gateway_timeout(), text_response(ANSWER))
    agent.run_agent(2601, QUESTION, History())
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert plain(call["config"])["http_options"]["timeout"] == llm.AGENT_TIMEOUT_S * 1000


# --- one retry on 5xx / timeout ----------------------------------------------

@pytest.mark.parametrize("error", list(RETRYABLE.values()), ids=list(RETRYABLE))
def test_one_transient_failure_is_retried_and_the_customer_gets_the_answer(gemini, catalog, caplog,
                                                                          sleeps, error):
    caplog.set_level(logging.WARNING)
    h = History()
    fake = gemini(error(), text_response(ANSWER))
    reply = agent.run_agent(2602, QUESTION, h)
    assert reply == ANSWER
    assert len(fake.calls) == 2
    assert fake.calls[1]["model"] == fake.calls[0]["model"]             # same model, no backup model
    assert dump(fake.calls[1]["contents"]) == dump(fake.calls[0]["contents"])   # same request again
    assert len(sleeps) == 1 and 0.5 <= sleeps[0] <= 1.0
    assert len(_agent_records(caplog, logging.WARNING)) == 1
    assert _errors(caplog) == []
    assert agent.last_run["llm_calls"] == 2
    assert h.get(2602)[-2:] == [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": ANSWER}]


@pytest.mark.parametrize("lang", ["uz_latn", "uz_cyrl", "ru"])
@pytest.mark.parametrize("error", [_gateway_timeout, server_error, timeout_error], ids=["504", "503", "timeout"])
def test_failing_twice_gives_the_polite_reply_in_the_chat_language(gemini, catalog, caplog, sleeps,
                                                                   error, lang):
    caplog.set_level(logging.WARNING)
    h = History()
    fake = gemini(error())                                   # repeats: the retry fails too
    reply = agent.run_agent(2603, QUESTION, h, lang=lang)
    expected = agent.APOLOGY if lang == "uz_latn" else TEXTS[lang]["error_reply"]
    assert reply == expected
    for raw in ("UNAVAILABLE", "DEADLINE_EXCEEDED", "timed out", "503", "504"):
        assert raw not in reply
    assert len(fake.calls) == 2                              # one retry, never more
    assert len(sleeps) == 1 and 0.5 <= sleeps[0] <= 1.0
    assert len(_agent_records(caplog, logging.WARNING)) == 1
    assert len(_errors(caplog)) == 1
    assert agent.last_run["llm_calls"] == 2
    assert h.get(2603) == []                                 # failed turn is not stored


# --- no retry on 429 / 400 ---------------------------------------------------

def test_quota_error_is_not_retried(gemini, catalog, caplog, sleeps):
    caplog.set_level(logging.WARNING)
    fake = gemini(quota_error(), text_response(ANSWER))
    reply = agent.run_agent(2604, QUESTION, History())
    assert reply == agent.APOLOGY
    assert len(fake.calls) == 1
    assert sleeps == []
    assert _agent_records(caplog, logging.WARNING) == []
    assert len(_errors(caplog)) == 1
    assert agent.last_run["llm_calls"] == 1


def test_invalid_key_is_not_retried(gemini, catalog, caplog, sleeps):
    caplog.set_level(logging.WARNING)
    fake = gemini(invalid_key_error(), text_response(ANSWER))
    reply = agent.run_agent(2605, "Qishki kurtka bormi? L razmer", History(), lang="ru")
    assert reply == TEXTS["ru"]["error_reply"]
    assert len(fake.calls) == 1
    assert sleeps == []
    assert _agent_records(caplog, logging.WARNING) == []
    assert len(_errors(caplog)) == 1


# --- retry inside the tool loop ----------------------------------------------

def test_retry_in_the_middle_of_a_tool_loop(gemini, catalog, caplog, sleeps):
    caplog.set_level(logging.WARNING)
    fake = gemini(call_response("find_products_tool", {"keywords": ["poyabzal"], "size": "42"}),
                  _gateway_timeout(),
                  text_response(ANSWER))
    reply = agent.run_agent(2606, QUESTION, History())
    assert reply == ANSWER
    assert len(fake.calls) == 3
    assert len(catalog) == 1                                 # the tool ran once, not again on the retry
    second, third = dump(fake.calls[1]["contents"]), dump(fake.calls[2]["contents"])
    assert second == third                                   # the retry resends the tool result
    assert "function_response" in third and "Krossovka Nike Air" in third
    assert len(sleeps) == 1 and 0.5 <= sleeps[0] <= 1.0
    assert agent.last_run["llm_calls"] == 3                  # every attempt counts
    assert [t["name"] for t in agent.last_run["tools"]] == ["find_products_tool"]
    assert len(_agent_records(caplog, logging.WARNING)) == 1
    assert _errors(caplog) == []


def test_each_model_step_gets_its_own_single_retry(gemini, catalog, caplog, sleeps):
    """A 503 on the first step and a 504 on the second step are each retried once."""
    caplog.set_level(logging.WARNING)
    fake = gemini(server_error(),
                  call_response("find_products_tool", {"keywords": ["poyabzal"], "size": "42"}),
                  _gateway_timeout(),
                  text_response(ANSWER))
    reply = agent.run_agent(2607, QUESTION, History())
    assert reply == ANSWER
    assert len(fake.calls) == 4
    assert len(sleeps) == 2
    assert agent.last_run["llm_calls"] == 4
    assert len(_agent_records(caplog, logging.WARNING)) == 2
    assert _errors(caplog) == []
