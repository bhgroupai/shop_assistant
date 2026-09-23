"""Ticket #23 — extraction on Gemini through the llm seam (fake client, no network).
The seam is documented at the top of tests/test_llm.py."""
import json
import logging
import re
import time

import anthropic
import pytest

from shop_assistant import config, extract, llm
from shop_assistant.models import Post
from tests.conftest import CAPTION_DVOYKA, CAPTION_KROSSOVKA, CAPTION_KURTKA
from tests.fake_gemini import FakeClient, call_response, dump, plain, quota_error, server_error, timeout_error

# What a good model returns for each real caption (conftest); the guards must still fix the bad fields.
ITEMS = {
    1234: {"id": 1234, "name": "Yengi kolleksiya", "category": "kiyim", "price": "980.000ming",
           "subscriber_price": None, "sizes": ["M", "L", "XL", "2XL", "3XL"], "colors": [],
           "keywords": ["dvoyka", "двойка", "костюм двойка", "two-piece set"], "season": "kuz"},
    1300: {"id": 1300, "name": "Krossovka Nike Air", "category": "poyabzal", "price": 350000,
           "subscriber_price": 320000, "sizes": ["40", "41", "42", "43"], "colors": [],
           "keywords": ["krossovka", "кроссовки", "sneakers", "nike"], "season": None},
    1400: {"id": 1400, "name": "Qishki kurtka", "category": "kiyim", "price": 1,
           "subscriber_price": None, "sizes": ["L", "XL"], "colors": ["qora", "kok"],
           "keywords": ["kurtka", "куртка", "jacket", "qishki kurtka"], "season": "qish"},
    1500: {"id": 1500, "name": "Kutib qoling", "category": "kiyim", "price": None,
           "subscriber_price": None, "sizes": [], "colors": [], "keywords": ["kolleksiya"], "season": None},
}


def _post(pid: int, caption: str, has_media: bool = True) -> Post:
    return Post(id=pid, date="2026-09-20T10:00:00", link=f"https://t.me/example_shop/{pid}",
                caption=caption, has_media=has_media)


POSTS = [_post(1234, CAPTION_DVOYKA), _post(1300, CAPTION_KROSSOVKA), _post(1400, CAPTION_KURTKA),
         _post(1500, "Yengi kolleksiya\nKuz mavsumi uchun\nKutib qoling", has_media=False)]


def _record_products(contents):
    """Answer like the model: one record per '### id=N' header in the prompt."""
    ids = [int(x) for x in re.findall(r"### id=(\d+)", dump(contents))]
    return call_response("record_products", {"products": [ITEMS[i] for i in ids]})


class _NoAnthropic:
    def __init__(self, *a, **kw):
        raise AssertionError("extract still builds an Anthropic client — ticket #23 moves it to llm.client()")


@pytest.fixture(autouse=True)
def _no_anthropic_no_sleep(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _NoAnthropic)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    return sleeps


@pytest.fixture
def sleeps(_no_anthropic_no_sleep):
    return _no_anthropic_no_sleep


@pytest.fixture
def gemini(monkeypatch):
    def install(*script):
        fake = FakeClient(list(script))
        monkeypatch.setattr(llm, "client", lambda: fake)
        return fake
    return install


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_extraction_forces_the_record_products_function(gemini):
    fake = gemini(_record_products)
    extract.extract_batch(POSTS[:2])
    call = fake.calls[0]
    assert call["model"] == getattr(config, "GEMINI_MODEL", None)
    cfg = plain(call["config"])
    fcc = cfg["tool_config"]["function_calling_config"]
    assert str(fcc["mode"]).upper() == "ANY"
    assert fcc["allowed_function_names"] == ["record_products"]
    assert [f["name"] for tool in cfg["tools"] for f in tool["function_declarations"]] == ["record_products"]
    assert cfg["http_options"]["timeout"] == llm.EXTRACT_TIMEOUT_S * 1000
    assert "record_products" in dump(cfg["system_instruction"])
    prompt = dump(call["contents"])
    assert "### id=1234" in prompt and "### id=1300" in prompt and "+998" not in prompt   # footer stripped


def test_extraction_returns_one_product_per_post_in_order(gemini):
    gemini(_record_products)
    out = extract.extract_batch(POSTS)
    assert [p.id for p in out] == [1234, 1300, 1400, 1500]
    kross = out[1]
    assert (kross.name, kross.category, kross.price, kross.subscriber_price) == \
        ("Krossovka Nike Air", "poyabzal", 350000, 320000)
    assert kross.sizes == ("40", "41", "42", "43")


def test_deterministic_guards_still_fix_model_output(gemini):
    gemini(_record_products)
    dvoyka, _, kurtka, teaser = extract.extract_batch(POSTS)
    assert dvoyka.price == 980000                      # price notation "980.000ming"
    assert dvoyka.name == "Dvoyka"                     # slogan "Yengi kolleksiya" replaced (product_name)
    assert kurtka.price == 1200000                     # junk price 1 -> price from the body
    assert teaser.category == "boshqa" and teaser.price is None    # is_announcement


def test_post_missing_from_the_answer_gets_the_fallback_record(gemini):
    gemini(call_response("record_products", {"products": [ITEMS[1234]]}))
    dvoyka, kross = extract.extract_batch(POSTS[:2])
    assert dvoyka.price == 980000
    assert kross.id == 1300 and kross.category == "boshqa" and kross.price is None


@pytest.mark.parametrize("error", [quota_error, server_error, timeout_error], ids=["429", "503", "timeout"])
def test_transient_errors_are_retried_with_backoff(gemini, sleeps, error):
    fake = gemini(error(), error(), _record_products)
    out = extract.extract_batch(POSTS[:2])
    assert [p.id for p in out] == [1234, 1300] and out[0].price == 980000
    assert len(fake.calls) == 3
    assert len(sleeps) == 2 and 0 < sleeps[0] < sleeps[1]


def test_final_failure_raises_and_returns_no_fallback_records(gemini, sleeps):
    fake = gemini(server_error())
    with pytest.raises(Exception):
        extract.extract_batch(POSTS[:2])
    assert 3 <= len(fake.calls) <= 10
    assert sleeps == sorted(sleeps) and sleeps[-1] > sleeps[0]


def test_no_posts_makes_no_request(monkeypatch):
    def boom():
        raise AssertionError("no request expected for an empty batch")
    monkeypatch.setattr(llm, "client", boom)
    assert extract.extract_batch([]) == []


# --- main(): never write half a batch; failed posts stay un-extracted -----------

@pytest.fixture
def data_files(tmp_path, monkeypatch):
    posts_path, products_path = tmp_path / "posts.jsonl", tmp_path / "products.jsonl"
    with posts_path.open("w", encoding="utf-8") as f:
        for p in POSTS[:3]:
            f.write(json.dumps({"id": p.id, "date": p.date, "link": p.link, "caption": p.caption,
                                "has_media": p.has_media}, ensure_ascii=False) + "\n")
    monkeypatch.setattr(config, "POSTS_PATH", posts_path)
    monkeypatch.setattr(config, "PRODUCTS_PATH", products_path)
    monkeypatch.setattr(config, "EXTRACT_BATCH", 2)
    return products_path


def _ids(path) -> list[int]:
    if not path.exists():
        return []
    return [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_main_keeps_failed_batch_unextracted_and_next_run_picks_it_up(gemini, data_files, caplog):
    gemini(_record_products, server_error())          # batch 1 ok, batch 2 fails on every retry
    extract.main()                                     # offline job: logs, does not crash
    assert _ids(data_files) == [1234, 1300]            # whole first batch, nothing of the second
    assert _errors(caplog)
    gemini(_record_products)
    extract.main()
    assert _ids(data_files) == [1234, 1300, 1400]      # the failed post is picked up, nothing duplicated


def test_main_writes_nothing_when_the_first_batch_fails(gemini, data_files):
    gemini(quota_error())
    extract.main()
    assert _ids(data_files) == []
