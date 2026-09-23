"""Ticket #18 — R2: nightly ingestion (systemd timer) + the bot picks up the new index without a restart.
Fake Gemini client (tests/fake_gemini.py) and a fake Telegram fetch; no network, no Telegram session.

CONTRACT (read before implementing):

* Entry point `python -m shop_assistant.ingest` → `ingest.main() -> int` (0 = the night succeeded, 1 = it failed,
  so systemd marks the unit failed). `ingest.run() -> bool` does, in order, stopping at the first failing step:
    1. fetch   — exactly what `fetch.main()` does: min_id = state.json "last_post_id", new posts appended to
                 posts.jsonl, last_post_id updated. Telegram is reached only through `fetch.fetch(channel, min_id,
                 limit)` looked up at call time (the tests replace it).
    2. extract — posts in posts.jsonl whose id is not in products.jsonl ("pending"), in batches of
                 config.EXTRACT_BATCH (the `extract.main()` behaviour: whole batches only; on a final failure the
                 batch is not written and its posts stay pending). run() must learn that extraction failed.
    3. index   — `index.update_index()` (below).
    4. state.json "last_index_at" = now as ISO datetime (`datetime.now().isoformat(timespec="seconds")`), keeping
       every other key. Written ONLY when all steps succeeded (also on a night with nothing new).
  A failing step (Telegram error, Gemini 429/5xx after the retries, model mismatch, …) logs an ERROR and makes
  run() return False; it never raises. Files the failed step would have written stay byte-identical; posts that
  were not extracted / not embedded stay pending and the next night picks them up with no duplicates.
  Fetched posts that were appended before a later step failed are kept (they are pending, not half-written).

* `index.update_index() -> int` — the incremental index. Embeds `normalise(product_text(p))` (D-3) with
  kind="document", retry=True ONLY for products of products.jsonl whose ids are not yet in embeddings_ids.json,
  appends their rows after the existing ones (ids in products.jsonl order), then rewrites embeddings.npy +
  embeddings_ids.json + embeddings_meta.json atomically (temp file + os.replace, meta last) — the existing rows
  are copied unchanged, never re-embedded. Returns the number of rows added.
    - nothing new → returns 0, sends no request and touches no file (so the bot does not reload for nothing);
    - no matrix yet (first run) → embeds all products (same result as `reindex()`);
    - the matrix on disk has another model or dimension in embeddings_meta.json, or no meta file at all →
      raises `index.ModelMismatch` (message names the stored model / dimension) with NO request and no file
      written: a full re-embed is `python -m shop_assistant.index`, run by hand, never by the timer
      (the free quota is shared with customers). run() turns it into an ERROR and False.
    - embedding fails after the retries → the error propagates, no file written.

* Gemini request log (feeds /stats, ticket #17): every Gemini HTTP request made during run() — extraction
  `generate_content` and embedding `embed_content`, INCLUDING retried attempts and failed ones — appends one JSON
  line to `config.DATA_DIR / ingest.INGEST_LOG_NAME` (== bot.INGEST_LOG_NAME == "gemini_ingest.jsonl"):
  {"ts": ISO datetime (local, starts with the date), "kind": "extract" | "embed", "model": str, "n": int}
  where n = posts / texts in that request. What we count is HTTP requests: one embed request of 5 texts is ONE
  line (it still costs 5 of the 100 texts/min embed quota — `n` records that). A customer's query embedding
  (search.semantic_search) is NOT an ingestion request and never writes this file (log.jsonl counts it).

* Budget (acceptance 3): 5 new posts → 1 extraction request (5 ≤ EXTRACT_BATCH = 20) + 1 embedding request
  (5 ≤ EMBED_BATCH = 100) = 2 HTTP requests (≤ 3). The embedding request carries only the 5 new texts.

* Reload without restart (D-7 kept: ingestion runs in its own oneshot unit, the bot is never restarted by it):
  `search.reload_if_changed() -> bool` compares an os.stat signature ((mtime_ns, size) or None when missing) of
  products.jsonl and the embeddings files (at least embeddings_meta.json, which update_index writes last)
  with `search._loaded_sig`, recorded by every `search.reload()` (and at import). Different → reload(), True;
  same → False and no file is read. `find_products`, `latest_posts` and `semantic_search` call it first, so the
  first customer search after the night run sees the new posts.

* Units (deploy/, installed by deploy.sh next to shop-assistant.service):
  deploy/shop-assistant-ingest.service — Type=oneshot, WorkingDirectory=%h/shop_assistant,
    ExecStart=%h/shop_assistant/.venv/bin/python -m shop_assistant.ingest (no systemctl / bot restart);
  deploy/shop-assistant-ingest.timer — OnCalendar at night (hour 0–6), Persistent=true (a night missed while the
    server was off runs at boot), WantedBy=timers.target. deploy.sh copies both and enables the timer.
"""
import configparser
import dataclasses
import datetime
import json
import logging
import os
import re
import time

import numpy as np
import pytest
from google.genai import errors

from shop_assistant import bot, config, fetch, index, ingest, llm, search
from shop_assistant.index import product_text
from shop_assistant.models import Post, Product
from shop_assistant.textnorm import normalise
from tests.conftest import CAPTION_DVOYKA, CAPTION_KROSSOVKA, CAPTION_KURTKA, FOOTER
from tests.fake_gemini import (FakeClient, call_response, dump, embed_response, fake_vector, quota_error,
                               server_error)

DIM = 8
OLD_MODEL = "text-embedding-004"

# Tonight's 5 new posts, in the real channel format (SDD §2.1).
NEW_CAPTIONS = {
    1310: "🧥Ayollar palto🧥\nRang: bej, qora\nRazmer: S M L\nNarx: 750.000" + FOOTER,
    1311: "🧶Qishki sviter\nRazmer: M L XL\nNarx:180.000ming" + FOOTER,
    1312: "🥾Charm botinka\nRazmer: 39 40 41 42\nNarx: 420.000" + FOOTER,
    1313: "👜Ayollar sumkasi\nRang: qora, jigarrang\nNarx: 260.000" + FOOTER,
    1314: "👖Klassik shim\nRazmer: 30 32 34\nNarx: 210.000" + FOOTER,
}
NEW_IDS = sorted(NEW_CAPTIONS)
ITEMS = {
    1310: {"name": "Ayollar palto", "category": "kiyim", "price": 750000, "sizes": ["S", "M", "L"],
           "colors": ["bej", "qora"], "keywords": ["palto", "пальто", "coat", "ayollar palto"], "season": "kuz"},
    1311: {"name": "Qishki sviter", "category": "kiyim", "price": 180000, "sizes": ["M", "L", "XL"],
           "colors": [], "keywords": ["sviter", "свитер", "sweater"], "season": "qish"},
    1312: {"name": "Charm botinka", "category": "poyabzal", "price": 420000, "sizes": ["39", "40", "41", "42"],
           "colors": [], "keywords": ["botinka", "ботинки", "boots"], "season": "kuz"},
    1313: {"name": "Ayollar sumkasi", "category": "aksessuar", "price": 260000, "sizes": [],
           "colors": ["qora", "jigarrang"], "keywords": ["sumka", "сумка", "bag"], "season": None},
    1314: {"name": "Klassik shim", "category": "kiyim", "price": 210000, "sizes": ["30", "32", "34"],
           "colors": [], "keywords": ["shim", "брюки", "trousers"], "season": None},
}
OLD_POSTS = [
    Post(id=900, date="2026-06-01T10:00:00", link="https://t.me/example_shop/900", caption=CAPTION_KURTKA),
    Post(id=1234, date="2026-09-10T14:02:00", link="https://t.me/example_shop/1234", caption=CAPTION_DVOYKA),
    Post(id=1300, date="2026-09-12T10:00:00", link="https://t.me/example_shop/1300", caption=CAPTION_KROSSOVKA),
]
NEW_POSTS = [Post(id=i, date="2026-09-22T12:00:00", link=f"https://t.me/example_shop/{i}", caption=c)
             for i, c in NEW_CAPTIONS.items()]


def _record_products(contents):
    """Answer like the model: one record per '### id=N' header in the prompt."""
    ids = [int(x) for x in re.findall(r"### id=(\d+)", dump(contents))]
    return call_response("record_products", {"products": [
        {"id": i, "subscriber_price": None, **ITEMS[i]} for i in ids]})


def _unit(text: str) -> np.ndarray:
    v = np.asarray(fake_vector(text, DIM), dtype=np.float32)
    return v / np.linalg.norm(v)


def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids(path) -> list[int]:
    return [d["id"] for d in _jsonl(path)]


def _state(shop) -> dict:
    return json.loads(shop.state.read_text(encoding="utf-8")) if shop.state.exists() else {}


def _snapshot(*paths) -> dict:
    return {p.name: (p.read_bytes() if p.exists() else None) for p in paths}


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def _later(*paths, seconds: int = 5):
    """Push mtimes forward so a stat-based check cannot miss a rewrite made within the same clock tick."""
    for p in paths:
        if p.exists():
            st = p.stat()
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


@dataclasses.dataclass
class Shop:
    dir: object
    posts: object
    products: object
    npy: object
    ids: object
    meta: object
    state: object
    ingest_log: object
    fetch_calls: list
    channel: list

    @property
    def index_files(self):
        return (self.npy, self.ids, self.meta)

    @property
    def all_files(self):
        return (self.posts, self.products, self.npy, self.ids, self.meta, self.state)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    return sleeps


@pytest.fixture
def shop(tmp_path, monkeypatch, products):
    """Yesterday's shop in tmp_path: 3 posts, their 3 products (conftest), a 3-row matrix built by
    config.EMBED_MODEL, state.last_post_id = 1300. The fake channel holds those 3 posts + tonight's 5."""
    monkeypatch.setattr(config, "EMBED_DIM", DIM)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    paths = {
        "POSTS_PATH": tmp_path / "posts.jsonl", "PRODUCTS_PATH": tmp_path / "products.jsonl",
        "EMBEDDINGS_PATH": tmp_path / "embeddings.npy", "EMBEDDINGS_IDS_PATH": tmp_path / "embeddings_ids.json",
        "EMBEDDINGS_META_PATH": tmp_path / "embeddings_meta.json", "STATE_PATH": tmp_path / "state.json",
        "LOG_PATH": tmp_path / "log.jsonl",
    }
    for name, path in paths.items():
        monkeypatch.setattr(config, name, path)
    paths["POSTS_PATH"].write_text("".join(json.dumps(dataclasses.asdict(p), ensure_ascii=False) + "\n"
                                           for p in OLD_POSTS), encoding="utf-8")
    paths["PRODUCTS_PATH"].write_text("".join(json.dumps(dataclasses.asdict(p), ensure_ascii=False) + "\n"
                                              for p in products), encoding="utf-8")
    np.save(paths["EMBEDDINGS_PATH"], np.stack([_unit(normalise(product_text(p))) for p in products]))
    paths["EMBEDDINGS_IDS_PATH"].write_text(json.dumps([p.id for p in products]))
    paths["EMBEDDINGS_META_PATH"].write_text(json.dumps({"model": config.EMBED_MODEL, "dim": DIM,
                                                         "n": len(products)}))
    paths["STATE_PATH"].write_text(json.dumps({"last_post_id": 1300}))

    channel = OLD_POSTS + NEW_POSTS
    fetch_calls: list[dict] = []

    def fake_fetch(channel_name, min_id=0, limit=500):
        fetch_calls.append({"channel": channel_name, "min_id": min_id, "limit": limit})
        return [p for p in channel if p.id > min_id][:limit]

    monkeypatch.setattr(fetch, "fetch", fake_fetch)
    return Shop(dir=tmp_path, posts=paths["POSTS_PATH"], products=paths["PRODUCTS_PATH"],
                npy=paths["EMBEDDINGS_PATH"], ids=paths["EMBEDDINGS_IDS_PATH"], meta=paths["EMBEDDINGS_META_PATH"],
                state=paths["STATE_PATH"], ingest_log=tmp_path / "gemini_ingest.jsonl",
                fetch_calls=fetch_calls, channel=channel)


@pytest.fixture
def gemini(monkeypatch):
    """Install a FakeClient behind llm.client(): `script` answers generate_content (last entry repeats),
    `embed` answers embed_content in order (then fake vectors)."""
    def install(script=(_record_products,), embed=()):
        fake = FakeClient(list(script), embed_script=list(embed))
        monkeypatch.setattr(llm, "client", lambda: fake)
        return fake
    return install


@pytest.fixture
def loaded_search(monkeypatch, shop):
    """search loaded from the shop's files, as the running bot would be; module state restored afterwards."""
    for name in ("PRODUCTS", "_by_id", "_matrix", "_ids", "_loaded_sig"):
        monkeypatch.setattr(search, name, getattr(search, name))
    search.reload()
    return shop


def _load(path) -> list[Product]:
    return index._load_products(path)


def _new_texts(shop) -> list[str]:
    return [normalise(product_text(p)) for p in _load(shop.products) if p.id in NEW_CAPTIONS]


# --- the happy night -------------------------------------------------------------------------------------

def test_night_run_fetches_from_last_post_id_and_indexes_only_new_posts(shop, gemini, products):
    old_matrix = np.load(shop.npy)
    gemini()
    assert ingest.run() is True
    assert [c["min_id"] for c in shop.fetch_calls] == [1300]
    assert _ids(shop.posts) == [900, 1234, 1300] + NEW_IDS
    assert _ids(shop.products) == [p.id for p in products] + NEW_IDS
    ids = json.loads(shop.ids.read_text())
    assert ids == [p.id for p in products] + NEW_IDS
    matrix = np.load(shop.npy)
    assert matrix.shape == (8, DIM) and matrix.dtype == np.float32
    assert np.array_equal(matrix[:3], old_matrix), "existing rows are copied, never re-embedded"
    for row, text in zip(matrix[3:], _new_texts(shop)):
        assert np.allclose(row, _unit(text), atol=1e-5)
    assert json.loads(shop.meta.read_text()) == {"model": config.EMBED_MODEL, "dim": DIM, "n": 8}
    assert _state(shop)["last_post_id"] == 1314


def test_five_new_posts_cost_at_most_three_gemini_requests(shop, gemini):
    fake = gemini()
    assert ingest.run() is True
    assert len(fake.calls) == 1, "5 posts fit one extraction batch"
    assert len(fake.embed_calls) == 1, "5 texts fit one embedding request"
    assert len(fake.calls) + len(fake.embed_calls) <= 3
    prompt = dump(fake.calls[0]["contents"])
    assert sorted(int(x) for x in re.findall(r"### id=(\d+)", prompt)) == NEW_IDS   # old posts not re-extracted
    assert list(fake.embed_calls[0]["contents"]) == _new_texts(shop)                 # old products not re-embedded


def test_every_gemini_request_is_logged_for_stats(shop, gemini):
    assert ingest.INGEST_LOG_NAME == bot.INGEST_LOG_NAME
    fake = gemini()
    ingest.run()
    lines = _jsonl(shop.ingest_log)
    assert len(lines) == len(fake.calls) + len(fake.embed_calls) == 2
    today = datetime.date.today()
    for line in lines:
        assert datetime.datetime.fromisoformat(line["ts"]).date() == today
    assert sorted(line["kind"] for line in lines) == ["embed", "extract"]
    embed_line = next(line for line in lines if line["kind"] == "embed")
    assert embed_line["model"] == config.EMBED_MODEL and embed_line["n"] == 5
    extract_line = next(line for line in lines if line["kind"] == "extract")
    assert extract_line["model"] == config.GEMINI_EXTRACT_MODEL and extract_line["n"] == 5
    stats = bot.compute_stats(today)
    assert stats["gemini_today"] == 2
    assert stats["last_index_at"] == _state(shop)["last_index_at"]


def test_last_index_at_is_written_on_success_and_keeps_last_post_id(shop, gemini):
    gemini()
    before = datetime.datetime.now().replace(microsecond=0)
    assert ingest.run() is True
    state = _state(shop)
    assert state["last_post_id"] == 1314
    stamp = datetime.datetime.fromisoformat(state["last_index_at"])
    assert before - datetime.timedelta(seconds=1) <= stamp <= datetime.datetime.now()


def test_night_with_nothing_new_sends_no_request_and_does_not_rewrite_the_index(shop, gemini):
    gemini()
    assert ingest.run() is True
    stats = {p.name: p.stat().st_mtime_ns for p in shop.index_files + (shop.products,)}
    lines = len(_jsonl(shop.ingest_log))
    fake = gemini()
    assert ingest.run() is True
    assert fake.calls == [] and fake.embed_calls == []
    assert {p.name: p.stat().st_mtime_ns for p in shop.index_files + (shop.products,)} == stats
    assert len(_jsonl(shop.ingest_log)) == lines
    assert [c["min_id"] for c in shop.fetch_calls] == [1300, 1314]
    assert "last_index_at" in _state(shop)


def test_main_exit_code_follows_the_run(shop, gemini, monkeypatch):
    gemini()
    assert ingest.main() == 0
    monkeypatch.setattr(fetch, "fetch", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("telegram down")))
    assert ingest.main() == 1


# --- failed nights: nothing half-written, pending stays pending --------------------------------------------

def test_extraction_429_leaves_products_and_index_intact_and_next_night_picks_them_up(shop, gemini, caplog):
    before = _snapshot(shop.products, *shop.index_files)
    fake = gemini(script=[quota_error()])                 # every extraction attempt: 429
    with caplog.at_level(logging.WARNING):
        assert ingest.run() is False
    assert _errors(caplog)
    assert _snapshot(shop.products, *shop.index_files) == before
    assert fake.embed_calls == []
    state = _state(shop)
    assert "last_index_at" not in state
    assert state["last_post_id"] == 1314                  # fetched posts are kept, they are pending
    assert _ids(shop.posts) == [900, 1234, 1300] + NEW_IDS
    assert len(_jsonl(shop.ingest_log)) == len(fake.calls) >= 2, "each retried attempt is a request"

    gemini()                                              # the next night: quota is back
    assert ingest.run() is True
    assert shop.fetch_calls[-1]["min_id"] == 1314
    assert _ids(shop.posts) == [900, 1234, 1300] + NEW_IDS   # nothing fetched twice
    assert _ids(shop.products) == [1234, 1300, 900] + NEW_IDS
    ids = json.loads(shop.ids.read_text())
    assert sorted(ids) == sorted(set(ids)) and len(ids) == 8
    assert "last_index_at" in _state(shop)


def test_embedding_429_leaves_the_index_intact_and_next_night_embeds_the_pending(shop, gemini, caplog):
    before = _snapshot(*shop.index_files)
    fake = gemini(embed=[quota_error() for _ in range(index.RETRIES)])
    with caplog.at_level(logging.WARNING):
        assert ingest.run() is False
    assert _errors(caplog)
    assert _snapshot(*shop.index_files) == before
    assert _ids(shop.products) == [1234, 1300, 900] + NEW_IDS   # the whole extraction batch, once
    assert "last_index_at" not in _state(shop)
    assert len(_jsonl(shop.ingest_log)) == len(fake.calls) + len(fake.embed_calls) == 1 + index.RETRIES

    fake = gemini()
    assert ingest.run() is True
    assert fake.calls == [], "already extracted: no second extraction"
    assert len(fake.embed_calls) == 1
    assert list(fake.embed_calls[0]["contents"]) == _new_texts(shop)
    assert json.loads(shop.ids.read_text()) == [1234, 1300, 900] + NEW_IDS
    assert _ids(shop.products) == [1234, 1300, 900] + NEW_IDS
    assert "last_index_at" in _state(shop)


def test_failed_fetch_changes_nothing_and_sends_no_gemini_request(shop, gemini, monkeypatch, caplog):
    def down(*a, **k):
        raise ConnectionError("telegram down")
    monkeypatch.setattr(fetch, "fetch", down)
    before = _snapshot(*shop.all_files)
    fake = gemini()
    with caplog.at_level(logging.WARNING):
        assert ingest.run() is False
    assert _errors(caplog)
    assert _snapshot(*shop.all_files) == before
    assert fake.calls == [] and fake.embed_calls == []
    assert _jsonl(shop.ingest_log) == []


@pytest.mark.parametrize("meta, dim, mentions", [
    ({"model": OLD_MODEL, "dim": DIM, "n": 3}, DIM, OLD_MODEL),
    ({"model": config.EMBED_MODEL, "dim": 16, "n": 3}, 16, "16"),
    (None, DIM, "shop_assistant.index"),
], ids=["other-model", "other-dim", "no-meta"])
def test_matrix_from_another_model_is_never_fully_re_embedded_at_night(shop, gemini, caplog, products,
                                                                       meta, dim, mentions):
    np.save(shop.npy, np.ones((3, dim), dtype=np.float32))
    if meta is None:
        shop.meta.unlink()
    else:
        shop.meta.write_text(json.dumps(meta))
    before = _snapshot(*shop.index_files)
    fake = gemini()
    with caplog.at_level(logging.WARNING):
        assert ingest.run() is False
    assert fake.embed_calls == [], "a full re-embed is a manual `python -m shop_assistant.index`"
    assert _snapshot(*shop.index_files) == before
    assert "last_index_at" not in _state(shop)
    errs = _errors(caplog)
    assert errs and any(mentions in r.getMessage() for r in errs), [r.getMessage() for r in errs]

    fake = gemini()
    with pytest.raises(index.ModelMismatch):
        index.update_index()
    assert fake.embed_calls == []
    assert _snapshot(*shop.index_files) == before


# --- index.update_index() directly -------------------------------------------------------------------------

def test_update_index_embeds_only_missing_ids_and_appends_rows(shop, gemini, products):
    kept = [products[0], products[2]]                      # 1234 and 900 are in the matrix, 1300 is not
    old = np.stack([_unit(normalise(product_text(p))) for p in kept])
    np.save(shop.npy, old)
    shop.ids.write_text(json.dumps([p.id for p in kept]))
    shop.meta.write_text(json.dumps({"model": config.EMBED_MODEL, "dim": DIM, "n": 2}))
    fake = gemini()
    assert index.update_index() == 1
    [call] = fake.embed_calls
    assert list(call["contents"]) == [normalise(product_text(products[1]))]
    assert json.loads(shop.ids.read_text()) == [1234, 900, 1300]
    matrix = np.load(shop.npy)
    assert np.array_equal(matrix[:2], old)
    assert np.allclose(matrix[2], _unit(normalise(product_text(products[1]))), atol=1e-5)
    assert json.loads(shop.meta.read_text()) == {"model": config.EMBED_MODEL, "dim": DIM, "n": 3}


def test_update_index_without_a_matrix_builds_it_from_all_products(shop, gemini, products):
    for p in shop.index_files:
        p.unlink()
    fake = gemini()
    assert index.update_index() == 3
    assert [t for c in fake.embed_calls for t in c["contents"]] == [normalise(product_text(p)) for p in products]
    assert json.loads(shop.ids.read_text()) == [p.id for p in products]
    assert json.loads(shop.meta.read_text()) == {"model": config.EMBED_MODEL, "dim": DIM, "n": 3}


def test_update_index_failure_writes_nothing(shop, gemini, products):
    shop.ids.write_text(json.dumps([1234, 1300]))
    np.save(shop.npy, np.load(shop.npy)[:2])
    shop.meta.write_text(json.dumps({"model": config.EMBED_MODEL, "dim": DIM, "n": 2}))
    before = _snapshot(*shop.index_files)
    gemini(embed=[server_error() for _ in range(index.RETRIES)])
    with pytest.raises(errors.ServerError):
        index.update_index()
    assert _snapshot(*shop.index_files) == before
    assert not [p for p in shop.dir.iterdir() if p.name.endswith(".tmp")], "no temp files left behind"


# --- the running bot sees the new index without a restart ------------------------------------------------

def test_reload_if_changed_notices_newer_files_only(loaded_search, gemini):
    shop = loaded_search
    assert search.reload_if_changed() is False
    gemini()
    assert ingest.run() is True
    _later(shop.products, *shop.index_files)
    assert search.reload_if_changed() is True
    assert {p.id for p in search.PRODUCTS} >= set(NEW_IDS)
    assert search._matrix.shape == (8, DIM) and search._ids[-5:] == NEW_IDS
    assert search.reload_if_changed() is False


def test_unchanged_files_are_not_read_again(loaded_search, monkeypatch):
    def no_read(*a, **k):
        raise AssertionError("nothing changed on disk: no reload expected")
    monkeypatch.setattr(search, "load_products", no_read)
    monkeypatch.setattr(search, "_load_matrix", no_read)
    assert search.reload_if_changed() is False
    assert [p.id for p in search.find_products(keywords=["krossovka"])] == [1300]


def test_customer_search_finds_tonights_post_without_restart(loaded_search, gemini):
    shop = loaded_search
    assert search.find_products(keywords=["palto"]) == []
    gemini()
    assert ingest.run() is True
    _later(shop.products, *shop.index_files)
    assert [p.id for p in search.find_products(keywords=["palto"])] == [1310]      # filter path, no reload() call
    palto = next(p for p in _load(shop.products) if p.id == 1310)
    fake = gemini(embed=[embed_response([normalise(product_text(palto))], DIM)])   # query lands on the palto row
    results = search.semantic_search("ayollar uchun issiq kiyim", limit=3)
    assert len(fake.embed_calls) == 1
    assert results and results[0].id == 1310


def test_customer_query_embeddings_are_not_counted_as_ingestion(loaded_search, gemini):
    shop = loaded_search
    gemini()
    ingest.run()
    lines = len(_jsonl(shop.ingest_log))
    gemini()
    search.semantic_search("qishki kurtka")
    assert len(_jsonl(shop.ingest_log)) == lines


# --- systemd units (D-7: a separate oneshot unit on a timer, never inside the bot) ---------------------------

DEPLOY = config.ROOT / "deploy"


def _unit_file(name: str) -> configparser.ConfigParser:
    path = DEPLOY / name
    assert path.exists(), f"{path} missing"
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str                                   # systemd keys are case-sensitive
    cp.read_string(path.read_text(encoding="utf-8"))
    return cp


def test_timer_runs_nightly_and_catches_up_after_downtime():
    timer = _unit_file("shop-assistant-ingest.timer")
    cal = timer.get("Timer", "OnCalendar", fallback="").strip()
    assert cal, "OnCalendar missing"
    m = re.search(r"(\d{1,2}):(\d{2})", cal)
    assert m and 0 <= int(m.group(1)) <= 6, f"not a night time: {cal!r}"
    assert timer.get("Timer", "Persistent", fallback="").strip().lower() == "true"
    assert timer.get("Install", "WantedBy", fallback="").strip() == "timers.target"
    unit = timer.get("Timer", "Unit", fallback="shop-assistant-ingest.service").strip()
    assert unit == "shop-assistant-ingest.service"


def test_ingest_service_is_a_oneshot_that_runs_the_pipeline_from_the_repo():
    svc = _unit_file("shop-assistant-ingest.service")
    assert svc.get("Service", "Type", fallback="").strip() == "oneshot"
    assert svc.get("Service", "WorkingDirectory", fallback="").strip() == "%h/shop_assistant"
    exec_start = svc.get("Service", "ExecStart", fallback="").strip()
    assert exec_start.startswith("%h/shop_assistant/.venv/bin/python")
    assert re.search(r"\s-m\s+shop_assistant\.ingest\b", exec_start), exec_start
    text = (DEPLOY / "shop-assistant-ingest.service").read_text(encoding="utf-8")
    assert "systemctl" not in text, "the bot reloads by itself (search.reload_if_changed); never restart it"
    assert "Restart=always" not in text


def test_deploy_script_installs_and_enables_the_timer():
    text = (config.ROOT / "deploy.sh").read_text(encoding="utf-8")
    assert "shop-assistant-ingest.service" in text and "shop-assistant-ingest.timer" in text
    assert any("enable" in line and "timer" in line.lower() for line in text.splitlines()), \
        "deploy.sh must `systemctl --user enable --now` the ingest timer"
