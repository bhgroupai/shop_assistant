"""Ticket #23.5 — product/query embeddings on the Gemini embedding API; Ollama leaves the project.
Fake client only, no network (seam: tests/test_llm.py docstring; fake: tests/fake_gemini.py).

CONTRACT (read before implementing):

* `index.embed(texts, kind="document", *, retry=False)` -> float32[N, config.EMBED_DIM], rows L2-normalised,
  in input order. It reaches Gemini only through `llm.client().models.embed_content(
  model=config.EMBED_MODEL, contents=<the batch as a list of str>,
  config=types.EmbedContentConfig(task_type=..., output_dimensionality=config.EMBED_DIM))`,
  looking `llm.client` up at call time. task_type is "RETRIEVAL_DOCUMENT" for kind="document" and
  "RETRIEVAL_QUERY" for kind="query"; any other kind -> ValueError before any request.
  One request per batch of config.EMBED_BATCH texts. Empty input -> shape (0, EMBED_DIM), no request.
* retry=False (query path, the default): one attempt; SDK / transport errors propagate, no sleep.
  retry=True (index CLI: `reindex()` uses it): 429 / 5xx / timeouts are retried, at most `index.RETRIES`
  attempts per batch, sleeping with `time.sleep` for strictly growing delays; an invalid key (400) is
  never retried. When attempts run out the last error is raised.
* `reindex()` writes embeddings.npy, embeddings_ids.json (unchanged format: list of ids) and the new
  `config.EMBEDDINGS_META_PATH` (data/embeddings_meta.json) = {"model": EMBED_MODEL, "dim": EMBED_DIM, "n": N}.
  If embedding fails, none of the three files is touched.
* `search._load_matrix()` returns the matrix only when the meta file exists and its model == EMBED_MODEL,
  its dim == EMBED_DIM and the matrix is N x EMBED_DIM. Otherwise (e.g. the old bge-m3 1024-d matrix)
  it logs ONE ERROR (naming the stored model, e.g. "bge-m3", or the stored dimension, e.g. "1024";
  with no meta file, naming the expected config.EMBED_MODEL) and returns an empty matrix -> semantic search is off
  (semantic_search returns [] without sending a request). No embeddings at all -> empty, no ERROR.
* `search.semantic_search` embeds `[normalise(text)]` with kind="query". If that fails for any reason
  (429, 503, timeout, invalid key, missing key) it returns [] and logs exactly one WARNING; never raises.
"""
import dataclasses
import json
import logging
import math
import re
import time

import numpy as np
import pytest
from google.genai import errors

from shop_assistant import config, index, llm, search
from shop_assistant.index import product_text
from shop_assistant.textnorm import normalise
from tests.fake_gemini import (FakeClient, embed_config, embed_response, fake_vector, invalid_key_error, quota_error,
                               server_error, timeout_error)

DIM = 8   # small test dimension; embed() must follow config.EMBED_DIM, whatever it is

QUERY = "что-нибудь для холодной погоды"
TEXTS = ["Qishki kurtka Rang qora kok Razmer L XL kurtka jacket",
         "Krossovka Nike Air Razmer 40 41 42 43 krossovka sneakers",
         "Dvoyka Yangi model Dvoyka Razmer M L XL 2XL 3XL sport kostyum",
         "кроссовка для бега",
         "sviter qishki"]


@pytest.fixture(autouse=True)
def _dim_and_no_sleep(monkeypatch):
    monkeypatch.setattr(config, "EMBED_DIM", DIM, raising=False)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    return sleeps


@pytest.fixture
def sleeps(_dim_and_no_sleep):
    return _dim_and_no_sleep


@pytest.fixture
def gemini(monkeypatch):
    """Install a FakeClient behind llm.client(); `embed_script` answers embed_content calls in order."""
    def install(*embed_script):
        fake = FakeClient(embed_script=list(embed_script))
        monkeypatch.setattr(llm, "client", lambda: fake)
        return fake
    return install


def _unit(text: str) -> np.ndarray:
    v = np.asarray(fake_vector(text, DIM), dtype=np.float32)
    return v / np.linalg.norm(v)


# --- embed(): request shape ---------------------------------------------------

def test_embed_sends_model_document_task_type_and_dimension(gemini):
    fake = gemini()
    index.embed(TEXTS[:2])
    [call] = fake.embed_calls
    assert call["model"] == config.EMBED_MODEL
    assert list(call["contents"]) == TEXTS[:2]
    cfg = embed_config(call["config"])
    assert cfg.get("task_type") == "RETRIEVAL_DOCUMENT"
    assert cfg.get("output_dimensionality") == DIM


def test_query_kind_uses_retrieval_query_task_type(gemini):
    fake = gemini()
    index.embed([normalise(QUERY)], kind="query")
    [call] = fake.embed_calls
    assert embed_config(call["config"]).get("task_type") == "RETRIEVAL_QUERY"
    assert list(call["contents"]) == [normalise(QUERY)]


def test_explicit_document_kind_matches_default(gemini):
    fake = gemini()
    index.embed(TEXTS[:1], kind="document")
    assert embed_config(fake.embed_calls[0]["config"]).get("task_type") == "RETRIEVAL_DOCUMENT"


def test_unknown_kind_is_rejected_before_any_request(gemini):
    fake = gemini()
    with pytest.raises(ValueError):
        index.embed(TEXTS[:1], kind="faq")
    assert fake.embed_calls == []


# --- embed(): output ------------------------------------------------------------

def test_embed_returns_float32_n_by_dim(gemini):
    gemini()
    m = index.embed(TEXTS)
    assert isinstance(m, np.ndarray)
    assert m.dtype == np.float32
    assert m.shape == (len(TEXTS), DIM)


def test_rows_are_unit_length_and_in_input_order(gemini):
    gemini()
    m = index.embed(TEXTS)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    for row, text in zip(m, TEXTS):
        assert np.allclose(row, _unit(text), atol=1e-5)


def test_empty_input_gives_zero_rows_and_no_request(gemini):
    fake = gemini()
    m = index.embed([])
    assert m.shape == (0, DIM)
    assert m.dtype == np.float32
    assert fake.embed_calls == []


@pytest.mark.parametrize("n, batch", [(5, 2), (4, 2), (5, 5), (1, 128)])
def test_documents_are_sent_in_batches(gemini, monkeypatch, n, batch):
    monkeypatch.setattr(config, "EMBED_BATCH", batch)
    fake = gemini()
    m = index.embed(TEXTS[:n])
    assert len(fake.embed_calls) == math.ceil(n / batch)
    sent = [t for call in fake.embed_calls for t in call["contents"]]
    assert sent == TEXTS[:n]
    assert all(len(call["contents"]) <= batch for call in fake.embed_calls)
    assert m.shape == (n, DIM)


# --- embed(): errors and retry --------------------------------------------------

@pytest.mark.parametrize("error", [quota_error, server_error, timeout_error, invalid_key_error])
def test_query_path_fails_fast_without_sleeping(gemini, sleeps, error):
    fake = gemini(error())
    with pytest.raises(Exception) as exc:
        index.embed([normalise(QUERY)], kind="query")
    assert not isinstance(exc.value, NotImplementedError)
    assert len(fake.embed_calls) == 1
    assert sleeps == []


def test_retry_recovers_from_transient_errors_with_growing_sleeps(gemini, sleeps):
    fake = gemini(quota_error(), server_error(), timeout_error())
    m = index.embed(TEXTS[:2], retry=True)
    assert m.shape == (2, DIM)
    assert len(fake.embed_calls) == 4
    assert len(sleeps) == 3
    assert all(s > 0 for s in sleeps)
    assert sleeps == sorted(sleeps) and len(set(sleeps)) == len(sleeps), f"delays must grow: {sleeps}"


def test_retry_gives_up_after_retries_attempts(gemini, sleeps):
    assert 4 <= index.RETRIES <= 6
    fake = gemini(*[quota_error() for _ in range(index.RETRIES + 3)])
    with pytest.raises(errors.ClientError):
        index.embed(TEXTS[:2], retry=True)
    assert len(fake.embed_calls) == index.RETRIES
    assert len(sleeps) == index.RETRIES - 1
    assert sleeps == sorted(sleeps) and len(set(sleeps)) == len(sleeps)


def test_invalid_key_is_never_retried(gemini, sleeps):
    fake = gemini(invalid_key_error())
    with pytest.raises(errors.ClientError):
        index.embed(TEXTS[:2], retry=True)
    assert len(fake.embed_calls) == 1
    assert sleeps == []


# --- reindex(): matrix + ids + model name --------------------------------------

@pytest.fixture
def data_dir(tmp_path, monkeypatch, products):
    """Real conftest products in a temp products.jsonl; all embedding paths in tmp_path."""
    path = tmp_path / "products.jsonl"
    path.write_text("".join(json.dumps(dataclasses.asdict(p), ensure_ascii=False) + "\n" for p in products),
                    encoding="utf-8")
    monkeypatch.setattr(config, "PRODUCTS_PATH", path)
    monkeypatch.setattr(config, "EMBEDDINGS_PATH", tmp_path / "embeddings.npy")
    monkeypatch.setattr(config, "EMBEDDINGS_IDS_PATH", tmp_path / "embeddings_ids.json")
    monkeypatch.setattr(config, "EMBEDDINGS_META_PATH", tmp_path / "embeddings_meta.json", raising=False)
    return tmp_path


def _write_matrix(tmp_path, matrix, ids, meta=None):
    np.save(tmp_path / "embeddings.npy", np.asarray(matrix, dtype=np.float32))
    (tmp_path / "embeddings_ids.json").write_text(json.dumps(ids))
    if meta is not None:
        (tmp_path / "embeddings_meta.json").write_text(json.dumps(meta))


def test_reindex_writes_matrix_ids_and_model_name(gemini, data_dir, products, sleeps):
    fake = gemini(quota_error())                      # one 429 at index time: retried, not fatal
    n = index.reindex()
    assert n == len(products)
    matrix = np.load(data_dir / "embeddings.npy")
    assert matrix.shape == (len(products), DIM)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    assert json.loads((data_dir / "embeddings_ids.json").read_text()) == [p.id for p in products]
    meta = json.loads((data_dir / "embeddings_meta.json").read_text())
    assert meta["model"] == config.EMBED_MODEL
    assert meta["dim"] == DIM
    assert meta["n"] == len(products)
    assert len(sleeps) == 1
    sent = [t for call in fake.embed_calls[1:] for t in call["contents"]]
    assert sent == [normalise(product_text(p)) for p in products]           # D-3
    assert all(embed_config(c["config"]).get("task_type") == "RETRIEVAL_DOCUMENT" for c in fake.embed_calls)


def test_failed_reindex_leaves_old_files_untouched(gemini, data_dir, products, sleeps):
    old = np.ones((3, 1024), dtype=np.float32)
    _write_matrix(data_dir, old, [1234, 1300, 900])
    gemini(*[server_error() for _ in range(20)])
    with pytest.raises(errors.ServerError):
        index.reindex()
    assert np.array_equal(np.load(data_dir / "embeddings.npy"), old)
    assert json.loads((data_dir / "embeddings_ids.json").read_text()) == [1234, 1300, 900]
    assert not (data_dir / "embeddings_meta.json").exists()
    assert sleeps, "the index CLI must retry before giving up"


# --- search: query path -----------------------------------------------------------

@pytest.fixture
def indexed(monkeypatch, products):
    """search holds a matrix of the conftest products embedded like reindex() would (fake vectors)."""
    matrix = np.stack([_unit(normalise(product_text(p))) for p in products])
    monkeypatch.setattr(search, "_matrix", matrix)
    monkeypatch.setattr(search, "_ids", [p.id for p in products])
    monkeypatch.setattr(search, "_by_id", {p.id: p for p in products})
    return products


def test_semantic_search_embeds_normalised_query_with_query_kind(gemini, indexed):
    kurtka = indexed[2]
    fake = gemini(embed_response([normalise(product_text(kurtka))], DIM))   # query lands on the kurtka row
    results = search.semantic_search(QUERY, limit=2)
    [call] = fake.embed_calls
    assert list(call["contents"]) == [normalise(QUERY)]                  # D-3
    assert embed_config(call["config"]).get("task_type") == "RETRIEVAL_QUERY"
    assert results[0].id == kurtka.id


@pytest.mark.parametrize("error", [quota_error, server_error, timeout_error, invalid_key_error])
def test_semantic_search_failure_returns_empty_with_one_warning(gemini, indexed, caplog, sleeps, error):
    gemini(error())
    with caplog.at_level(logging.WARNING):
        results = search.semantic_search(QUERY)
    assert results == []
    assert [r.levelno for r in caplog.records if r.levelno >= logging.WARNING] == [logging.WARNING]
    assert sleeps == []


def test_semantic_search_without_api_key_returns_empty_with_one_warning(indexed, monkeypatch, caplog):
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm.genai, "Client", lambda **kw: pytest.fail("no client without a key"))
    with caplog.at_level(logging.WARNING):
        results = search.semantic_search(QUERY)
    assert results == []
    assert [r.levelno for r in caplog.records if r.levelno >= logging.WARNING] == [logging.WARNING]


# --- search: model guard on load ---------------------------------------------------

@pytest.fixture
def fresh_search(monkeypatch, data_dir):
    """Let search.reload() read data_dir; module state is restored after the test."""
    for name in ("PRODUCTS", "_by_id", "_matrix", "_ids"):
        monkeypatch.setattr(search, name, getattr(search, name))
    return data_dir


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_matrix_with_own_model_and_dim_is_loaded(fresh_search, products, caplog):
    matrix = np.stack([_unit(normalise(product_text(p))) for p in products])
    _write_matrix(fresh_search, matrix, [p.id for p in products],
                  {"model": config.EMBED_MODEL, "dim": DIM, "n": len(products)})
    with caplog.at_level(logging.WARNING):
        loaded, ids = search._load_matrix()
    assert loaded.shape == (len(products), DIM)
    assert ids == [p.id for p in products]
    assert _errors(caplog) == []


def _assert_refused(fresh_search, gemini, caplog, must_mention: str):
    fake = gemini()
    with caplog.at_level(logging.WARNING):
        loaded, ids = search._load_matrix()
        search.reload()
    assert loaded.shape[0] == 0 and ids == []
    errs = _errors(caplog)
    assert len(errs) == 2, "one ERROR per load (the direct call + reload)"
    assert all(must_mention in r.getMessage() for r in errs), [r.getMessage() for r in errs]
    assert search.semantic_search(QUERY) == []
    assert fake.embed_calls == [], "semantic search must be off: no request"


def test_matrix_from_another_model_is_refused(fresh_search, gemini, products, caplog):
    matrix = np.stack([_unit(normalise(product_text(p))) for p in products])     # same dim, other model
    _write_matrix(fresh_search, matrix, [p.id for p in products], {"model": "bge-m3", "dim": DIM, "n": 3})
    _assert_refused(fresh_search, gemini, caplog, "bge-m3")


def test_matrix_with_another_dimension_is_refused(fresh_search, gemini, products, caplog):
    matrix = np.ones((len(products), 1024), dtype=np.float32)
    _write_matrix(fresh_search, matrix, [p.id for p in products],
                  {"model": config.EMBED_MODEL, "dim": 1024, "n": len(products)})
    _assert_refused(fresh_search, gemini, caplog, "1024")


def test_old_matrix_without_model_name_is_refused(fresh_search, gemini, products, caplog):
    matrix = np.ones((len(products), 1024), dtype=np.float32)                   # the bge-m3 matrix on the server
    _write_matrix(fresh_search, matrix, [p.id for p in products])
    _assert_refused(fresh_search, gemini, caplog, config.EMBED_MODEL)


def test_no_embeddings_yet_is_quiet(fresh_search, caplog):
    with caplog.at_level(logging.WARNING):
        loaded, ids = search._load_matrix()
    assert loaded.shape[0] == 0 and ids == []
    assert _errors(caplog) == []


# --- the grep (config checks: tests/test_config.py) ----------------------------------------------------------

GONE = re.compile(r"ollama|anthropic|gemma|bge-m3", re.IGNORECASE)


def test_no_local_model_or_anthropic_trace_in_code_or_requirements():
    files = sorted((config.ROOT / "shop_assistant").glob("*.py")) + [config.ROOT / "requirements.txt"]
    hits = [f"{f.name}:{n}: {line.strip()}"
            for f in files
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1)
            if GONE.search(line)]
    assert hits == []


def test_tests_do_not_import_anthropic_or_ollama():
    imports = re.compile(r"^\s*(import|from)\s+(anthropic|ollama)\b", re.MULTILINE)
    offenders = [f.name for f in sorted((config.ROOT / "tests").glob("*.py")) if imports.search(f.read_text())]
    assert offenders == []
