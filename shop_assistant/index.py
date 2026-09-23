"""Embed products with the Gemini embedding API, persist matrix + model name. SDD §3.3, FR-5/6. Tickets #6, #23.5."""
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import httpx   # google-genai's transport; timeouts / dropped connections surface as httpx errors
import numpy as np
from google.genai import errors, types

from shop_assistant import config, llm
from shop_assistant.models import Product
from shop_assistant.textnorm import normalise

log = logging.getLogger(__name__)

RETRIES = 5           # attempts per batch when embed(..., retry=True) (index CLI only)
BACKOFF_S = 10.0      # first wait; doubles each retry (10, 20, 40, 80 s) — outlasts the per-minute quota window
                      # (100 texts/min free tier, config.py). Re-index 2026-09-23: the 2nd batch got through on
                      # attempt 4 (after 70 s), so 4 attempts was too tight
EMBED_TIMEOUT_S = 10  # per embed request; the query path must not hold a customer turn for long

_TASK_TYPES = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}


def product_text(p: Product) -> str:
    """Text that gets embedded: name + ' ' + body + ' ' + ' '.join(keywords)."""
    return f"{p.name} {p.body} {' '.join(p.keywords)}".strip()


def _is_transient(e: Exception) -> bool:
    """429 quota, 5xx and timeouts / connection drops are worth retrying; a bad key or request is not."""
    if isinstance(e, errors.ServerError):
        return True
    if isinstance(e, errors.ClientError):
        return e.code == 429
    return isinstance(e, (httpx.TransportError, TimeoutError, ConnectionError))


def _embed_batch(batch: list[str], task_type: str, retry: bool) -> list[list[float]]:
    """One embed_content request for `batch`; with `retry`, transient errors are retried with growing sleeps."""
    attempts = RETRIES if retry else 1
    for attempt in range(attempts):
        try:
            resp = llm.client().models.embed_content(
                model=config.EMBED_MODEL, contents=list(batch),
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=config.EMBED_DIM,
                    http_options=types.HttpOptions(timeout=EMBED_TIMEOUT_S * 1000)))
        except Exception as e:
            if attempt == attempts - 1 or not _is_transient(e):
                raise
            delay = BACKOFF_S * 2 ** attempt
            log.warning("embed batch failed (%s), retry %d/%d in %.0f s",
                        type(e).__name__, attempt + 1, attempts - 1, delay)
            time.sleep(delay)
            continue
        vectors = [e.values for e in (resp.embeddings or [])]
        if len(vectors) != len(batch):
            raise RuntimeError(f"{config.EMBED_MODEL} returned {len(vectors)} embeddings for {len(batch)} texts")
        return vectors
    raise AssertionError("unreachable: the last attempt returns or raises")


def embed(texts: list[str], kind: str = "document", *, retry: bool = False) -> np.ndarray:
    """Gemini embeddings (ticket #23.5) → float32[N, config.EMBED_DIM], every row L2-normalised.

    One `llm.client().models.embed_content(model=config.EMBED_MODEL, contents=<list of str>,
    config=types.EmbedContentConfig(task_type=..., output_dimensionality=config.EMBED_DIM))` per batch of
    config.EMBED_BATCH texts; task_type RETRIEVAL_DOCUMENT for kind="document", RETRIEVAL_QUERY for
    kind="query"; any other kind → ValueError. Empty `texts` → (0, EMBED_DIM) array, no request.
    retry=False (query path): one attempt, SDK errors propagate. retry=True (index CLI / reindex): transient
    errors (429, 5xx, timeouts) are retried up to RETRIES attempts per batch with growing time.sleep;
    an invalid key (400) is never retried; the last error is raised when attempts run out.
    """
    if kind not in _TASK_TYPES:
        raise ValueError(f"embed kind must be one of {sorted(_TASK_TYPES)}, got {kind!r}")
    dim = config.EMBED_DIM
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    rows: list[list[float]] = []
    step = config.EMBED_BATCH
    for i in range(0, len(texts), step):
        rows.extend(_embed_batch(texts[i:i + step], _TASK_TYPES[kind], retry))
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape != (len(texts), dim):
        raise RuntimeError(f"{config.EMBED_MODEL} returned shape {matrix.shape}, expected {(len(texts), dim)}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms > 0, norms, 1.0)).astype(np.float32)


def _parse_product(d: dict) -> Product:
    return Product(
        id=d["id"],
        date=d["date"],
        link=d.get("link", ""),
        name=d["name"],
        category=d["category"],
        price=d.get("price"),
        subscriber_price=d.get("subscriber_price"),
        sizes=tuple(d.get("sizes") or ()),
        colors=tuple(d.get("colors") or ()),
        keywords=tuple(d.get("keywords") or ()),
        season=d.get("season"),
        body=d.get("body", ""),
        stale=d.get("stale", False),
    )


def _load_products(path: Path | None = None) -> list[Product]:
    """products.jsonl → Product list (lists → tuples). Missing file → []."""
    path = config.PRODUCTS_PATH if path is None else path
    if not path.exists():
        return []
    prods: list[Product] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prods.append(_parse_product(json.loads(line)))
    return prods


def _atomic_write(path: Path, writer) -> None:
    """Write via a temp file in the same dir, then os.replace (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def reindex() -> int:
    """Rewrite embeddings.npy + embeddings_ids.json + embeddings_meta.json for all products (D-4).
    Embeds normalise(product_text(p)) — D-3 — with retries (offline). Embedding fails → no file touched.
    Returns N."""
    products = _load_products()
    ids = [p.id for p in products]
    matrix = embed([normalise(product_text(p)) for p in products], kind="document", retry=True)
    meta = {"model": config.EMBED_MODEL, "dim": config.EMBED_DIM, "n": len(ids)}

    def _write_npy(tmp: str) -> None:
        with open(tmp, "wb") as f:
            np.save(f, matrix)

    def _write_json(value):
        def write(tmp: str) -> None:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(value, f)
        return write

    _atomic_write(config.EMBEDDINGS_PATH, _write_npy)
    _atomic_write(config.EMBEDDINGS_IDS_PATH, _write_json(ids))
    _atomic_write(config.EMBEDDINGS_META_PATH, _write_json(meta))   # last: the matrix counts only once it exists
    return len(products)


def reindex_faq() -> int:
    """Same for faq.jsonl → faq_embeddings.npy (R2, ticket #16)."""
    raise NotImplementedError("ticket #16")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(reindex())
