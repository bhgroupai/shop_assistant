"""Embed products with Ollama (bge-m3), persist matrix. SDD §3.3, FR-5/6. Ticket #6."""
import json
import os
import tempfile
import time
from pathlib import Path

import httpx   # ollama's transport; ConnectError/ReadTimeout live here
import numpy as np
import ollama

from shop_assistant import config
from shop_assistant.models import Product
from shop_assistant.textnorm import normalise

_RETRIES = 3
_RETRY_SLEEP = 1.0   # seconds


def product_text(p: Product) -> str:
    """Text that gets embedded: name + ' ' + body + ' ' + ' '.join(keywords)."""
    return f"{p.name} {p.body} {' '.join(p.keywords)}".strip()


def _embed_chunk(client: "ollama.Client", chunk: list[str]) -> list[list[float]]:
    """One /api/embed call, retried on connection errors."""
    last: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            return client.embed(model=config.EMBED_MODEL, input=chunk)["embeddings"]
        except (httpx.TransportError, ConnectionError, TimeoutError, OSError) as e:
            last = e
            if attempt < _RETRIES - 1:
                time.sleep(_RETRY_SLEEP)
    assert last is not None
    raise last


def embed(texts: list[str]) -> np.ndarray:
    """Ollama /api/embed with config.EMBED_MODEL at config.OLLAMA_URL, batches of config.EMBED_BATCH → float32[N, D]."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    client = ollama.Client(host=config.OLLAMA_URL)
    rows: list[list[float]] = []
    for start in range(0, len(texts), config.EMBED_BATCH):
        rows.extend(_embed_chunk(client, texts[start:start + config.EMBED_BATCH]))
    return np.asarray(rows, dtype=np.float32)


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
    """Rewrite embeddings.npy + embeddings_ids.json for all products (D-4). Embeds normalise(product_text(p)) — D-3. Returns N."""
    products = _load_products()
    ids = [p.id for p in products]
    matrix = embed([normalise(product_text(p)) for p in products])

    def _write_npy(tmp: str) -> None:
        with open(tmp, "wb") as f:
            np.save(f, matrix)

    def _write_ids(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ids, f)

    _atomic_write(config.EMBEDDINGS_PATH, _write_npy)
    _atomic_write(config.EMBEDDINGS_IDS_PATH, _write_ids)
    return len(products)


def reindex_faq() -> int:
    """Same for faq.jsonl → faq_embeddings.npy (R2, ticket #16)."""
    raise NotImplementedError("ticket #16")


if __name__ == "__main__":
    print(reindex())
