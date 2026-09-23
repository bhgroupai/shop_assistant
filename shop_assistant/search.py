"""Filter search, semantic fallback, latest, faq. SDD §3.5, FR-8/10/11/12/16/24. Tickets #7, #8."""
import dataclasses
import datetime
import json
import logging
from pathlib import Path

import numpy as np

from shop_assistant import config
from shop_assistant.models import FaqEntry, Product
from shop_assistant.textnorm import normalise

log = logging.getLogger(__name__)


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


def load_products(path: Path | None = None) -> list[Product]:
    if path is None:
        path = config.PRODUCTS_PATH
    if not path.exists():
        return []
    prods: list[Product] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = _parse_product(json.loads(line))
            if not p.keywords and not p.sizes and p.price is None:
                continue   # announcement / non-product post (extract fallback): never a search result
            prods.append(p)
    return prods


def _no_matrix() -> tuple[np.ndarray, list[int]]:
    return np.zeros((0, config.EMBED_DIM), dtype=np.float32), []


def _load_matrix() -> tuple[np.ndarray, list[int]]:
    """embeddings.npy + embeddings_ids.json → (float32[N, D], ids). Missing → empty, quietly.
    Built by another embedding model / dimension, no embeddings_meta.json, or shapes that disagree →
    one ERROR and empty: semantic search stays off until `python -m shop_assistant.index` (#23.5)."""
    if not (config.EMBEDDINGS_PATH.exists() and config.EMBEDDINGS_IDS_PATH.exists()):
        return _no_matrix()
    where = f"{config.EMBEDDINGS_PATH}"
    fix = f"re-index with {config.EMBED_MODEL} ({config.EMBED_DIM}-d): python -m shop_assistant.index"
    try:
        matrix = np.load(config.EMBEDDINGS_PATH).astype(np.float32, copy=False)
        with open(config.EMBEDDINGS_IDS_PATH, "r", encoding="utf-8") as f:
            ids = [int(i) for i in json.load(f)]
        meta = None
        if config.EMBEDDINGS_META_PATH.exists():
            with open(config.EMBEDDINGS_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
    except (OSError, ValueError) as e:
        log.error("semantic search off: cannot read %s (%s); %s", where, e, fix)
        return _no_matrix()
    if not isinstance(meta, dict):
        log.error("semantic search off: %s has no model name (%s missing), expected %s; %s",
                  where, config.EMBEDDINGS_META_PATH.name, config.EMBED_MODEL, fix)
        return _no_matrix()
    if meta.get("model") != config.EMBED_MODEL:
        log.error("semantic search off: %s was built by %s, config.EMBED_MODEL is %s; %s",
                  where, meta.get("model"), config.EMBED_MODEL, fix)
        return _no_matrix()
    if meta.get("dim") != config.EMBED_DIM:
        log.error("semantic search off: %s has dimension %s, config.EMBED_DIM is %s; %s",
                  where, meta.get("dim"), config.EMBED_DIM, fix)
        return _no_matrix()
    if matrix.ndim != 2 or matrix.shape != (len(ids), config.EMBED_DIM):
        log.error("semantic search off: %s has shape %s but %d ids and dimension %s; %s",
                  where, matrix.shape, len(ids), config.EMBED_DIM, fix)
        return _no_matrix()
    return matrix, ids


PRODUCTS: list[Product] = load_products()
_by_id: dict[int, Product] = {p.id: p for p in PRODUCTS}
_matrix, _ids = _load_matrix()


def is_stale(date: str, today: str, stale_days: int) -> bool:
    """True when `date` (ISO) is more than stale_days before `today` (FR-16)."""
    d_product = datetime.date.fromisoformat(date.split("T")[0])
    d_today = datetime.date.fromisoformat(today.split("T")[0])
    return (d_today - d_product).days > stale_days


def keyword_match(customer_keywords: list[str], product: Product) -> bool:
    """Any normalised customer keyword is a substring of any product keyword or of normalise(name)."""
    norm_kws = [normalise(k) for k in customer_keywords if normalise(k)]
    if not norm_kws:
        return False
    norm_name = normalise(product.name)
    norm_prod_kws = [normalise(pk) for pk in product.keywords if normalise(pk)]
    for kw in norm_kws:
        if kw in norm_name:
            return True
        for pk in norm_prod_kws:
            if kw in pk:
                return True
    return False


def is_sellable(p: Product) -> bool:
    """Announcements, teasers and chatter are extracted as category `boshqa`; they are kept in
    products.jsonl for the record but never shown to a customer (FR-11)."""
    return p.category != "boshqa"


def find_products(category: str | None = None, min_price: int | None = None,
                  max_price: int | None = None, size: str | None = None,
                  color: str | None = None, keywords: list[str] | None = None,
                  limit: int = 5, products: list[Product] | None = None) -> list[Product]:
    """ANDed filters over products.jsonl; newest first; each result carries `stale` (FR-8, FR-16)."""
    catalog = PRODUCTS if products is None else products
    results: list[Product] = []

    norm_category = normalise(category) if category is not None else None
    norm_size = normalise(size) if size is not None else None
    norm_color = normalise(color) if color is not None else None

    for p in catalog:
        if not is_sellable(p):
            continue
        if norm_category is not None and normalise(p.category) != norm_category:
            continue
        if min_price is not None and (p.price is None or p.price < min_price):
            continue
        if max_price is not None and (p.price is None or p.price > max_price):
            continue
        if norm_size is not None and not any(normalise(s) == norm_size for s in p.sizes):
            continue
        if norm_color is not None and not any(normalise(c) == norm_color for c in p.colors):
            continue
        if keywords and not keyword_match(keywords, p):
            continue
        results.append(p)

    results.sort(key=lambda p: (p.date, p.id), reverse=True)
    if limit is not None:
        if limit <= 0:
            return []
        results = results[:limit]

    return _with_stale(results)


def _with_stale(products: list[Product]) -> list[Product]:
    """Copies with `stale` computed against today (FR-16)."""
    today = datetime.date.today().isoformat()
    return [
        dataclasses.replace(p, stale=is_stale(p.date, today, config.STALE_DAYS))
        for p in products
    ]


def latest_posts(n: int = 5, products: list[Product] | None = None) -> list[Product]:
    """Most recent n products by date (FR-12)."""
    return find_products(limit=n, products=products)


def semantic_search(text: str, max_price: int | None = None, limit: int = 5) -> list[Product]:
    """index.embed([normalise(text)], kind="query") — D-3 —, cosine over the matrix, then price filter, top-k
    (FR-10). Fails fast (#23.5): if the query embedding fails for any reason → [] and one WARNING, never raises."""
    from shop_assistant import index   # lazy: index imports the Gemini SDK; avoids import cycles
    if limit <= 0 or _matrix.shape[0] == 0:
        return []
    try:
        q = index.embed([normalise(text)], kind="query")[0]
    except Exception as e:   # 429 / 5xx / timeout / invalid or missing key: the agent goes on without it
        log.warning("semantic_search: query embedding failed, no semantic results (%s: %s)",
                    type(e).__name__, str(e)[:200])
        return []
    results: list[Product] = []
    for i in cosine_top_k(q, _matrix, k=limit * 4):
        p = _by_id.get(_ids[i])
        if p is None or not is_sellable(p):
            continue
        if max_price is not None and (p.price is None or p.price > max_price):
            continue
        results.append(p)
        if len(results) >= limit:
            break
    return _with_stale(results)


def cosine_top_k(query, matrix, k: int) -> list[int]:
    """Indices of the k rows of `matrix` most similar to `query` (pure numpy)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or k <= 0:
        return []
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    matrix_n = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    query_n = query / (np.linalg.norm(query) + 1e-9)
    scores = matrix_n @ query_n
    return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


def search_faq(text: str, limit: int = 3) -> list[FaqEntry]:
    """Semantic search over faq entries. Stub returns [] until R2 (ticket #16)."""
    # R2: ticket #16 — FAQ store not built yet
    return []


_loaded_sig = None   # (mtime_ns, size) signature of the files the current catalog/matrix was loaded from (#18)


def reload_if_changed() -> bool:
    """Ticket #18: cheap check (os.stat only) whether products.jsonl / the embeddings files on disk differ
    from what was last loaded; if so reload() and return True, else False. Called by find_products /
    latest_posts / semantic_search so the bot picks up the nightly index without a restart."""
    raise NotImplementedError("ticket #18")


def reload() -> None:
    """Re-read products.jsonl + .npy from disk (admin /reindex)."""
    global PRODUCTS, _by_id, _matrix, _ids
    PRODUCTS = load_products()
    _by_id = {p.id: p for p in PRODUCTS}
    _matrix, _ids = _load_matrix()


if __name__ == "__main__":
    import sys
    def _line(p: Product) -> str:
        return (f"  {p.id} · {p.name} · {p.price} · {' '.join(p.sizes) or '-'} · {p.date}"
                f" · {'stale' if p.stale else 'fresh'}")

    q = " ".join(sys.argv[1:])
    print("filters:")
    for p in find_products(keywords=q.split()):
        print(_line(p))
    print("semantic:")
    try:
        if _matrix.shape[0] == 0:
            print(f"  no embeddings at {config.EMBEDDINGS_PATH} — run python -m shop_assistant.index")
        for p in semantic_search(q):
            print(_line(p))
    except Exception as e:   # no embeddings / embedding API down — keep the CLI usable (FR-24)
        print(f"  semantic_search unavailable: {type(e).__name__}: {e}")
