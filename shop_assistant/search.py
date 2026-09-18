"""Filter search, semantic fallback, latest, faq. SDD §3.5, FR-8/10/11/12/16/24. Tickets #7, #8."""
import dataclasses
import datetime
import json
from pathlib import Path

from shop_assistant import config
from shop_assistant.models import FaqEntry, Product
from shop_assistant.textnorm import normalise


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
            prods.append(_parse_product(json.loads(line)))
    return prods


PRODUCTS: list[Product] = load_products()


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

    today = datetime.date.today().isoformat()
    return [
        dataclasses.replace(p, stale=is_stale(p.date, today, config.STALE_DAYS))
        for p in results
    ]


def latest_posts(n: int = 5, products: list[Product] | None = None) -> list[Product]:
    """Most recent n products by date (FR-12)."""
    return find_products(limit=n, products=products)


def semantic_search(text: str, max_price: int | None = None, limit: int = 5) -> list[Product]:
    """index.embed([normalise(text)]) — D-3 —, cosine over the matrix, then price filter, top-k (FR-10)."""
    raise NotImplementedError("ticket #8")


def cosine_top_k(query, matrix, k: int) -> list[int]:
    """Indices of the k rows of `matrix` most similar to `query` (pure numpy)."""
    raise NotImplementedError("ticket #8")


def search_faq(text: str, limit: int = 3) -> list[FaqEntry]:
    """Semantic search over faq entries. Stub returns [] until R2 (ticket #16)."""
    raise NotImplementedError("ticket #8")


def reload() -> None:
    """Re-read products.jsonl + .npy from disk (admin /reindex)."""
    global PRODUCTS
    PRODUCTS = load_products()


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:])
    print(find_products(keywords=q.split()))
    try:
        print(semantic_search(q))
    except NotImplementedError as e:
        print(f"semantic_search: {e}")
