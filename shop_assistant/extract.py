"""Footer strip + LLM structured extraction (Ollama via Anthropic SDK). SDD §3.2, FR-3a/3b/4. Ticket #5."""
import dataclasses
import json
import re
import time
import unicodedata

from shop_assistant import config
from shop_assistant.models import Post, Product
from shop_assistant.textnorm import normalise

# --- FR-3a: footer / emoji ---------------------------------------------------

# +998, 9+ consecutive digits, or the local "90 123 45 67" / "(90) 123-45-67" layout.
_PHONE_RE = re.compile(r"\+998|\d{9,}|\(?\d{2,3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")
_HANDLE_RE = re.compile(r"(?<![\w.])@\w{3,}")
_BOILERPLATE = ("manzil", "dastavka", "yetkazib", "адрес", "доставка", "📍")
_EMOJI_RE = re.compile(
    "[" + chr(0x1F300) + "-" + chr(0x1FAFF)        # symbols & pictographs, emoticons, etc.
    + chr(0x2600) + "-" + chr(0x27BF)              # misc symbols, dingbats
    + chr(0xFE0F) + chr(0x200D) + "]"              # variation selector, ZWJ
)


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def _is_footer_line(line: str) -> bool:
    low = line.lower()
    if any(w in low for w in _BOILERPLATE):
        return True
    if _HANDLE_RE.search(line):
        return True
    return bool(_PHONE_RE.search(line))


def _strip_emoji(s: str) -> str:
    s = _EMOJI_RE.sub("", s)
    return "".join(ch for ch in s if unicodedata.category(ch) not in ("So", "Sk"))


def strip_footer(caption: str) -> str:
    """Drop lines with phone / @handle / 📍 / delivery boilerplate; strip emoji; collapse blank lines (FR-3a)."""
    out: list[str] = []
    for raw in caption.splitlines():
        if _is_footer_line(raw):
            continue
        line = _strip_emoji(raw).strip()
        if line:
            out.append(line)
    return "\n".join(out)


# --- FR-3b: price notation ---------------------------------------------------

_PRICE_RE = re.compile(r"(\d{1,3}(?:[.\s]\d{3})+|\d+)\s*(ming|минг|k)?", re.IGNORECASE)


def parse_price(s: str) -> int | None:
    """Shop notation → so'm: '980.000ming' → 980000, '1.200.000' → 1200000, 'narxi so'rang' → None (FR-3b)."""
    if not s:
        return None
    matches = list(_PRICE_RE.finditer(s))
    if not matches:
        return None
    # Prefer the first number that looks like money (thousands groups, suffix or >= 1000)
    # so that "2XL" in a size line does not win over "980.000ming".
    m = next((x for x in matches if x.group(2) or len(x.group(1)) > 3), matches[0])
    value = int(_digits_only(m.group(1)))
    if m.group(2) and value < 10000:
        value *= 1000
    return value or None


_PRICE_LINE_WORDS = ("narx", "нарх", "цена", "price", "so'm", "сум", "сўм")


def _body_price(body: str) -> int | None:
    """Price from the body: the first 'Narx…' line that is not the subscriber line, else parse_price(body)."""
    for line in body.splitlines():
        low = line.lower()
        if "obunachi" in low or "подписчик" in low:
            continue
        if any(w in low for w in _PRICE_LINE_WORDS):
            price = parse_price(line)
            if price is not None:
                return price
    return parse_price(body)


# --- FR-4: LLM extraction ----------------------------------------------------

_TOOL = {
    "name": "record_products",
    "description": "Record the structured product for every post in the message, one item per post id.",
    "input_schema": {
        "type": "object",
        "properties": {
            "products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "post id from the '### id=' header"},
                        "name": {"type": "string", "description": "short product name as written in the post"},
                        "category": {"type": "string", "enum": config.CATEGORIES},
                        "price": {"type": ["integer", "null"], "description": "regular price in so'm"},
                        "subscriber_price": {"type": ["integer", "null"],
                                             "description": "price for Telegram subscribers in so'm, if stated"},
                        "sizes": {"type": "array", "items": {"type": "string"}},
                        "colors": {"type": "array", "items": {"type": "string"}},
                        "keywords": {"type": "array", "items": {"type": "string"},
                                     "description": "4-8 search synonyms across uz-Latin, uz-Cyrillic, ru, en"},
                        "season": {"type": ["string", "null"],
                                   "description": "kuz / qish / bahor / yoz or null"},
                    },
                    "required": ["id", "name", "category", "price", "subscriber_price",
                                 "sizes", "colors", "keywords", "season"],
                },
            }
        },
        "required": ["products"],
    },
}

_SYSTEM = f"""You extract structured product records from Telegram posts of an Uzbek clothing shop.
Each post starts with a header line '### id=<id>'. Call record_products exactly once with one item per post,
using the same id.

Categories (choose exactly one): {", ".join(config.CATEGORIES)}.
  kiyim = clothing, poyabzal = shoes, aksessuar = accessories, boshqa = anything else.

Prices are in so'm and written in shop notation. Convert to a plain integer:
  "980.000ming" -> 980000, "980 000 so'm" -> 980000, "1.200.000" -> 1200000, "350.000" -> 350000, "980k" -> 980000.
  A line like "Telegram obunachilariga narx: 320.000" is the subscriber price -> subscriber_price = 320000.
  The regular price goes in price. If no price is given, use null.

sizes: list of strings exactly as written (e.g. ["M","L","XL","2XL","3XL"] or ["40","41","42"]).
colors: list of colour names as written, [] if none.
season: kuz / qish / bahor / yoz if the post says so, else null.
keywords: 4-8 short search synonyms for the product across four scripts/languages:
  Uzbek Latin, Uzbek Cyrillic, Russian and English (e.g. dvoyka, двойка, костюм двойка, two-piece set, sport kostyum).
"""


def _client():
    import anthropic
    return anthropic.Anthropic()


def _fallback(post: Post, body: str) -> Product:
    first = body.split("\n", 1)[0].strip() if body else ""
    return Product(
        id=post.id, date=post.date[:10], link=post.link,
        name=first or f"post {post.id}", category="boshqa",
        price=_body_price(body), subscriber_price=None, body=body,
    )


def _to_str_tuple(v) -> tuple[str, ...]:
    if not isinstance(v, list):
        return ()
    return tuple(str(x).strip() for x in v if str(x).strip())


def _to_int(v) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    return parse_price(str(v))


def _build(post: Post, body: str, item: dict) -> Product:
    category = item.get("category")
    if category not in config.CATEGORIES:
        category = "boshqa"
    price = _to_int(item.get("price"))
    if price is None:
        price = _body_price(body)
    name = str(item.get("name") or "").strip() or (body.split("\n", 1)[0] if body else f"post {post.id}")
    season = item.get("season")
    return Product(
        id=post.id, date=post.date[:10], link=post.link,
        name=name, category=category,
        price=price, subscriber_price=_to_int(item.get("subscriber_price")),
        sizes=_to_str_tuple(item.get("sizes")),
        colors=_to_str_tuple(item.get("colors")),
        keywords=tuple(normalise(str(k)) for k in item.get("keywords") or [] if str(k).strip()),
        season=str(season).strip() or None if season else None,
        body=body,
    )


def _call_llm(client, posts: list[Post], bodies: dict[int, str]) -> dict[int, dict]:
    user = "\n\n".join(f"### id={p.id}\n{bodies[p.id]}" for p in posts)
    resp = client.messages.create(
        model=config.MODEL,
        max_tokens=config.EXTRACT_MAX_TOKENS,
        system=_SYSTEM,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "record_products"},
        thinking=config.THINKING,
        messages=[{"role": "user", "content": user}],
    )
    items: dict[int, dict] = {}
    for block in resp.content:            # gemma4 may emit a `thinking` block first (SDD §3.2)
        if getattr(block, "type", None) != "tool_use":
            continue
        data = block.input
        if isinstance(data, str):
            data = json.loads(data)
        for item in (data or {}).get("products", []) or []:
            try:
                items[int(item.get("id"))] = item
            except (TypeError, ValueError):
                continue
    return items


def extract_batch(posts: list[Post]) -> list[Product]:
    """Batches of config.EXTRACT_BATCH per call (NFR-3)."""
    if not posts:
        return []
    client = _client()
    out: list[Product] = []
    for start in range(0, len(posts), config.EXTRACT_BATCH):
        chunk = posts[start:start + config.EXTRACT_BATCH]
        bodies = {p.id: strip_footer(p.caption) for p in chunk}
        items = _call_llm(client, chunk, bodies)
        for p in chunk:
            item = items.get(p.id)
            out.append(_build(p, bodies[p.id], item) if item else _fallback(p, bodies[p.id]))
    return out


def extract(post: Post) -> Product:
    """One forced tool-use call (`record_product`) → Product. Keywords in uz-Latin/uz-Cyrillic/ru/en, normalised (FR-4)."""
    return extract_batch([post])[0]


# --- CLI ---------------------------------------------------------------------

def _load_posts() -> list[Post]:
    posts: list[Post] = []
    if not config.POSTS_PATH.exists():
        return posts
    with config.POSTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            posts.append(Post(id=int(d["id"]), date=d["date"], link=d["link"],
                              caption=d.get("caption", ""), has_media=d.get("has_media", True)))
    return posts


def _done_ids() -> set[int]:
    ids: set[int] = set()
    if not config.PRODUCTS_PATH.exists():
        return ids
    with config.PRODUCTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(json.loads(line)["id"]))
    return ids


def main() -> None:
    """CLI: process posts not yet in products.jsonl."""
    done = _done_ids()
    todo = [p for p in _load_posts() if p.id not in done and p.caption.strip()]
    print(f"{len(done)} products already extracted; {len(todo)} posts to process")
    if not todo:
        return
    config.PRODUCTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with config.PRODUCTS_PATH.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), config.EXTRACT_BATCH):
            chunk = todo[start:start + config.EXTRACT_BATCH]
            t0 = time.perf_counter()
            products = extract_batch(chunk)
            dt = time.perf_counter() - t0
            for prod in products:
                f.write(json.dumps(dataclasses.asdict(prod), ensure_ascii=False) + "\n")
            f.flush()
            count += len(products)
            print(f"{count}/{len(todo)} products  (batch of {len(chunk)} in {dt:.1f}s)")


if __name__ == "__main__":
    main()
