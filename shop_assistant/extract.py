"""Footer strip + LLM structured extraction (Gemini via google-genai). SDD §3.2, FR-3a/3b/4. Tickets #5, #23."""
import dataclasses
import json
import logging
import re
import time
import unicodedata

import httpx
from google.genai import errors, types

from shop_assistant import config, llm
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

# One separator per number: "630.000 399.000" (old/new price) is two numbers, not 630000399000.
_PRICE_RE = re.compile(r"(\d{1,3}(?:\.\d{3})+|\d{1,3}(?: \d{3})+|\d+)\s*(ming|минг|k)?", re.IGNORECASE)


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
  boshqa also covers chatter and announcements that name no concrete item and give no price and
  no sizes (a bare username, a greeting, "Kutib qoling", a discount notice). A post that lists
  sizes or a price is a product even when its first line is a teaser like "Yengi kolleksiya".

Prices are in so'm and written in shop notation. Convert to a plain integer:
  "980.000ming" -> 980000, "980 000 so'm" -> 980000, "1.200.000" -> 1200000, "350.000" -> 350000, "980k" -> 980000.
  A line like "Telegram obunachilariga narx: 320.000" is the subscriber price -> subscriber_price = 320000.
  The regular price goes in price. If no price is given, use null.

sizes: list of strings exactly as written (e.g. ["M","L","XL","2XL","3XL"] or ["40","41","42"]).
colors: list of colour names as written, [] if none.
season: kuz / qish / bahor / yoz if the post says so, else null.
keywords: 4-8 short search synonyms for the product across four scripts/languages:
  Uzbek Latin, Uzbek Cyrillic, Russian and English (e.g. dvoyka, двойка, костюм двойка, two-piece set, sport kostyum).

name: what the customer reads first. It is the PRODUCT TYPE (krossovka, kurtka, sviter, dvoyka, poyabzal,
  kostyum, ko'ylak, futbolka, sumka...) plus the brand/model if the post names one. 2-4 words.
  Never a slogan, greeting, emoji line or "new collection" text, even if that is the first line of the post.
  If the post does not say what the item is, use the category word: kiyim / poyabzal / aksessuar.
  Examples:
    "Yengi kolleksiya / Krossovka Nike Air / Razmer: 40 41 42"  -> name "Krossovka Nike Air"  (not "Yengi kolleksiya")
    "Okam bu modella siz uchun eng yahshilari / Qishki kurtka Barena"  -> name "Qishki kurtka Barena"  (not the slogan)
    "New collection / Razmer: M L XL / Narx: 350.000"  -> name "Kiyim"  (type unknown -> category word, not "New collection")
"""


log = logging.getLogger(__name__)

RETRIES = 5                # attempts per request on 429 / 5xx / timeout
BACKOFF_S = 5.0            # first wait; doubles each retry (5, 10, 20, 40 s) — lets a per-minute quota recover


def _sane_price(v: int | None) -> int | None:
    """Nothing in this shop costs under 10 000 so'm; the model returns 1 / 2026 for announcement posts."""
    return v if v is not None and v >= config.MIN_PRICE else None


def _fallback(post: Post, body: str) -> Product:
    return Product(
        id=post.id, date=post.date[:10], link=post.link,
        name=product_name({}, "boshqa", body), category="boshqa",
        price=None, subscriber_price=None, body=body,   # model skipped it: not a product post → no price guessing
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


_SLOGANS = frozenset({
    "yengi kolleksiya", "yangi kolleksiya", "yangi kolleksiya keldi", "yengi kolleksiya keldi",
    "new collection", "новая коллекция", "янги коллекция", "енги коллекция",
    "yangi model", "yengi model", "yangi tovar", "yengi tovar", "yangi mahsulot",
    "unknown", "none", "null", "n/a", "product", "mahsulot",
    "okam bu modella siz uchun eng yahshilari", "okam bu modella siz uchun eng yaxshilari",
    "assalomu alaykum", "assalomu aleykum", "salom", "hello", "привет", "здравствуйте",
    "sale", "aksiya", "акция", "chegirma", "скидка", "top", "hit", "хит",
})
_NAME_MAX_WORDS = 4
_TRAIL_PUNCT_RE = re.compile(r"[\s!?.,:;‼️…\-–—*#\"'«»()\[\]]+$")


def _norm_name(s: str) -> str:
    """Lower-case, collapse whitespace, drop emoji and trailing punctuation — for slogan matching."""
    s = _strip_emoji(s)
    s = " ".join(s.split()).lower()
    return _TRAIL_PUNCT_RE.sub("", s).strip()


def _is_bad_name(name: str) -> bool:
    if not name:
        return True
    low = name.lower()
    if "://" in low or low.startswith("www.") or "t.me/" in low:
        return True
    if not any(ch.isalpha() for ch in name):
        return True
    return _norm_name(name) in _SLOGANS


def _cap_words(s: str) -> str:
    words = s.split()[:_NAME_MAX_WORDS]
    out = " ".join(words)
    return out[:1].upper() + out[1:] if out else out


def product_name(item: dict, category: str, body: str) -> str:
    """Deterministic guard: return a customer-readable product name (type word + brand/model)
    when the model's `item["name"]` is a known slogan, empty, a URL or has no letters."""
    raw = item.get("name") if isinstance(item, dict) else None
    name = " ".join(str(raw).split()) if raw is not None else ""
    if not _is_bad_name(name):
        return name
    keywords = [" ".join(str(k).split()) for k in (item.get("keywords") or []) if isinstance(item, dict)]
    keywords = [k for k in keywords if any(ch.isalpha() for ch in k) and not _is_bad_name(k)]
    if keywords:
        return _cap_words(keywords[0])
    if category in config.CATEGORIES and category != "boshqa":
        return _cap_words(category)
    # boshqa / unknown category: the first body line that has letters, unless it is itself a slogan.
    first = next((" ".join(l.split()) for l in (body or "").splitlines() if any(ch.isalpha() for ch in l)), "")
    if first and not _is_bad_name(first):
        return _cap_words(first)
    return "Mahsulot"


def is_announcement(post: Post, body: str, price: int | None, sizes: tuple[str, ...]) -> bool:
    """Deterministic guard for the category: a post that carries no sellable signal — empty body,
    or no media together with no price and no sizes — is an announcement, not a product."""
    if not body.strip():
        return True
    return not post.has_media and price is None and not sizes


def _build(post: Post, body: str, item: dict) -> Product:
    category = item.get("category")
    if category not in config.CATEGORIES:
        category = "boshqa"
    price = _sane_price(_to_int(item.get("price")))
    if price is None:
        price = _sane_price(_body_price(body))
    if is_announcement(post, body, price, _to_str_tuple(item.get("sizes"))):
        category = "boshqa"
    name = product_name(item, category, body)
    season = item.get("season")
    return Product(
        id=post.id, date=post.date[:10], link=post.link,
        name=name, category=category,
        price=price, subscriber_price=_sane_price(_to_int(item.get("subscriber_price"))),
        sizes=_to_str_tuple(item.get("sizes")),
        colors=_to_str_tuple(item.get("colors")),
        keywords=tuple(normalise(str(k)) for k in item.get("keywords") or [] if str(k).strip()),
        season=str(season).strip() or None if season else None,
        body=body,
    )


def _request_config() -> types.GenerateContentConfig:
    """Force the record_products function (mode ANY, one allowed name); automatic calling off."""
    return types.GenerateContentConfig(
        system_instruction=_SYSTEM,
        tools=[types.Tool(function_declarations=llm.tool_declarations([_TOOL]))],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY, allowed_function_names=[_TOOL["name"]])),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        http_options=types.HttpOptions(timeout=llm.EXTRACT_TIMEOUT_S * 1000),
    )


def _is_transient(e: Exception) -> bool:
    """429 quota, 5xx and timeouts / connection drops are worth retrying; a bad key or request is not."""
    if isinstance(e, errors.ServerError):
        return True
    if isinstance(e, errors.ClientError):
        return e.code == 429
    return isinstance(e, httpx.TransportError)


def _generate(contents: str):
    """One generate_content request, retried with exponential backoff on transient errors.
    The last error is re-raised, so the caller writes nothing for this batch."""
    for attempt in range(RETRIES):
        try:
            return llm.client().models.generate_content(
                model=config.GEMINI_EXTRACT_MODEL, contents=contents, config=_request_config())
        except (errors.APIError, httpx.TransportError) as e:
            if not _is_transient(e) or attempt == RETRIES - 1:
                raise
            wait = BACKOFF_S * 2 ** attempt
            log.warning("extraction request failed (%s), retry %d/%d in %.0f s",
                        type(e).__name__, attempt + 1, RETRIES - 1, wait)
            time.sleep(wait)


def _call_llm(posts: list[Post], bodies: dict[int, str]) -> dict[int, dict]:
    user = "\n\n".join(f"### id={p.id}\n{bodies[p.id]}" for p in posts)
    resp = _generate(user)
    content = resp.candidates[0].content if resp.candidates else None
    calls = [p.function_call for p in (content.parts if content is not None else None) or []
             if p.function_call and p.function_call.name == _TOOL["name"]]
    if not calls:
        # Blocked / empty candidate or no function call despite mode ANY: the caller retries / splits.
        raise _LLMParseError("no record_products call in the response")
    items: dict[int, dict] = {}
    for call in calls:
        data = call.args or {}
        if isinstance(data, str):
            data = json.loads(data)
        for item in data.get("products", []) or []:
            try:
                items[int(item.get("id"))] = item
            except (TypeError, ValueError, AttributeError):
                continue
    return items


class _LLMParseError(RuntimeError):
    pass


def _extract_chunk(chunk: list[Post]) -> list[Product]:
    """One LLM call for `chunk`; on a parse failure retry once, then split in halves; a single
    post that still fails becomes a fallback record (no price guessing)."""
    bodies = {p.id: strip_footer(p.caption) for p in chunk}
    items: dict[int, dict] | None = None
    for _ in range(2):
        try:
            items = _call_llm(chunk, bodies)
            break
        except _LLMParseError:
            continue
    if items is None:
        if len(chunk) == 1:
            return [_fallback(chunk[0], bodies[chunk[0].id])]
        mid = len(chunk) // 2
        return _extract_chunk(chunk[:mid]) + _extract_chunk(chunk[mid:])
    return [(_build(p, bodies[p.id], items[p.id]) if items.get(p.id) else _fallback(p, bodies[p.id]))
            for p in chunk]


def extract_batch(posts: list[Post]) -> list[Product]:
    """Batches of config.EXTRACT_BATCH per call (NFR-3)."""
    if not posts:
        return []
    out: list[Product] = []
    for start in range(0, len(posts), config.EXTRACT_BATCH):
        out.extend(_extract_chunk(posts[start:start + config.EXTRACT_BATCH]))
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
            try:
                products = extract_batch(chunk)
            except (errors.APIError, httpx.HTTPError, RuntimeError) as e:
                # Quota / server / timeout after all retries, bad or missing key: stop here. Nothing of
                # this batch is written, so these posts stay un-extracted and the next run picks them up.
                log.error("extraction stopped at batch %d-%d of %d (%s: %s); %d products written this run",
                          start + 1, start + len(chunk), len(todo), type(e).__name__, e, count)
                return
            dt = time.perf_counter() - t0
            f.write("".join(json.dumps(dataclasses.asdict(prod), ensure_ascii=False) + "\n" for prod in products))
            f.flush()
            count += len(products)
            print(f"{count}/{len(todo)} products  (batch of {len(chunk)} in {dt:.1f}s)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
