"""@beta_tool wrappers with compact text output. SDD §3.6, FR-11, NFR-2. Ticket #9."""
from anthropic import beta_tool

from shop_assistant import config, search
from shop_assistant.models import FaqEntry, Product

NO_RESULTS = "no results"
NO_FAQ = "no faq entries"


def format_product(p: Product) -> str:
    """One line: `name · price · sizes · date · link · [eskirgan]` (last tag only when stale)."""
    price = "narx: so'rang" if p.price is None else str(p.price)
    sizes = ",".join(p.sizes) if p.sizes else "-"
    parts = [p.name or "", price, sizes, p.date or "", p.link or ""]
    line = " · ".join(parts)
    if p.stale:
        line += " · [eskirgan]"
    return line.replace("\n", " ")


def _format_products(products: list[Product]) -> str:
    return "\n".join(format_product(p) for p in products) or NO_RESULTS


def _format_faq(entry: FaqEntry) -> str:
    return f"{entry.question} — {entry.answer}".replace("\n", " ")


@beta_tool
def find_products_tool(category: str | None = None, min_price: int | None = None,
                       max_price: int | None = None, size: str | None = None,
                       color: str | None = None, keywords: list[str] | None = None) -> str:
    """Filter the shop catalog. All given filters are ANDed; newest posts first.
    Returns one product per line: name · price · sizes · date · link (· [eskirgan] if the post is old),
    or "no results".

    Args:
        category: One of: kiyim (clothes), poyabzal (shoes), aksessuar (accessories), boshqa (other).
        min_price: Minimum price in so'm (UZS).
        max_price: Maximum price in so'm (UZS).
        size: Exact size string as printed on the post, e.g. "42" or "XL".
        color: Color name in Uzbek, e.g. "qora", "oq", "kok".
        keywords: Product keywords in any language or script (uz Latin, uz Cyrillic, ru, en),
            e.g. ["krossovka"], ["кроссовки"], ["kurtka", "jacket"].
    """
    products = search.find_products(category=category, min_price=min_price, max_price=max_price,
                                    size=size, color=color, keywords=keywords,
                                    limit=config.MAX_RESULTS)
    return _format_products(products)


@beta_tool
def semantic_search_tool(text: str, max_price: int | None = None) -> str:
    """Meaning-based search over the catalog for descriptive questions when find_products_tool
    returned "no results" (e.g. "something warm for winter", "подарок для мамы").
    Returns one product per line (same format as find_products_tool) or "no results".

    Args:
        text: The customer's description of what they want, in any language or script.
        max_price: Maximum price in so'm (UZS).
    """
    products = search.semantic_search(text, max_price=max_price, limit=config.MAX_RESULTS)
    return _format_products(products)


@beta_tool
def latest_posts_tool(n: int = 5) -> str:
    """The newest posts in the shop channel ("what's new?", "yangi tovarlar bormi?").
    Returns one product per line (same format as find_products_tool) or "no results".

    Args:
        n: How many latest products to return (1-5).
    """
    n = max(1, min(int(n), config.MAX_RESULTS))
    return _format_products(search.latest_posts(n))


@beta_tool
def search_faq_tool(text: str) -> str:
    """Search the owner's earlier answers about delivery, payment, address, working hours,
    returns and other shop questions. Returns one entry per line as "question — answer",
    or "no faq entries".

    Args:
        text: The customer's question, in any language or script.
    """
    entries = search.search_faq(text, limit=config.MAX_RESULTS)
    return "\n".join(_format_faq(e) for e in entries) or NO_FAQ


@beta_tool
def ask_owner(question: str, post_ids: list[int]) -> str:
    """Forward the customer's question to the shop owner. Use when the customer asks about stock or
    availability, when no relevant product was found, or when the question is about orders, delivery,
    payment or anything not in the catalog. Returns "forwarded"; then tell the customer the owner
    will reply soon.

    Args:
        question: The customer's question, quoted as they wrote it.
        post_ids: Numeric ids of the products the question is about (from the post links), or [].
    """
    from shop_assistant import bot   # lazy: bot imports telethon and the agent
    bot.escalate_sync(question, list(post_ids or []))
    return "forwarded"


TOOLS: list = [find_products_tool, semantic_search_tool, latest_posts_tool, search_faq_tool, ask_owner]
