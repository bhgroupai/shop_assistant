"""Tool wrappers with compact text output. SDD §3.6, FR-11, NFR-2. Tickets #9, #23.5."""
import dataclasses
import inspect
import re
import types
import typing
from typing import Any, Callable

from shop_assistant import config, search
from shop_assistant.models import FaqEntry, Product


@dataclasses.dataclass
class Tool:
    """A plain tool definition: what the model sees (`name`, `description`, JSON-schema `input_schema`)
    and `call(args)`, which runs `func(**args)`. `llm.tool_declarations` converts it for Gemini."""
    name: str
    description: str
    input_schema: dict
    func: Callable[..., Any]

    def call(self, args: dict) -> str:
        return str(self.func(**(args or {})))


_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_schema(hint) -> dict:
    """Python type hint → JSON schema (str, int, float, bool, list[X], X | None)."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in args if a is not type(None)]
        if len(inner) != 1:
            raise TypeError(f"unsupported tool parameter type {hint}")
        schema = _json_schema(inner[0])
        return {**schema, "type": [schema["type"], "null"]} if len(inner) < len(args) else schema
    if origin is list:
        return {"type": "array", "items": _json_schema(args[0]) if args else {}}
    if hint in _JSON_TYPES:
        return {"type": _JSON_TYPES[hint]}
    raise TypeError(f"unsupported tool parameter type {hint}")


def _split_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Google-style docstring → (summary, {arg: description}). Continuation lines of an argument are
    kept on their own lines (dedented)."""
    doc = inspect.cleandoc(doc or "")
    summary, _, args_block = doc.partition("\nArgs:\n")
    lines = [ln for ln in args_block.splitlines() if ln.strip()]
    indent = min((len(ln) - len(ln.lstrip()) for ln in lines), default=0)   # the argument-name column
    args: dict[str, str] = {}
    current = None
    for line in lines:
        m = re.match(r"(\w+):\s*(.*)$", line.strip())
        if m and len(line) - len(line.lstrip()) == indent:
            current = m.group(1)
            args[current] = m.group(2)
        elif current is not None:
            args[current] += "\n" + line.strip()
    return summary.strip(), args


def tool(func: Callable[..., str]) -> Tool:
    """Decorator: build a Tool from the function's signature (types, defaults → required) and docstring."""
    summary, arg_docs = _split_docstring(func.__doc__)
    hints = typing.get_type_hints(func)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in inspect.signature(func).parameters.items():
        prop = _json_schema(hints[name])
        if name in arg_docs:
            prop["description"] = arg_docs[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return Tool(name=func.__name__, description=summary, input_schema=schema, func=func)


NO_RESULTS = "no results"
NO_FAQ = "no faq entries"
ASK_PRICE = "narxi: so'rab beraman"
OFFER_PRICE = "Narxini bilmoqchi bo'lsangiz raqamini yozing"


def format_product(p: Product) -> str:
    """One line: `name · price · sizes · date · link · [eskirgan]` (last tag only when stale)."""
    price = "narx: so'rang" if p.price is None else str(p.price)
    sizes = ",".join(p.sizes) if p.sizes else "-"
    parts = [p.name or "", price, sizes, p.date or "", p.link or ""]
    line = " · ".join(parts)
    if p.stale:
        line += " · [eskirgan]"
    return line.replace("\n", " ")


def format_products(products: list[Product]) -> str:
    """Numbered lines `1. <format_product line>`; price None renders `narxi: so'rab beraman`;
    empty list → NO_RESULTS. Ends with the offer line
    `Narxini bilmoqchi bo'lsangiz raqamini yozing` only when at least one item has no price."""
    if not products:
        return NO_RESULTS
    lines = []
    for i, p in enumerate(products, start=1):
        line = format_product(p)
        if p.price is None:
            line = line.replace("narx: so'rang", ASK_PRICE, 1)
        lines.append(f"{i}. {line}")
    if any(p.price is None for p in products):
        lines.append(OFFER_PRICE)
    return "\n".join(lines)


def _format_faq(entry: FaqEntry) -> str:
    return f"{entry.question} — {entry.answer}".replace("\n", " ")


@tool
def find_products_tool(category: str | None = None, min_price: int | None = None,
                       max_price: int | None = None, size: str | None = None,
                       color: str | None = None, keywords: list[str] | None = None) -> str:
    """Filter the shop catalog. All given filters are ANDed; newest posts first.
    Returns one numbered product per line: N. name · price · sizes · date · link (· [eskirgan] if the
    post is old), or "no results". Items without a price show "narxi: so'rab beraman"; then a final
    offer line tells the customer to write the item number to ask for its price.

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
    return format_products(products)


@tool
def semantic_search_tool(text: str, max_price: int | None = None) -> str:
    """Meaning-based search over the catalog for descriptive questions when find_products_tool
    returned "no results" (e.g. "something warm for winter", "подарок для мамы").
    Returns one product per line (same format as find_products_tool) or "no results".

    Args:
        text: The customer's description of what they want, in any language or script.
        max_price: Maximum price in so'm (UZS).
    """
    products = search.semantic_search(text, max_price=max_price, limit=config.MAX_RESULTS)
    return format_products(products)


@tool
def latest_posts_tool(n: int = 5) -> str:
    """The newest posts in the shop channel ("what's new?", "yangi tovarlar bormi?").
    Returns one product per line (same format as find_products_tool) or "no results".

    Args:
        n: How many latest products to return (1-5).
    """
    n = max(1, min(int(n), config.MAX_RESULTS))
    return format_products(search.latest_posts(n))


@tool
def search_faq_tool(text: str) -> str:
    """Search the owner's earlier answers about delivery, payment, address, working hours,
    returns and other shop questions. Returns one entry per line as "question — answer",
    or "no faq entries".

    Args:
        text: The customer's question, in any language or script.
    """
    entries = search.search_faq(text, limit=config.MAX_RESULTS)
    return "\n".join(_format_faq(e) for e in entries) or NO_FAQ


@tool
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
