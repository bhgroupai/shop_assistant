"""Run the agent offline over eval/questions.jsonl and print AC-2..AC-4. Ticket #12.

Each line of questions.jsonl: {"q": "...", "expected": {"posts": [ids]} | {"escalate": true}}

    python -m eval.run_eval                 # real agent (Ollama), ask_owner stubbed
    python -m eval.run_eval --dry [FILE]    # fake agent + fake tools: proves the plumbing only
"""
import datetime
import json
import re
import statistics
import time
from pathlib import Path

_LINK_RE = re.compile(r"t\.me/[A-Za-z0-9_]+/(\d+)")
_PRICE_RE = re.compile(r"\d{1,3}(?:[ .]\d{3})+|\d{4,}")           # 350 000 · 350.000 · 350000
_SIZE_RE = re.compile(r"(?<![\w.])\d{1,2}(?![\w.])")                # standalone 1–2 digits; "2XL" is skipped
_NOISE_RE = re.compile(r"https?://\S+|\bt\.me/\S+|\d{4}-\d{2}-\d{2}")   # links and ISO dates are not prices/sizes


def load_questions(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _candidate_numbers(answer: str) -> set[int]:
    answer = _NOISE_RE.sub(" ", answer)
    prices = {int(re.sub(r"[ .]", "", m)) for m in _PRICE_RE.findall(answer)}
    rest = _PRICE_RE.sub(" ", answer)
    sizes = {int(m) for m in _SIZE_RE.findall(rest)}
    return {p for p in prices if p >= 1000} | sizes


def contains_invented_numbers(answer: str, products: list, question: str = "") -> bool:
    """True when the reply states a price or size not present in the referenced records (AC-3, FR-15).
    Numbers the customer wrote in `question` are allowed (echoing "iPhone 15" is not inventing)."""
    allowed: set[int] = set(_candidate_numbers(question)) if question else set()
    for p in products:
        allowed.update(v for v in (p.price, p.subscriber_price) if v is not None)
        allowed.update(int(s) for s in p.sizes if s.isdigit())
    return bool(_candidate_numbers(answer) - allowed)


def _is_correct(r: dict) -> bool:
    expected = r.get("expected") or {}
    if expected.get("escalate"):
        return bool(r["escalated"])
    return bool(set(expected.get("posts", [])) & set(r["posts"])) and not r["escalated"]


def score(results: list[dict]) -> dict:
    """results: [{"expected": ..., "posts": [ids], "escalated": bool, "invented": bool, "semantic_only": bool}]
    → {"ac2": n_correct, "ac3": n_invented, "ac4": n_semantic_only, "total": n}."""
    correct = [_is_correct(r) for r in results]
    return {
        "ac2": sum(correct),
        "ac3": sum(1 for r in results if r["invented"]),
        "ac4": sum(1 for r, ok in zip(results, correct) if ok and r["semantic_only"]),
        "total": len(results),
    }


# --- tool patching ---------------------------------------------------------

def _ids_in(text) -> set[int]:
    return {int(m) for m in _LINK_RE.findall(str(text))}


class _Capture:
    """Per-question record of what the tools returned and whether ask_owner was called."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.outputs: dict[str, list[str]] = {}
        self.ask_owner_calls: list[dict] = []

    def ids(self, tool: str) -> set[int]:
        return set().union(*(_ids_in(o) for o in self.outputs.get(tool, [])))

    def filter_empty(self) -> bool:
        return not self.ids("find_products_tool")


def _patch_tool(tool, name: str, wrapper):
    """Replace the callable behind a @beta_tool object (or a plain function) so the agent's
    tool list — which holds the same object — sees the wrapper. BetaFunctionTool.call() goes
    through `_func_with_validate`, hence both attributes."""
    if hasattr(tool, "_func_with_validate"):
        tool.func = wrapper
        tool._func_with_validate = wrapper
        return tool
    return wrapper


def _install_capture(tools_mod, capture: _Capture) -> None:
    def make_recorder(name, original):
        def recorder(**kw):
            out = original(**kw)
            capture.outputs.setdefault(name, []).append(str(out))
            return out
        return recorder

    def fake_ask_owner(**kw):
        capture.ask_owner_calls.append(kw)
        return "forwarded"

    for name in ("find_products_tool", "semantic_search_tool", "latest_posts_tool"):
        tool = getattr(tools_mod, name)
        original = getattr(tool, "_func_with_validate", tool)
        setattr(tools_mod, name, _patch_tool(tool, name, make_recorder(name, original)))
    setattr(tools_mod, "ask_owner", _patch_tool(tools_mod.ask_owner, "ask_owner", fake_ask_owner))


# --- dry run: fake agent + fake tools so the plumbing can be exercised without Ollama/Telegram ---

def _dry_setup():
    import types
    from anthropic import beta_tool
    from shop_assistant import config
    from shop_assistant.models import Product

    catalog = [
        Product(id=1300, date="2026-09-12", link=f"https://t.me/{config.CHANNEL}/1300", name="Krossovka Nike Air",
                category="poyabzal", price=350000, subscriber_price=320000, sizes=("40", "41", "42", "43"),
                keywords=("krossovka", "sneakers", "nike"), body="Krossovka Nike Air"),
        Product(id=1234, date="2026-09-10", link=f"https://t.me/{config.CHANNEL}/1234", name="Dvoyka",
                category="kiyim", price=980000, subscriber_price=None, sizes=("M", "L", "XL", "2XL"),
                keywords=("dvoyka", "kostyum"), body="Yangi model Dvoyka"),
    ]

    def fmt(p: Product) -> str:
        return f"{p.name} · {p.price} · {' '.join(p.sizes)} · {p.date} · {p.link}"

    @beta_tool
    def find_products_tool(keywords: list[str] | None = None) -> str:
        """Keyword filter over the dry catalog."""
        hits = [p for p in catalog if any(k.lower() in p.keywords for k in keywords or [])]
        return "\n".join(fmt(p) for p in hits) or "no results"

    @beta_tool
    def semantic_search_tool(text: str, max_price: int | None = None) -> str:
        """Pretend semantic search: 'sport' → Dvoyka."""
        return fmt(catalog[1]) if "sport" in text.lower() else "no results"

    @beta_tool
    def ask_owner(question: str, post_ids: list[int]) -> str:
        """Real escalation — must never run in eval."""
        raise RuntimeError("real ask_owner called: patching failed")

    tools_mod = types.SimpleNamespace(find_products_tool=find_products_tool,
                                      semantic_search_tool=semantic_search_tool, ask_owner=ask_owner)
    tools_mod.TOOLS = [find_products_tool, semantic_search_tool, ask_owner]

    class History:
        def __init__(self) -> None:
            self.msgs: list = []

    def run_agent(chat_id: int, text: str, history=None) -> str:
        # Same order as the system prompt (SDD §3.7): filters → semantic → ask_owner. Calls go through .call()
        # on the objects in TOOLS exactly like the SDK tool_runner does.
        find, sem, ask = tools_mod.TOOLS
        out = find.call({"keywords": text.split()})
        if out == "no results":
            out = sem.call({"text": text})
        if out == "no results":
            ask.call({"question": text, "post_ids": []})
            return "Egasi tez orada javob beradi"
        if "invent" in text:
            out += "\nrazmer 45 ham bor, 300 000 so'm"   # deliberately invented (AC-3 must catch it)
        return out

    return run_agent, History, tools_mod, catalog


# --- runner ------------------------------------------------------------------

def main(dry: bool = False, questions_path=None) -> None:
    from shop_assistant import config

    if dry:
        run_agent, History, tools_mod, catalog = _dry_setup()
    else:
        from shop_assistant import agent, search, tools
        run_agent, History, tools_mod, catalog = agent.run_agent, agent.History, tools, search.PRODUCTS

    capture = _Capture()
    _install_capture(tools_mod, capture)

    by_id = {p.id: p for p in catalog}
    questions = load_questions(questions_path or config.ROOT / "eval" / "questions.jsonl")
    results: list[dict] = []
    for i, item in enumerate(questions):
        q, expected = item["q"], item.get("expected", {})
        capture.reset()
        t0 = time.perf_counter()
        answer = run_agent(i, q, History())
        ms = int((time.perf_counter() - t0) * 1000)

        posts = sorted(capture.ids("find_products_tool") | capture.ids("semantic_search_tool") | capture.ids("latest_posts_tool"))
        escalated = bool(capture.ask_owner_calls)
        expected_ids = set(expected.get("posts", []))
        r = {
            "q": q, "expected": expected, "posts": posts, "escalated": escalated,
            "invented": contains_invented_numbers(answer, [by_id[i] for i in posts if i in by_id], q),
            "semantic_only": capture.filter_empty() and bool(expected_ids & capture.ids("semantic_search_tool")),
            "ms": ms, "answer": answer,
        }
        results.append(r)
        mark = "✓" if _is_correct(r) else "✗"
        print(f"{mark} | {ms:5d} ms | {q[:40]:<40} | {answer[:60]!r}")

    s = score(results)
    times = [r["ms"] for r in results]
    median = statistics.median(times) if times else 0
    p95 = statistics.quantiles(times, n=20)[-1] if len(times) >= 2 else (times[0] if times else 0)
    print(f"\nAC-2 correct {s['ac2']}/{s['total']}  AC-3 invented {s['ac3']}  AC-4 semantic-only {s['ac4']}")
    print(f"median {median:.0f} ms  p95 {p95:.0f} ms")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.DATA_DIR / f"eval_{datetime.date.today().isoformat()}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--dry"]
    main(dry="--dry" in sys.argv, questions_path=Path(args[0]) if args else None)
