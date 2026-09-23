"""Spike for ticket #23: does the Gemini free-tier model do our tool calls? Not shipped in the bot.

Needs GEMINI_API_KEY in .env (use a separate AI Studio key/project for development: the free quota
is per project per day). Every request here counts against that quota.

    uv run python scripts/spike_gemini.py agent            # (a) 10 runs, one Uzbek question, 5 agent tools
    uv run python scripts/spike_gemini.py extract 5 10 20  # (b) forced record_products on N real captions

(a) expects a find_products_tool call on >= 9/10 runs and < 3 s per call.
(b) expects a valid record for every post of the batch; pick the largest 100%-valid size for
    config.EXTRACT_BATCH. Captions come from data/posts.jsonl (run fetch first).
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types  # noqa: E402

from shop_assistant import config, extract, llm  # noqa: E402
from shop_assistant.agent import SYSTEM_PROMPT  # noqa: E402
from shop_assistant.models import Post  # noqa: E402
from shop_assistant.tools import TOOLS  # noqa: E402

QUESTION = "Krossovka bormi? 42 razmer, 400 minggacha"
REQUIRED = ("id", "name", "category", "price", "subscriber_price", "sizes", "colors", "keywords", "season")


def spike_agent(runs: int = 10) -> None:
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=llm.tool_declarations(TOOLS))],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        http_options=types.HttpOptions(timeout=llm.AGENT_TIMEOUT_S * 1000),
    )
    hits, times = 0, []
    for i in range(runs):
        t0 = time.perf_counter()
        resp = llm.client().models.generate_content(model=config.GEMINI_MODEL, contents=QUESTION, config=cfg)
        dt = time.perf_counter() - t0
        times.append(dt)
        parts = (resp.candidates[0].content.parts if resp.candidates and resp.candidates[0].content else None) or []
        calls = [p.function_call for p in parts if p.function_call]
        desc = ", ".join(f"{c.name}({json.dumps(dict(c.args or {}), ensure_ascii=False)})" for c in calls) \
            or f"TEXT: {(resp.text or '')[:80]!r}"
        hits += any(c.name == "find_products_tool" for c in calls)
        print(f"run {i + 1:2d}  {dt:5.2f} s  {desc}")
    print(f"\nmodel {config.GEMINI_MODEL}: find_products_tool on {hits}/{runs} runs; "
          f"median {statistics.median(times):.2f} s, max {max(times):.2f} s")


def _load_posts() -> list[Post]:
    if not config.POSTS_PATH.exists():
        sys.exit(f"{config.POSTS_PATH} not found — run `uv run python -m shop_assistant.fetch` first")
    posts = []
    for line in config.POSTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("caption", "").strip():
                posts.append(Post(id=int(d["id"]), date=d["date"], link=d["link"], caption=d["caption"],
                                  has_media=d.get("has_media", True)))
    return posts


def spike_extract(sizes: list[int]) -> None:
    posts = _load_posts()
    for n in sizes:
        chunk = posts[:n]
        bodies = {p.id: extract.strip_footer(p.caption) for p in chunk}
        t0 = time.perf_counter()
        try:
            items = extract._call_llm(chunk, bodies)
        except extract._LLMParseError as e:
            print(f"batch {n:2d}: NO VALID CALL ({e})")
            continue
        dt = time.perf_counter() - t0
        valid = [p.id for p in chunk if isinstance(items.get(p.id), dict) and all(k in items[p.id] for k in REQUIRED)]
        missing = [p.id for p in chunk if p.id not in valid]
        print(f"batch {n:2d}: {len(valid)}/{len(chunk)} valid records in {dt:.1f} s"
              + (f"; missing/invalid ids {missing}" if missing else ""))
        for p in chunk[:3]:
            print("   ", json.dumps(items.get(p.id), ensure_ascii=False)[:160])


def main() -> None:
    args = sys.argv[1:] or ["agent"]
    if args[0] == "agent":
        spike_agent()
    elif args[0] == "extract":
        spike_extract([int(a) for a in args[1:]] or [5])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
