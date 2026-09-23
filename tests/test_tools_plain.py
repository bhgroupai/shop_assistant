"""Ticket #23.5 — tools.TOOLS without the anthropic package.

CONTRACT: every TOOLS entry is a plain definition (no anthropic `beta_tool`) with attributes
`name` (str), `description` (str), `input_schema` (JSON-schema dict: type object, properties, required)
and `call(args: dict) -> str`, which runs the same function as before. TOOLS stays the single source of
truth for `llm.tool_declarations`. What the model sees must not change: tests/tool_declarations.json is
the Gemini view of the #23 tools (descriptions compared with whitespace collapsed). Updated for #25: the three
listing tools take `offset` and their descriptions mention the range/total line (tests/test_tools_paging.py).
"""
import json
from pathlib import Path

from google.genai import types

from shop_assistant import config, llm, search, tools
from shop_assistant.tools import TOOLS

SNAPSHOT = json.loads((Path(__file__).parent / "tool_declarations.json").read_text(encoding="utf-8"))
NAMES = ["find_products_tool", "semantic_search_tool", "latest_posts_tool", "search_faq_tool", "ask_owner"]


def _squash(text: str) -> str:
    return " ".join(text.split())


def test_same_five_tools_in_order():
    assert [t.name for t in TOOLS] == NAMES


def test_tools_are_not_anthropic_objects():
    for t in TOOLS:
        assert not type(t).__module__.startswith("anthropic"), f"{t.name}: {type(t).__module__}"


def test_entries_have_description_and_object_schema():
    for t in TOOLS:
        assert isinstance(t.description, str) and t.description.strip()
        assert t.input_schema["type"] == "object"
        assert isinstance(t.input_schema["properties"], dict)
        assert callable(t.call)


def test_gemini_view_is_unchanged():
    decls = llm.tool_declarations(TOOLS)
    assert [d["name"] for d in decls] == [s["name"] for s in SNAPSHOT]
    for got, want in zip(decls, SNAPSHOT):
        assert _squash(got["description"]) == want["description"], got["name"]
        params = dict(got["parameters"])
        want_params = dict(want["parameters"])
        assert sorted(params.pop("required", [])) == sorted(want_params.pop("required", [])), got["name"]
        assert params == want_params, got["name"]


def test_declarations_are_accepted_by_google_genai():
    decls = llm.tool_declarations(TOOLS)
    tool = types.Tool(function_declarations=[types.FunctionDeclaration.model_validate(d) for d in decls])
    assert [f.name for f in tool.function_declarations] == NAMES


def _tool(name):
    return next(t for t in TOOLS if t.name == name)


def test_call_runs_find_products_with_the_model_args(monkeypatch, products):
    seen = {}

    def find_products(**kw):
        seen.update(kw)
        return [products[1]]

    monkeypatch.setattr(search, "find_products", find_products)
    out = _tool("find_products_tool").call({"keywords": ["krossovka"], "size": "42"})
    assert out.startswith("1. Krossovka Nike Air · 350000")
    assert seen["keywords"] == ["krossovka"] and seen["size"] == "42"


def test_call_runs_latest_posts_and_clamps_n(monkeypatch, products):
    # #25: the tool may ask for more than a page (to know the total) and slices the page itself
    two = products[:2]
    monkeypatch.setattr(search, "latest_posts", lambda n=5, products=None: two)
    out = _tool("latest_posts_tool").call({"n": 50})
    numbered = [ln for ln in out.splitlines() if ln[:1].isdigit()]
    assert 1 <= len(numbered) <= config.MAX_RESULTS
    assert numbered[0].startswith("1. Dvoyka")


def test_call_runs_semantic_search(monkeypatch, products):
    monkeypatch.setattr(search, "semantic_search", lambda text, max_price=None, limit=5: [products[2]])
    out = _tool("semantic_search_tool").call({"text": "что-нибудь для холодной погоды", "max_price": 1500000})
    assert "Qishki kurtka" in out


def test_call_runs_search_faq_with_no_entries():
    assert _tool("search_faq_tool").call({"text": "Dostavka bormi?"}) == tools.NO_FAQ
