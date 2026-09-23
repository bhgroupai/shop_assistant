"""Ticket #23 — shared Gemini client + tool-schema helper (shop_assistant/llm.py).

THE SEAM (read this before implementing — tests for llm, agent and extract all rely on it):

* `llm.client()` returns the one google-genai `genai.Client`, created lazily on first call with
  `genai.Client(api_key=<GEMINI_API_KEY>)` (key read via `config.secret("GEMINI_API_KEY")`) and cached
  in the module attribute `llm._client`. Importing llm never creates a client and never needs the key.
* agent.py and extract.py must reach the SDK ONLY through `llm.client().models.generate_content(
  model=config.GEMINI_MODEL, contents=..., config=types.GenerateContentConfig(...))`, looking
  `llm.client` up at call time (`from shop_assistant import llm` ... `llm.client()`), never
  `from shop_assistant.llm import client`. Tests monkeypatch `llm.client` to return
  tests/fake_gemini.FakeClient, which returns real `google.genai.types.GenerateContentResponse`
  objects (function_call / text parts) or raises real `google.genai.errors.ClientError` (429, 400),
  `errors.ServerError` (503) or `httpx.ReadTimeout`.
* The per-request timeout goes in the request config: `http_options=types.HttpOptions(timeout=ms)`
  with ms = `llm.AGENT_TIMEOUT_S * 1000` (agent) or `llm.EXTRACT_TIMEOUT_S * 1000` (extraction).
* Tool declarations come from `llm.tool_declarations(tools)` -> list of plain dicts
  {"name", "description", "parameters"} passed as `tools=[types.Tool(function_declarations=...)]`
  (or the equivalent dict). TOOLS / extract._TOOL stay the single source of truth and are never mutated.
* Extraction backoff sleeps with `time.sleep` (tests patch `time.sleep`, so no real waiting).
* Errors are logged with the `logging` module (tests count ERROR records with caplog).
"""
import copy
import importlib

import pytest
from google.genai import types

from shop_assistant import config, llm
from shop_assistant.extract import _TOOL
from shop_assistant.tools import TOOLS

GEMINI_REJECTS = {"additionalProperties", "title", "default", "anyOf", "oneOf", "allOf", "$schema", "$defs", "$ref"}


def _walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _anthropic_dicts():
    return [t.to_dict() for t in TOOLS]


# --- tool_declarations -------------------------------------------------------

def test_one_declaration_per_tool_with_names_and_descriptions():
    decls = llm.tool_declarations(TOOLS)
    assert [d["name"] for d in decls] == [t.name for t in TOOLS]
    assert [d["description"] for d in decls] == [t.description for t in TOOLS]


def test_declaration_keeps_properties_and_required():
    decls = {d["name"]: d for d in llm.tool_declarations(TOOLS)}
    for t in TOOLS:
        params = decls[t.name]["parameters"]
        assert params["type"] == "object"
        assert set(params["properties"]) == set(t.input_schema["properties"])
        assert sorted(params.get("required", [])) == sorted(t.input_schema.get("required", []))
    assert decls["ask_owner"]["parameters"]["properties"]["post_ids"]["items"]["type"] == "integer"


def test_declarations_have_no_keywords_gemini_rejects():
    for decl in llm.tool_declarations(TOOLS) + llm.tool_declarations([_TOOL]):
        for node in _walk(decl["parameters"]):
            if isinstance(node, dict):
                assert not (GEMINI_REJECTS & set(node)), f"{decl['name']}: {GEMINI_REJECTS & set(node)}"
                if "type" in node:
                    assert isinstance(node["type"], str), f"{decl['name']}: type must be one string"


def test_optional_anyof_null_becomes_nullable_type():
    decls = {d["name"]: d for d in llm.tool_declarations(TOOLS)}
    category = decls["find_products_tool"]["parameters"]["properties"]["category"]
    assert category["type"] == "string" and category.get("nullable") is True
    assert "kiyim" in category["description"]
    keywords = decls["find_products_tool"]["parameters"]["properties"]["keywords"]
    assert keywords["type"] == "array" and keywords["items"]["type"] == "string"


def test_extraction_tool_dict_is_converted_with_enum_and_nullable_price():
    [decl] = llm.tool_declarations([_TOOL])
    assert decl["name"] == "record_products"
    item = decl["parameters"]["properties"]["products"]["items"]
    assert item["properties"]["category"]["enum"] == config.CATEGORIES
    assert item["properties"]["price"]["type"] == "integer" and item["properties"]["price"]["nullable"] is True
    assert item["properties"]["sizes"]["items"]["type"] == "string"
    assert "id" in item["required"]


def test_declarations_are_accepted_by_google_genai():
    decls = llm.tool_declarations(TOOLS) + llm.tool_declarations([_TOOL])
    tool = types.Tool(function_declarations=[types.FunctionDeclaration.model_validate(d) for d in decls])
    assert [f.name for f in tool.function_declarations] == [t.name for t in TOOLS] + ["record_products"]


def test_tools_are_not_mutated():
    before_tools, before_extract = _anthropic_dicts(), copy.deepcopy(_TOOL)
    llm.tool_declarations(TOOLS)
    llm.tool_declarations([_TOOL])
    assert _anthropic_dicts() == before_tools
    assert _TOOL == before_extract


def test_no_tools_gives_no_declarations():
    assert llm.tool_declarations([]) == []


# --- client ------------------------------------------------------------------

class _RecordingClient:
    made: list[dict] = []

    def __init__(self, **kwargs):
        _RecordingClient.made.append(kwargs)


@pytest.fixture
def recording_genai(monkeypatch):
    _RecordingClient.made = []
    monkeypatch.setattr(llm.genai, "Client", _RecordingClient)
    monkeypatch.setattr(llm, "_client", None)
    return _RecordingClient


def test_import_does_not_create_a_client(recording_genai, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    importlib.reload(llm)                 # must not need the key and must not build a client
    assert recording_genai.made == []
    assert llm._client is None


def test_client_is_created_once_from_the_key(recording_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-shop-assistant")
    first = llm.client()
    second = llm.client()
    assert first is second
    assert len(recording_genai.made) == 1
    assert recording_genai.made[0].get("api_key") == "AIza-test-key-shop-assistant"


def test_missing_key_raises_clear_error(recording_genai, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.client()
    assert recording_genai.made == []


def test_empty_key_raises_clear_error(recording_genai, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.client()


def test_timeouts():
    assert 10 <= llm.AGENT_TIMEOUT_S <= 20
    assert llm.EXTRACT_TIMEOUT_S > llm.AGENT_TIMEOUT_S
