"""Shared Gemini client + tool-schema helper (google-genai SDK). SDD §3.7, §3.9. Ticket #23.

agent.py and extract.py reach Gemini only through `llm.client().models.generate_content(...)`,
looked up at call time, so tests can swap `llm.client` for a fake.
"""
import copy

from google import genai  # noqa: F401  (tests patch `llm.genai.Client`)

from shop_assistant import config

AGENT_TIMEOUT_S = 15       # per customer-agent request
EXTRACT_TIMEOUT_S = 90     # per extraction request (a batch of captions takes longer)

_client = None             # cached genai.Client, created on first use by client()

# JSON-schema keywords that tool definitions may carry but Gemini's Schema rejects or does not need.
_DROP_KEYS = frozenset({"additionalProperties", "title", "default", "$schema", "$defs", "$ref"})


def client():
    """Return the shared genai.Client, created lazily on first call from GEMINI_API_KEY
    (via config.secret) and cached in `_client`. Missing key -> RuntimeError naming GEMINI_API_KEY."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.secret("GEMINI_API_KEY"))
    return _client


def _field(tool, name: str):
    return tool.get(name) if isinstance(tool, dict) else getattr(tool, name, None)


def _schema(node):
    """JSON schema (tool input_schema) -> Gemini Schema dict. Works on a copy; never mutates `node`."""
    if isinstance(node, list):
        return [_schema(v) for v in node]
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    out: dict = {}
    nullable = False
    for key, value in node.items():
        if key in _DROP_KEYS:
            continue
        if key in ("anyOf", "oneOf"):
            # Optional[X] from pydantic: anyOf [X, {"type": "null"}] -> X with nullable.
            variants = [v for v in value if not (isinstance(v, dict) and v.get("type") == "null")]
            nullable = nullable or len(variants) < len(value)
            if len(variants) != 1:
                raise ValueError(f"cannot convert {key} with {len(variants)} non-null variants for Gemini")
            for k, v in _schema(variants[0]).items():
                out.setdefault(k, v)
            continue
        if key == "allOf":
            if len(value) != 1:
                raise ValueError("cannot convert allOf with several schemas for Gemini")
            for k, v in _schema(value[0]).items():
                out.setdefault(k, v)
            continue
        if key == "type" and isinstance(value, list):
            types_ = [t for t in value if t != "null"]
            nullable = nullable or len(types_) < len(value)
            if len(types_) != 1:
                raise ValueError(f"cannot convert type {value} for Gemini")
            out["type"] = types_[0]
            continue
        if key == "properties":
            out[key] = {name: _schema(prop) for name, prop in value.items()}
            continue
        out[key] = _schema(value)
    if nullable:
        out["nullable"] = True
    return out


def tool_declarations(tools) -> list[dict]:
    """Tool defs (dicts or objects with name / description / input_schema, e.g.
    tools.TOOLS or extract._TOOL) -> Gemini function declarations as plain dicts
    {"name", "description", "parameters"}. Strips JSON-schema keywords Gemini rejects; never mutates input."""
    decls: list[dict] = []
    for tool in tools:
        decls.append({
            "name": _field(tool, "name"),
            "description": _field(tool, "description") or "",
            "parameters": _schema(_field(tool, "input_schema") or {"type": "object", "properties": {}}),
        })
    return decls
