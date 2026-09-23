"""Shared Gemini client + tool-schema helper (google-genai SDK). SDD §3.7, §3.9. Ticket #23.

STUB written with the tests — the implementer fills it in. See tests/test_llm.py for the contract.
"""
from google import genai  # noqa: F401  (tests patch `llm.genai.Client`)

AGENT_TIMEOUT_S = 15       # per customer-agent request
EXTRACT_TIMEOUT_S = 90     # per extraction request (a batch of captions takes longer)

_client = None             # cached genai.Client, created on first use by client()


def client():
    """Return the shared genai.Client, created lazily on first call from GEMINI_API_KEY
    (via config.secret) and cached in `_client`. Missing key -> RuntimeError naming GEMINI_API_KEY."""
    raise NotImplementedError("ticket #23")


def tool_declarations(tools) -> list[dict]:
    """Anthropic-style tool defs (dicts or objects with name / description / input_schema, e.g.
    tools.TOOLS or extract._TOOL) -> Gemini function declarations as plain dicts
    {"name", "description", "parameters"}. Strips JSON-schema keywords Gemini rejects; never mutates input."""
    raise NotImplementedError("ticket #23")
