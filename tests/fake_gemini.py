"""Fake google-genai client for tests (tickets #23, #23.5). No network.

The fake replaces `shop_assistant.llm.client` (see tests/test_llm.py docstring for the seam).
Its `models.generate_content(model=..., contents=..., config=...)` records every call and returns
real `google.genai.types.GenerateContentResponse` objects (or raises real `google.genai.errors`).
Its `models.embed_content(model=..., contents=..., config=...)` (ticket #23.5) records every call in
`embed_calls` and returns a real `types.EmbedContentResponse` with one deterministic, NOT unit-length
vector per text (`fake_vector`), at the requested `output_dimensionality`.
"""
import json
import zlib

import numpy as np

from google.genai import errors, types


def text_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason="STOP")])


def call_response(name: str, args: dict) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]),
        finish_reason="STOP")])


def empty_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[])


def quota_error() -> errors.ClientError:
    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                              "message": "Resource has been exhausted (e.g. check quota)."}})


def server_error() -> errors.ServerError:
    return errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE",
                                              "message": "The model is overloaded. Please try again later."}})


def invalid_key_error() -> errors.ClientError:
    return errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                              "message": "API key not valid. Please pass a valid API key."}})


def timeout_error():
    import httpx
    return httpx.ReadTimeout("The read operation timed out")


def plain(obj):
    """google-genai pydantic objects / dicts / lists -> JSON-able plain data (drops None)."""
    if hasattr(obj, "model_dump"):
        return plain(obj.model_dump(mode="json", exclude_none=True))
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    return obj


def dump(obj) -> str:
    return json.dumps(plain(obj), ensure_ascii=False)


def fake_vector(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding of `text`: same text -> same vector; norm is about 3*sqrt(dim), not 1."""
    rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
    return [float(x) for x in rng.normal(0.0, 3.0, size=dim)]


def embed_response(texts: list[str], dim: int) -> types.EmbedContentResponse:
    return types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=fake_vector(t, dim)) for t in texts])


def embed_config(config) -> dict:
    """EmbedContentConfig or dict -> plain dict, e.g. {"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 768}."""
    return plain(config) if config is not None else {}


class _Models:
    def __init__(self, script, embed_script=None):
        self._script = list(script)
        self.calls: list[dict] = []
        self._embed_script = list(embed_script or [])
        self.embed_calls: list[dict] = []

    def embed_content(self, *, model, contents, config=None):
        """Records the call; the next `embed_script` entry (exception, response, or callable(contents, config))
        decides the answer; with no script (or when the script is used up) answer with fake_vector per text."""
        self.embed_calls.append({"model": model, "contents": contents, "config": config})
        step = self._embed_script.pop(0) if self._embed_script else None
        if callable(step) and not isinstance(step, BaseException):
            step = step(contents, config)
        if isinstance(step, BaseException):
            raise step
        if step is not None:
            return step
        texts = [contents] if isinstance(contents, str) else list(contents)
        dim = embed_config(config).get("output_dimensionality") or 3072   # the API default when none is asked
        return embed_response(texts, dim)

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if callable(step):
            step = step(contents)
        if isinstance(step, BaseException):
            raise step
        return step


class FakeClient:
    """`script`: list of responses / exceptions / callables(contents) -> response; the last entry repeats.
    `embed_script`: embed_content answers in order (exceptions / responses / callables(contents, config)),
    each used once; after that every text gets its fake_vector."""

    def __init__(self, script=(), embed_script=None):
        self.models = _Models(list(script) or [text_response("")], embed_script)

    @property
    def calls(self) -> list[dict]:
        return self.models.calls

    @property
    def embed_calls(self) -> list[dict]:
        return self.models.embed_calls
