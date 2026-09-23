"""Fake google-genai client for tests (ticket #23). No network.

The fake replaces `shop_assistant.llm.client` (see tests/test_llm.py docstring for the seam).
Its `models.generate_content(model=..., contents=..., config=...)` records every call and returns
real `google.genai.types.GenerateContentResponse` objects (or raises real `google.genai.errors`).
"""
import json

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


class _Models:
    def __init__(self, script):
        self._script = list(script)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if callable(step):
            step = step(contents)
        if isinstance(step, BaseException):
            raise step
        return step


class FakeClient:
    """`script`: list of responses / exceptions / callables(contents) -> response; the last entry repeats."""

    def __init__(self, script):
        self.models = _Models(script)

    @property
    def calls(self) -> list[dict]:
        return self.models.calls
