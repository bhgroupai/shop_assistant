"""Spike for ticket #23.5: is a Gemini embedding model good enough across scripts? Not shipped in the bot.

Repeats spike #3 (cross-script cosine similarity, raw and after textnorm.normalise) with the Gemini
embedding API. Gate: every spelling variant of krossovka >= 0.80 after normalisation (the old local
model reached 0.87). Needs GEMINI_API_KEY in .env; every model/dimension costs one embed request
per task type.

    uv run python scripts/spike_gemini_embed.py                                  # config.EMBED_MODEL / EMBED_DIM
    uv run python scripts/spike_gemini_embed.py gemini-embedding-001:768 gemini-embedding-2:768
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from google.genai import types  # noqa: E402

from shop_assistant import config, llm  # noqa: E402
from shop_assistant.textnorm import normalise  # noqa: E402

GATE = 0.80
SPELLING = [("krossovka", "кроссовка"), ("krossovka", "krosovka"), ("кроссовка", "krosovka"),
            ("krossovka", "кроссовки")]
RU_UZ = [("свитер", "sviter"), ("для холодной погоды", "qishki"), ("зимняя куртка", "qishki kurtka"),
         ("обувь", "poyabzal")]
UNRELATED = [("krossovka", "telefon"), ("свитер", "poyabzal")]   # floor: what "not the same thing" scores


def _embed(model: str, dim: int, texts: list[str], task: str) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    resp = llm.client().models.embed_content(
        model=model, contents=texts,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=dim))
    ms = (time.perf_counter() - t0) * 1000
    if len(resp.embeddings) != len(texts):   # gemini-embedding-2 folds a list of texts into ONE embedding
        raise SystemExit(f"{model}: {len(resp.embeddings)} embeddings for {len(texts)} texts — cannot batch")
    return np.asarray([e.values for e in resp.embeddings], dtype=np.float32), ms


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run(model: str, dim: int) -> bool:
    pairs = SPELLING + RU_UZ + UNRELATED
    words = sorted({w for p in pairs for w in p})
    print(f"\n== {model}, {dim}-d ==")
    ok = True
    for task in ("RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY"):
        raw, ms_raw = _embed(model, dim, words, task)
        norm, ms_norm = _embed(model, dim, [normalise(w) for w in words], task)
        v_raw, v_norm = dict(zip(words, raw)), dict(zip(words, norm))
        print(f"-- task_type {task}; {len(words)} texts per request, {ms_raw:.0f} / {ms_norm:.0f} ms; "
              f"row norms {np.linalg.norm(raw, axis=1).min():.3f}..{np.linalg.norm(raw, axis=1).max():.3f}")
        print(f"{'pair':<44}{'raw':>8}{'normalised':>12}")
        for group, gated in ((SPELLING, True), (RU_UZ, False), (UNRELATED, False)):
            for a, b in group:
                r, n = _cos(v_raw[a], v_raw[b]), _cos(v_norm[a], v_norm[b])
                flag = ""
                if gated and task == "RETRIEVAL_DOCUMENT" and n < GATE:
                    ok, flag = False, "  < gate"
                print(f"{a + ' / ' + b:<44}{r:8.3f}{n:12.3f}{flag}")
    # the real use: a query embedded as RETRIEVAL_QUERY against documents embedded as RETRIEVAL_DOCUMENT
    q, _ = _embed(model, dim, [normalise(a) for a, _ in RU_UZ], "RETRIEVAL_QUERY")
    d, _ = _embed(model, dim, [normalise(b) for _, b in RU_UZ], "RETRIEVAL_DOCUMENT")
    print("-- query (RETRIEVAL_QUERY) vs document (RETRIEVAL_DOCUMENT), normalised")
    for (a, b), qv, dv in zip(RU_UZ, q, d):
        print(f"{a + ' -> ' + b:<44}{'':8}{_cos(qv, dv):12.3f}")
    print(f"gate (spelling variants >= {GATE} normalised, RETRIEVAL_DOCUMENT): {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str]) -> int:
    specs = argv or [f"{config.EMBED_MODEL}:{config.EMBED_DIM}"]
    results = []
    for spec in specs:
        model, _, dim = spec.partition(":")
        results.append(run(model, int(dim or config.EMBED_DIM)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
