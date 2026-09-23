"""FAQ store: the owner's relayed answers, embedded and searched by meaning. SDD §2.4, §3.5a, FR-22. Ticket #23.7.

Contract: tests/test_faq.py (module docstring). In short:
- data/faq.jsonl (config.FAQ_PATH): one {"ts", "question", "answer", "post_ids"} per owner answer, append-only.
- data/faq_embeddings.npy (config.FAQ_EMBEDDINGS_PATH): row i = index.embed([question of line i], kind="document").
- data/faq_embeddings_meta.json (config.FAQ_META_PATH): {"model", "dim", "n"} — the model guard, as for products.
- add() saves first, then tries to embed every entry that has no row yet; an embedding failure never raises.
- search() embeds the query with kind="query" and returns entries whose cosine >= THRESHOLD.
All paths are read from config at call time. Questions and queries are embedded as normalise(text) (D-3).
"""
import datetime
import json
import logging
import threading

import numpy as np

from shop_assistant import config
from shop_assistant.models import FaqEntry
from shop_assistant.textnorm import normalise

log = logging.getLogger(__name__)

# Min cosine(question, query) for a hit. Measured 2026-09-23 with gemini-embedding-001, 768-d, both sides
# normalise()d, question as RETRIEVAL_DOCUMENT, query as RETRIEVAL_QUERY (so even the verbatim question is only 0.865):
#   paraphrases that should match: "Samarqandga yetkazib berish necha pul?" 0.837, "доставка в Самарканд сколько?"
#   0.840, "Samarqandga dostavka bormi narxi qancha" 0.851, "Самарқандга доставка қанча?" 0.860, "kartaga to'lov
#   bormi?" 0.837, "qanday buyurtma beraman?" 0.835, "Click orqali to'lasam bo'ladimi?" 0.772, "soat nechagacha
#   ishlaysizlar?" 0.774, "во сколько открываетесь?" 0.731 (missed: escalates, the safe side);
#   should not match: unrelated product questions max 0.718 ("dvoyka narxi qancha?" vs the delivery question),
#   "krossovka 42 bormi?" 0.671, "что нового?" 0.622; a paraphrase against another FAQ question max 0.706.
#   Not separable by cosine: "dastavka Toshkentga qancha?" vs the Samarqand question 0.831 — the tool line shows
#   the stored question, and the prompt tells the model to use an answer only when it fits.
THRESHOLD = 0.75

_lock = threading.Lock()   # add() runs in a bot worker thread while an agent thread may search()


class _Mismatch(Exception):
    """The stored matrix was built by another model / dimension (or has no meta): never used or extended."""


def _parse(d: dict) -> FaqEntry:
    return FaqEntry(ts=str(d["ts"]), question=str(d["question"]), answer=str(d["answer"]),
                    post_ids=tuple(int(i) for i in d.get("post_ids") or ()))


def load() -> list[FaqEntry]:
    """faq.jsonl → FaqEntry list in file order (post_ids as a tuple). Missing file → []; bad lines skipped."""
    path = config.FAQ_PATH
    if not path.exists():
        return []
    entries: list[FaqEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                entries.append(_parse(json.loads(line)))
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                log.warning("%s line %d skipped: %s", path, n, e)
    return entries


def _load_matrix() -> np.ndarray:
    """faq_embeddings.npy checked against its meta → float32[n, EMBED_DIM]. No .npy → (0, EMBED_DIM).
    Raises _Mismatch (the message says what is wrong and how to fix it) when it must not be used."""
    npy, meta_path, dim = config.FAQ_EMBEDDINGS_PATH, config.FAQ_META_PATH, config.EMBED_DIM
    if not npy.exists():
        return np.zeros((0, dim), dtype=np.float32)
    fix = f"re-index with {config.EMBED_MODEL} ({dim}-d): python -m shop_assistant.faq"
    try:
        matrix = np.load(npy).astype(np.float32, copy=False)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    except (OSError, ValueError) as e:
        raise _Mismatch(f"cannot read {npy} ({e}); {fix}") from e
    if not isinstance(meta, dict):
        raise _Mismatch(f"{npy} has no model name ({meta_path.name} missing), expected {config.EMBED_MODEL}; {fix}")
    if meta.get("model") != config.EMBED_MODEL:
        raise _Mismatch(f"{npy} was built by {meta.get('model')}, config.EMBED_MODEL is {config.EMBED_MODEL}; {fix}")
    if meta.get("dim") != dim:
        raise _Mismatch(f"{npy} has dimension {meta.get('dim')}, config.EMBED_DIM is {dim}; {fix}")
    if matrix.ndim != 2 or matrix.shape[1] != dim or matrix.shape[0] != meta.get("n"):
        raise _Mismatch(f"{npy} has shape {matrix.shape} but meta says n={meta.get('n')}, dim={dim}; {fix}")
    return matrix


def _write(matrix: np.ndarray) -> None:
    """Rewrite the .npy, then the meta (last: the rows count only once they exist). Each file atomically."""
    from shop_assistant import index   # lazy: index imports the Gemini SDK
    meta = {"model": config.EMBED_MODEL, "dim": config.EMBED_DIM, "n": int(matrix.shape[0])}

    def write_npy(tmp: str) -> None:
        with open(tmp, "wb") as f:
            np.save(f, matrix.astype(np.float32, copy=False))

    def write_meta(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    index._atomic_write(config.FAQ_EMBEDDINGS_PATH, write_npy)
    index._atomic_write(config.FAQ_META_PATH, write_meta)


def _sync(entries: list[FaqEntry], matrix: np.ndarray) -> np.ndarray:
    """Embed entries[len(matrix):] in one kind="document" request (no retries), persist, return the new
    matrix. Raises on embedding errors (nothing written then)."""
    from shop_assistant import index
    pending = entries[matrix.shape[0]:]
    if not pending:
        return matrix
    rows = index.embed([normalise(e.question) for e in pending], kind="document", retry=False)
    matrix = np.vstack([matrix, rows]).astype(np.float32)
    _write(matrix)
    return matrix


def add(question: str, answer: str, post_ids=(), ts: str | None = None) -> FaqEntry:
    """Append one entry to faq.jsonl (ts defaults to now, ISO seconds), then sync(). Returns the entry.
    Embedding errors (quota, network, model mismatch) are logged, never raised: the entry stays saved
    and is embedded by the next successful add()/search()."""
    if ts is None:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
    entry = FaqEntry(ts=ts, question=question, answer=answer, post_ids=tuple(int(i) for i in post_ids or ()))
    line = json.dumps({"ts": entry.ts, "question": entry.question, "answer": entry.answer,
                       "post_ids": list(entry.post_ids)}, ensure_ascii=False)
    with _lock:
        config.FAQ_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.FAQ_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        try:
            _sync(load(), _load_matrix())
        except _Mismatch as e:
            log.error("faq entry saved but not embedded: %s", e)
        except Exception as e:   # 429 / 5xx / timeout / bad key: embedded by the next add() or search()
            log.warning("faq entry saved, embedding postponed (%s: %s)", type(e).__name__, str(e)[:200])
    return entry


def sync() -> int:
    """Embed the entries that have no row yet (one index.embed(..., kind="document") call, retry=False) and
    rewrite faq_embeddings.npy + meta. Returns how many were embedded. Raises on embedding errors;
    does nothing (returns 0) when the stored matrix was built by another model / dimension."""
    with _lock:
        try:
            matrix = _load_matrix()
        except _Mismatch as e:
            log.error("faq sync skipped: %s", e)
            return 0
        return _sync(load(), matrix).shape[0] - matrix.shape[0]


def search(query: str, k: int = 3) -> list[FaqEntry]:
    """Top-k entries with cosine >= THRESHOLD, best first. Embeds pending entries first (failure → go on
    with what is embedded). Query via index.embed([...], kind="query"); failure → [] + one WARNING.
    No embedded entries → [] without a request. Model / dimension mismatch → [] + one ERROR, no request."""
    from shop_assistant import index, search   # lazy: search loads the catalog at import
    if k <= 0:
        return []
    with _lock:
        entries = load()
        try:
            matrix = _load_matrix()
        except _Mismatch as e:
            log.error("faq search off: %s", e)
            return []
        if len(entries) > matrix.shape[0]:
            try:
                matrix = _sync(entries, matrix)
            except Exception as e:
                log.warning("faq: pending entries not embedded, searching the %d embedded ones (%s: %s)",
                            matrix.shape[0], type(e).__name__, str(e)[:200])
    if matrix.shape[0] == 0:
        return []
    try:
        q = index.embed([normalise(query)], kind="query")[0]
    except Exception as e:   # the agent goes on without FAQ hits (then ask_owner)
        log.warning("faq search: query embedding failed, no FAQ results (%s: %s)", type(e).__name__, str(e)[:200])
        return []
    scores = matrix @ q   # rows and query are L2-normalised by index.embed → cosine
    return [entries[i] for i in search.cosine_top_k(q, matrix, k) if scores[i] >= THRESHOLD]


def reindex() -> int:
    """Re-embed every entry with the current config.EMBED_MODEL (retry=True) and rewrite matrix + meta.
    The fix after a model change. Returns the number of entries."""
    from shop_assistant import index
    with _lock:
        entries = load()
        matrix = index.embed([normalise(e.question) for e in entries], kind="document", retry=True)
        _write(matrix)
    return len(entries)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(reindex())
