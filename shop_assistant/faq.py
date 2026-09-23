"""FAQ store: the owner's relayed answers, embedded and searched by meaning. SDD §2.4, §3.8, FR-22. Ticket #23.7.

Contract: tests/test_faq.py (module docstring). In short:
- data/faq.jsonl (config.FAQ_PATH): one {"ts", "question", "answer", "post_ids"} per owner answer, append-only.
- data/faq_embeddings.npy (config.FAQ_EMBEDDINGS_PATH): row i = index.embed([question of line i], kind="document").
- data/faq_embeddings_meta.json (config.FAQ_META_PATH): {"model", "dim", "n"} — the model guard, as for products.
- add() saves first, then tries to embed every entry that has no row yet; an embedding failure never raises.
- search() embeds the query with kind="query" and returns entries whose cosine >= THRESHOLD.
All paths are read from config at call time.
"""
import logging

from shop_assistant.models import FaqEntry

log = logging.getLogger(__name__)

THRESHOLD = 0.75   # min cosine(question, query) for a hit; TODO(#23.7): measure on real paraphrases


def load() -> list[FaqEntry]:
    """faq.jsonl → FaqEntry list in file order (post_ids as a tuple). Missing file → []; bad lines skipped."""
    raise NotImplementedError("ticket #23.7")


def add(question: str, answer: str, post_ids=(), ts: str | None = None) -> FaqEntry:
    """Append one entry to faq.jsonl (ts defaults to now, ISO seconds), then sync(). Returns the entry.
    Embedding errors (quota, network, model mismatch) are logged, never raised: the entry stays saved
    and is embedded by the next successful add()/search()."""
    raise NotImplementedError("ticket #23.7")


def sync() -> int:
    """Embed the entries that have no row yet (one index.embed(..., kind="document") call, retry=False) and
    rewrite faq_embeddings.npy + meta. Returns how many were embedded. Raises on embedding errors;
    does nothing (returns 0) when the stored matrix was built by another model / dimension."""
    raise NotImplementedError("ticket #23.7")


def search(query: str, k: int = 3) -> list[FaqEntry]:
    """Top-k entries with cosine >= THRESHOLD, best first. Embeds pending entries first (failure → go on
    with what is embedded). Query via index.embed([...], kind="query"); failure → [] + one WARNING.
    No embedded entries → [] without a request. Model / dimension mismatch → [] + one ERROR, no request."""
    raise NotImplementedError("ticket #23.7")


def reindex() -> int:
    """Re-embed every entry with the current config.EMBED_MODEL (retry=True) and rewrite matrix + meta.
    The fix after a model change. Returns the number of entries."""
    raise NotImplementedError("ticket #23.7")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(reindex())
