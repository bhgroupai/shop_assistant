"""Nightly ingestion pipeline: fetch new posts -> extract pending -> incremental index. Ticket #18.

Run by the systemd user timer deploy/shop-assistant-ingest.timer (D-7: never inside the bot process):
    python -m shop_assistant.ingest
Contract: tests/test_ingest.py docstring.
"""
import logging
import sys

log = logging.getLogger(__name__)

INGEST_LOG_NAME = "gemini_ingest.jsonl"   # in config.DATA_DIR; same file bot.compute_stats counts (#17)


def run() -> bool:
    """fetch (min_id = state.last_post_id) -> extract pending posts -> index.update_index() -> write
    state.last_index_at. Every Gemini HTTP request made meanwhile appends one line to
    config.DATA_DIR / INGEST_LOG_NAME. A failing step logs ERROR, stops the run and returns False;
    previous files stay intact and pending posts stay pending. True on success."""
    raise NotImplementedError("ticket #18")


def main() -> int:
    """CLI entry point: 0 when run() succeeds, 1 otherwise (so systemd marks a failed night)."""
    raise NotImplementedError("ticket #18")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
