"""Nightly ingestion pipeline: fetch new posts -> extract pending -> incremental index. Ticket #18.

Run by the systemd user timer deploy/shop-assistant-ingest.timer (D-7: never inside the bot process):
    python -m shop_assistant.ingest
Contract: tests/test_ingest.py docstring.
"""
import datetime
import json
import logging
import sys

from shop_assistant import config, extract, fetch, index

log = logging.getLogger(__name__)

INGEST_LOG_NAME = "gemini_ingest.jsonl"   # in config.DATA_DIR; same file bot.compute_stats counts (#17)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _log_request(kind: str, model: str, n: int) -> None:
    """One line per Gemini HTTP request of the nightly run (read by bot.compute_stats, #17)."""
    line = json.dumps({"ts": _now(), "kind": kind, "model": model, "n": n}, ensure_ascii=False)
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.DATA_DIR / INGEST_LOG_NAME, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:   # the stats line is not worth failing the night for
        log.warning("cannot write %s (%s)", INGEST_LOG_NAME, e)


def _read_state() -> dict:
    try:
        with open(config.STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _write_last_index_at() -> None:
    """state.json "last_index_at" = now, every other key kept; atomic (temp file + os.replace)."""
    state = _read_state()
    state["last_index_at"] = _now()

    def write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)

    index._atomic_write(config.STATE_PATH, write)


def _steps() -> None:
    """The pipeline; raises on the first failing step."""
    fetch.main()
    if not extract.main():
        raise RuntimeError("extraction stopped before all pending posts were extracted (see the ERROR above); "
                           "they stay pending for the next night")
    added = index.update_index()
    log.info("index: %d new rows", added)


def run() -> bool:
    """fetch (min_id = state.last_post_id) -> extract pending posts -> index.update_index() -> write
    state.last_index_at. Every Gemini HTTP request made meanwhile appends one line to
    config.DATA_DIR / INGEST_LOG_NAME. A failing step logs ERROR, stops the run and returns False;
    previous files stay intact and pending posts stay pending. True on success."""
    extract.on_request = index.on_request = _log_request
    try:
        _steps()
        _write_last_index_at()
    except Exception as e:   # a failed night must never crash: systemd gets exit 1, the bot is untouched
        log.error("nightly ingestion failed: %s: %s", type(e).__name__, e)
        return False
    finally:
        extract.on_request = index.on_request = None
    log.info("nightly ingestion done")
    return True


def main() -> int:
    """CLI entry point: 0 when run() succeeds, 1 otherwise (so systemd marks a failed night)."""
    return 0 if run() else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
