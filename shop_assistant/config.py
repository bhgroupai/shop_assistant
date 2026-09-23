"""Constants and secret names. SDD §3.9. Ticket #1."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CHANNEL = os.environ.get("TG_CHANNEL", "")   # shop channel username without @; set in .env
STALE_DAYS = 60            # FR-16
MAX_RESULTS = 5            # FR-18
FETCH_LIMIT = 500          # FR-1
CATEGORIES = ["kiyim", "poyabzal", "aksessuar", "boshqa"]   # SDD §2.2
HISTORY_TURNS = 10         # FR-17
EXTRACT_BATCH = 20         # NFR-3; spike 2026-09-23: 5/10/20 all 100% valid → 20 (7 requests for 134 posts)
MIN_PRICE = 10_000         # so'm; smaller "prices" from the model are junk → None
EMBED_BATCH = 128          # NFR-3

# Gemini API, free tier, via google-genai (SDD §1.1, SRS C-2). Agent + extraction; shared client in llm.py.
# Measured 2026-09-23 on the free key: gemini-3.5-flash / 2.5-flash = 5 requests/minute (429 quotaValue),
#   too few for customers (1.8 requests per message, max 4); 3.5-flash-lite took 16 requests in 15 s with
#   no 429, median 0.77 s, and called find_products on 10/10 spike runs. Extraction is offline and ~7
#   requests a night, so it uses the stronger model (fewer mis-categorised items than flash-lite).
# TODO: requests/day per model — read from AI Studio -> Rate limits (not exposed by the API).
GEMINI_MODEL = "gemini-3.5-flash-lite"   # customer agent
GEMINI_EXTRACT_MODEL = "gemini-3.5-flash"   # post extraction
MAX_ITERATIONS = 8               # model requests per customer turn
EMBED_MODEL = "bge-m3"           # Ollama, multilingual embeddings, 1024-d — until #23.5
GEMINI_DAILY_LIMIT = 250        # free-tier requests/day shown by /stats (#17); TODO(#23): set from AI Studio
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SESSION_DIR = ROOT / "session"
POSTS_PATH = DATA_DIR / "posts.jsonl"
PRODUCTS_PATH = DATA_DIR / "products.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_IDS_PATH = DATA_DIR / "embeddings_ids.json"
FAQ_PATH = DATA_DIR / "faq.jsonl"
FAQ_EMBEDDINGS_PATH = DATA_DIR / "faq_embeddings.npy"
STATE_PATH = DATA_DIR / "state.json"
LOG_PATH = DATA_DIR / "log.jsonl"
LANGUAGES_PATH = DATA_DIR / "languages.json"   # chat id → language code (#24, FR-13a)

# Secrets — read lazily so importing config never fails without .env (NFR-6).
ENV_KEYS = ("TG_CHANNEL", "TG_API_ID", "TG_API_HASH", "TG_BOT_TOKEN", "TG_OWNER_ID", "GEMINI_API_KEY")


def secret(name: str) -> str:
    """Return env var `name` or raise a clear error naming the missing key."""
    if name not in ENV_KEYS:
        raise KeyError(f"{name} is not a known secret; see ENV_KEYS")
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set — add it to .env")
    return value
