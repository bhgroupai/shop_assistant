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
EXTRACT_BATCH = 5          # NFR-3; TODO(#23): measure 5 vs 10 vs 20 on Gemini (scripts/spike_gemini.py) and keep the largest 100%-valid size
MIN_PRICE = 10_000         # so'm; smaller "prices" from the model are junk → None
EMBED_BATCH = 128          # NFR-3

# Gemini API, free tier, via google-genai (SDD §1.1, SRS C-2). Agent + extraction; shared client in llm.py.
# TODO(#23): confirm this is the current free-tier Flash model and fill in its limits from
#   AI Studio -> Rate limits (do not copy numbers from memory or blog posts):
#   requests/minute = ?, requests/day = ?   (one customer message averages 1.8 requests, max 4)
GEMINI_MODEL = "gemini-2.5-flash"
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
