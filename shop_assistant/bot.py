"""Telethon bot: customer handler, escalation, owner relay, logging. SDD §3.8. Tickets #13, #14."""
import asyncio
import json
import logging
import re
import time
from datetime import datetime

from shop_assistant import config

log = logging.getLogger(__name__)

ESC_PREFIX = "#esc"
_ESC_RE = re.compile(r"^#esc (\d+)")

ERROR_REPLY = "Kechirasiz, xatolik yuz berdi. Birozdan keyin qayta urinib ko'ring."
ESCALATED_REPLY = "Egasi tez orada javob beradi 🙏"
OWNER_HINT_NOT_REPLY = "Mijozga javob berish uchun #esc xabariga reply qiling."
OWNER_HINT_NOT_ESC = "Bu #esc xabari emas."
OWNER_SENT = "✓ yuborildi"

# Set inside run() after the client starts; escalate_sync() (called from the agent's
# worker thread) uses it to schedule coroutines on the bot's loop.
main_loop: asyncio.AbstractEventLoop | None = None
_bot = None


# ---------------------------------------------------------------- pure helpers (#13, #14)

def log_turn(chat_id: int, question: str, tools: list[dict], answer: str,
             escalated: bool, ms: int, usd: float) -> dict:
    """Build + append one log.jsonl line (SDD §2.6, FR-25). Returns the record."""
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "chat_id": chat_id,
        "question": question,
        "tools": tools,
        "answer": answer,
        "escalated": escalated,
        "ms": ms,
        "usd": usd,
    }
    path = config.LOG_PATH  # read at call time — tests monkeypatch it
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def should_handle(is_private: bool, sender_id: int, owner_id: int) -> bool:
    """Customers only: private chat and not the owner (C-5)."""
    return bool(is_private) and sender_id != owner_id


def format_escalation(customer_id: int, question: str, links: list[str]) -> str:
    """'#esc <customer_id>\\n<question>\\n<links>' — header is the routing key (D-5)."""
    return f"{ESC_PREFIX} {customer_id}\n{question}\n" + "\n".join(links)


def parse_esc_header(text: str) -> int | None:
    """customer_id from a quoted '#esc <id>' message; None when the owner did not reply to one."""
    first_line = (text or "").split("\n", 1)[0]
    m = _ESC_RE.match(first_line)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- client (lazy: no .env at import)

def owner_id() -> int:
    return int(config.secret("TG_OWNER_ID"))


def _client():
    """Create the Telethon bot client on first use so importing bot.py never needs .env."""
    global _bot
    if _bot is None:
        from telethon import TelegramClient
        _bot = TelegramClient(str(config.SESSION_DIR / "bot"),
                              int(config.secret("TG_API_ID")),
                              config.secret("TG_API_HASH"))
    return _bot


# ---------------------------------------------------------------- handlers

async def handle_customer(event) -> None:
    """to_thread(run_agent) → reply; log the turn."""
    try:
        from shop_assistant import agent  # lazy: agent imports the LLM stack
        text = event.raw_text or "<media>"
        t0 = time.monotonic()
        reply = await asyncio.to_thread(agent.run_agent, event.chat_id, text)
        ms = int((time.monotonic() - t0) * 1000)
        await event.reply(reply)
        last = getattr(agent, "last_run", {}) or {}
        log_turn(event.chat_id, text, last.get("tools", []), reply,
                 last.get("escalated", False), ms, last.get("usd", 0.0))
    except Exception:
        log.exception("handle_customer failed for chat %s", getattr(event, "chat_id", "?"))
        try:
            await event.reply(ERROR_REPLY)
        except Exception:
            log.exception("could not send error reply")


async def escalate(customer_id: int, question: str, post_ids: list[int]) -> None:
    """Message the owner; tell the customer 'Egasi tez orada javob beradi' (FR-19)."""
    client = _client()
    links = [f"https://t.me/{config.CHANNEL}/{i}" for i in post_ids]
    await client.send_message(owner_id(), format_escalation(customer_id, question, links))
    await client.send_message(customer_id, ESCALATED_REPLY)


def escalate_sync(question: str, post_ids: list[int]) -> None:
    """Blocking wrapper for the agent's worker thread (tools run under to_thread)."""
    from shop_assistant import agent
    customer_id = agent.current_chat_id.get()
    if main_loop is None:
        raise RuntimeError("bot.main_loop is not set — escalate_sync() must run while the bot is running")
    fut = asyncio.run_coroutine_threadsafe(escalate(customer_id, question, post_ids), main_loop)
    fut.result(timeout=15)


async def handle_owner_reply(event) -> None:
    """Owner reply-to '#esc' → forward text to the customer (FR-21); hint if not a reply."""
    if not event.is_reply:
        await event.reply(OWNER_HINT_NOT_REPLY)
        return
    quoted = await event.get_reply_message()
    cid = parse_esc_header(quoted.raw_text if quoted else "")
    if cid is None:
        await event.reply(OWNER_HINT_NOT_ESC)
        return
    await _client().send_message(cid, event.raw_text)
    await event.reply(OWNER_SENT)


# ---------------------------------------------------------------- entry point (#15)

def run() -> None:
    """Start the bot client and register handlers."""
    global main_loop
    from telethon import events

    client = _client()
    owner = owner_id()

    async def dispatch(event) -> None:
        if event.sender_id == owner and event.is_private:
            await handle_owner_reply(event)
        elif should_handle(event.is_private, event.sender_id, owner):
            await handle_customer(event)
        # groups / channels: ignored (C-5)

    client.add_event_handler(dispatch, events.NewMessage(incoming=True))
    client.start(bot_token=config.secret("TG_BOT_TOKEN"))
    main_loop = client.loop
    log.info("bot started; owner=%s", owner)
    client.run_until_disconnected()
