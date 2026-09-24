"""Telethon bot: customer handler, escalation, owner relay, logging. SDD §3.8. Tickets #13, #14."""
import asyncio
import io
import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path

from shop_assistant import config, lang
from shop_assistant.tools import TOOL_LINE_RE

log = logging.getLogger(__name__)

ESC_PREFIX = "#esc"
_ESC_RE = re.compile(r"^#esc (\d+)")

# Customer-facing texts live in lang.TEXTS (#24); these are the Uzbek Latin values, kept for reference.
ERROR_REPLY = lang.TEXTS["uz_latn"]["error_reply"]
ESCALATED_REPLY = lang.TEXTS["uz_latn"]["escalated_reply"]
OWNER_HINT_NOT_REPLY = "Mijozga javob berish uchun #esc xabariga reply qiling."
OWNER_HINT_NOT_ESC = "Bu #esc xabari emas."
OWNER_SENT = "✓ yuborildi"
MAX_FORWARD = 5  # posts illustrated per reply (ticket #19)
CAPTION_LIMIT = 1024  # Telegram media caption limit
STALE_NOTE = lang.TEXTS["uz_latn"]["stale_note"]

# Set inside run() after the client starts; escalate_sync() (called from the agent's
# worker thread) uses it to schedule coroutines on the bot's loop.
main_loop: asyncio.AbstractEventLoop | None = None
_bot = None


# ---------------------------------------------------------------- pure helpers (#13, #14)

def log_turn(chat_id: int, question: str, tools: list[dict], answer: str,
             escalated: bool, ms: int, usd: float, llm_calls: int = 1) -> dict:
    """Build + append one log.jsonl line (SDD §2.6, FR-25). Returns the record.
    Ticket #17: the record also carries `llm_calls` (Gemini requests this turn, see llm_calls_for)."""
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "chat_id": chat_id,
        "question": question,
        "tools": tools,
        "answer": answer,
        "escalated": escalated,
        "ms": ms,
        "usd": usd,
        "llm_calls": int(llm_calls),
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


def post_ids_in(text: str) -> list[int]:
    """Ids of `t.me/<config.CHANNEL>/<id>` links in `text`, in order of first appearance,
    without duplicates, at most 5. Used to forward the matching channel posts (ticket #19)."""
    pattern = re.compile(r"(?:https?://)?t\.me/" + re.escape(config.CHANNEL) + r"/(\d+)\b")
    ids: list[int] = []
    for m in pattern.finditer(text or ""):
        i = int(m.group(1))
        if i not in ids:
            ids.append(i)
            if len(ids) == MAX_FORWARD:
                break
    return ids


def strip_tool_lines(text: str | None) -> str:
    """Ticket #27: `text` without the listing tools' model-only lines (#25) — whole lines that are the
    range/total line (`ko'rsatildi 1–5, jami 23; keyingilari: shu filtrlar bilan offset=5`,
    `ko'rsatildi 21–23, jami 23; boshqa yo'q`) or the past-the-end line
    (`boshqa natija yo'q (jami 23, offset=30)`). The pattern lives next to the formatter in tools.py.
    Everything else is kept as is: product lines, the offer line, the model's own sentences (also
    "Jami 23 ta ..." / "Всего найдено ..."). Blank lines left behind are collapsed; None / "" → "".
    Applied by handle_customer to every customer reply (plain text and carousel)."""
    if not text:
        return ""
    out: list[str] = []
    for line in text.split("\n"):
        if TOOL_LINE_RE.match(line):
            continue
        if not line.strip() and (not out or not out[-1].strip()):
            continue                          # no leading blank, no run of blanks
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)



# ---------------------------------------------------------------- owner commands (ticket #17)

INGEST_LOG_NAME = "gemini_ingest.jsonl"  # in config.DATA_DIR; one line {"ts": ISO, ...} per ingestion Gemini request


def llm_calls_for(last_run: dict) -> int:
    """Gemini requests (LLM + embedding) of one agent turn, from agent.last_run:
    last_run["llm_calls"] when it is an int, else 1 + len(tools) + number of tools named
    "semantic_search_tool" (each one embeds the query)."""
    last_run = last_run or {}
    explicit = last_run.get("llm_calls")
    if isinstance(explicit, int) and not isinstance(explicit, bool):
        return explicit
    tools = last_run.get("tools") or []
    embeds = sum(1 for t in tools if isinstance(t, dict) and t.get("name") == "semantic_search_tool")
    return 1 + len(tools) + embeds


def _jsonl_today(path: Path, today: str):
    """Yield the records of a .jsonl file whose `ts` date is `today`; missing file → nothing."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping malformed line in %s", path)
                continue
            if str(rec.get("ts", ""))[:10] == today:
                yield rec


def compute_stats(today: date, log_path: Path | None = None, state_path: Path | None = None,
                  ingest_log_path: Path | None = None) -> dict:
    """Counts for /stats. Paths default (at call time) to config.LOG_PATH, config.STATE_PATH,
    config.DATA_DIR / INGEST_LOG_NAME. Only lines whose `ts` date == today count.
    Keys: indexed_posts (len(search.PRODUCTS)), last_index_at (state.json value or None),
    questions_today, escalations_today, gemini_today (sum of llm_calls — 1 when absent — plus
    today's ingest-log lines), gemini_limit (config.GEMINI_DAILY_LIMIT)."""
    from shop_assistant import search  # lazy: search loads the catalog at import
    log_path = log_path or config.LOG_PATH
    state_path = state_path or config.STATE_PATH
    ingest_log_path = ingest_log_path or (config.DATA_DIR / INGEST_LOG_NAME)
    day = today.isoformat()

    questions = escalations = gemini = 0
    for rec in _jsonl_today(log_path, day):
        questions += 1
        if rec.get("escalated"):
            escalations += 1
        n = rec.get("llm_calls")
        gemini += n if isinstance(n, int) and not isinstance(n, bool) else 1
    gemini += sum(1 for _ in _jsonl_today(ingest_log_path, day))

    last_index_at = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8") or "{}")
            last_index_at = state.get("last_index_at")
        except json.JSONDecodeError:
            log.warning("could not parse %s", state_path)

    return {
        "indexed_posts": len(search.PRODUCTS),
        "last_index_at": last_index_at,
        "questions_today": questions,
        "escalations_today": escalations,
        "gemini_today": gemini,
        "gemini_limit": config.GEMINI_DAILY_LIMIT,
    }


def format_stats(stats: dict) -> str:
    """Lines: 'Indexed posts: N', 'Last index: <ts or ->', 'Questions today: N',
    'Escalations today: N', 'Gemini: N / LIMIT today'."""
    return "\n".join([
        f"Indexed posts: {stats['indexed_posts']}",
        f"Last index: {stats['last_index_at'] or '-'}",
        f"Questions today: {stats['questions_today']}",
        f"Escalations today: {stats['escalations_today']}",
        f"Gemini: {stats['gemini_today']} / {stats['gemini_limit']} today",
    ])


async def handle_stats(event) -> None:
    """Owner /stats → reply format_stats(compute_stats(date.today()))."""
    await event.reply(format_stats(compute_stats(date.today())))


async def handle_reindex(event) -> None:
    """Owner /reindex → search.reload(), then reply with the new product count."""
    from shop_assistant import search
    await asyncio.to_thread(search.reload)  # disk read; keep the loop free
    await event.reply(f"Reindex OK: {len(search.PRODUCTS)} products")


def _command(text: str) -> str:
    """'/stats@shop_bot extra' → '/stats'; '' when the text is not a command."""
    parts = (text or "").strip().split(maxsplit=1)
    if not parts or not parts[0].startswith("/"):
        return ""
    return parts[0].split("@", 1)[0].lower()


async def route(event, owner: int) -> None:
    """Top-level NewMessage dispatch (run() uses it). Owner in private: '/stats' → handle_stats,
    '/reindex' → handle_reindex, anything else → handle_owner_reply. Customers (should_handle)
    → handle_customer, whatever the text (a customer's /stats is a normal question).
    Groups / channels: ignored."""
    if event.sender_id == owner and event.is_private:
        cmd = _command(event.raw_text)
        if cmd == "/stats":
            await handle_stats(event)
        elif cmd == "/reindex":
            await handle_reindex(event)
        else:
            await handle_owner_reply(event)
    elif should_handle(event.is_private, event.sender_id, owner):
        await handle_customer(event)
    # groups / channels: ignored (C-5)


# ---------------------------------------------------------------- carousel reply (ticket #19)

_PRICE_WORD = re.compile(r"^(?:narxi?|нархи?|цена)\s*:\s*", re.IGNORECASE)


def _is_ask_price(text: str) -> bool:
    """True when `text` carries the "price: I'll ask" phrase in any language (lang.TEXTS)."""
    low = (text or "").lower()
    return any(t["ask_price"].lower() in low for t in lang.TEXTS.values())


def _fmt_price(p: str, code: str = lang.DEFAULT) -> str:
    digits = p.replace(".", "").replace(" ", "")
    if digits.isdigit():
        return f"{int(digits):,}".replace(",", " ") + " " + lang.CURRENCY[code]
    if _is_ask_price(p):                       # "narxi: so'rab beraman" → "so'rab beraman" in the chat's language
        p = lang.TEXTS[code]["ask_price"]
    return _PRICE_WORD.sub("", p)


_NUMBERED = re.compile(r"^(\d+)\. (.+)$", re.MULTILINE)


def _numbered_lines(reply: str) -> list[tuple[int, str]]:
    """[(item number, rest of the line)] of the numbered lines of `reply`, in order."""
    return [(int(m.group(1)), m.group(2)) for m in _NUMBERED.finditer(reply or "")]


def _first_number(reply: str) -> int:
    """The number of the first numbered line (6 on a second page, #25); 1 when there is none."""
    lines = _numbered_lines(reply)
    return lines[0][0] if lines else 1


def carousel_caption(reply: str, current: int, lang: str = lang.DEFAULT) -> str:
    """Card for the `current`-th numbered line of `reply` (0-based position, not the number, #25): title
    line `N. name` with the line's own number, then price, sizes (omitted when '-'), date and a stale
    warning when the line carries one; labels and the note from lang.TEXTS[lang] (#24). Falls back to the
    whole reply when that line does not exist. Cut to CAPTION_LIMIT."""
    t = _texts(lang)
    lines = _numbered_lines(reply)
    if not 0 <= current < len(lines):
        return (reply or "")[:CAPTION_LIMIT]
    n, line = lines[current]
    stale = "eskirgan" in line.lower()
    line = re.sub(r"\s*\(.*?\)\s*$", "", line)          # trailing "(… eskirgan …)" note
    parts = [p.strip() for p in line.split(" · ")]
    parts = [p for p in parts if p and not p.startswith("[") and "t.me/" not in p]
    name, price, sizes, date = (parts + ["", "", "", ""])[:4]
    out = [f"{n}. {name}", f"{t['price_label']} {_fmt_price(price, t['_code'])}"]
    if sizes and sizes != "-":
        out.append(f"{t['sizes_label']} " + ", ".join(s.strip() for s in sizes.split(",")))
    if date:
        out.append(f"{t['date_label']} {date}")
    if stale:
        out.append(t["stale_note"])
    return "\n".join(out)[:CAPTION_LIMIT]


def carousel_data(kind: str, idx: int, ids: list[int]) -> bytes:
    """Callback payload `<kind>:<idx>:<id,id,…>` as bytes; kind is 'c' (navigate) or 'p' (ask price)."""
    return f"{kind}:{idx}:{','.join(str(i) for i in ids)}".encode()


def parse_carousel_data(data: bytes) -> tuple[str, int, list[int]] | None:
    """Inverse of carousel_data; None for anything malformed or unknown kind."""
    try:
        text = (data or b"").decode()
    except UnicodeDecodeError:
        return None
    m = re.fullmatch(r"([cp]):(\d+):(\d+(?:,\d+)*)", text)
    if not m:
        return None
    return m.group(1), int(m.group(2)), [int(i) for i in m.group(3).split(",")]




def _texts(code: str) -> dict:
    """lang.TEXTS for `code` (unknown → Uzbek Latin), plus the resolved code under '_code'."""
    code = code if code in lang.TEXTS else lang.DEFAULT
    return {**lang.TEXTS[code], "_code": code}


def _chat_lang(chat_id: int) -> str:
    """The stored language of `chat_id` (lang.get_store()); Uzbek Latin when nothing is stored."""
    try:
        return lang.get_store().get(chat_id) or lang.DEFAULT
    except Exception:
        log.exception("language store lookup failed for chat %s", chat_id)
        return lang.DEFAULT


def _inline(text: str, data: bytes):
    from telethon import Button
    return Button.inline(text, data)


def _url(text: str, url: str):
    from telethon import Button
    return Button.url(text, url)


def carousel_buttons(idx: int, ids: list[int], ask_price: bool, lang: str = lang.DEFAULT,
                     first: int = 1) -> list[list]:
    """Inline keyboard rows: row 0 = ◀ · `<first+idx>/<first+n-1>` · ▶ (omitted when n == 1; `first` is the
    number of the reply's first item, 6 on a second page, #25; callback data stays position-based);
    last row = ask-price (only when ask_price) + view-in-channel url button, labels in `lang` (#24)."""
    t = _texts(lang)
    n = len(ids)
    rows: list[list] = []
    if n > 1:
        rows.append([
            _inline("◀", carousel_data("c", (idx - 1) % n, ids)),
            _inline(f"{first + idx}/{first + n - 1}", carousel_data("c", idx, ids)),
            _inline("▶", carousel_data("c", (idx + 1) % n, ids)),
        ])
    last: list = []
    if ask_price:
        last.append(_inline(t["ask_price_button"], carousel_data("p", idx, ids)))
    last.append(_url(t["view_in_channel_button"], f"https://t.me/{config.CHANNEL}/{ids[idx]}"))
    rows.append(last)
    return rows


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

# Carousel state. _media_cache: channel post id -> photo/video, or None when the post has neither
# (so ◀/▶ never refetch). _captions: carousel message id -> full agent reply; empty after a restart,
# then the caption is rebuilt from the catalog.
_media_cache: dict[int, object] = {}
_placeholder_media = None
_captions: dict[int, str] = {}


async def _placeholder(client):
    """The channel's profile photo for posts without a photo/video. Bots may not list profile
    photos, so it is downloaded once as bytes; a named buffer makes Telegram treat it as a photo.
    After the first send the uploaded Photo replaces it (no re-upload)."""
    global _placeholder_media
    if _placeholder_media is None:
        _placeholder_media = await client.download_profile_photo(config.CHANNEL, file=bytes) or None
    if isinstance(_placeholder_media, (bytes, bytearray)):
        buf = io.BytesIO(_placeholder_media)
        buf.name = "logo.jpg"
        return buf
    return _placeholder_media


def _remember_placeholder(sent_media, msg) -> None:
    """After sending the placeholder buffer, keep the resulting Photo so later sends don't re-upload."""
    global _placeholder_media
    if isinstance(sent_media, io.BytesIO) and msg is not None and getattr(msg, "photo", None):
        _placeholder_media = msg.photo


async def _media_for(client, ids: list[int]) -> dict[int, object]:
    """{post id: photo or video} for the channel posts `ids`; posts without one get the channel
    logo so every item of the reply is navigable and numbering stays 1:1. Cached per id."""
    missing = [i for i in ids if i not in _media_cache]
    if missing:
        msgs = await client.get_messages(config.CHANNEL, ids=missing)
        for i, m in zip(missing, msgs):
            _media_cache[i] = (m.photo or m.video) if m is not None else None
    fallback = await _placeholder(client)
    return {i: (_media_cache.get(i) or fallback) for i in ids if (_media_cache.get(i) or fallback) is not None}


def _ask_price_on(reply: str, number_idx: int) -> bool:
    """True when the `number_idx`-th numbered line of `reply` (0-based position, #25) has no price
    (the ask-price phrase in any language)."""
    lines = _numbered_lines(reply)
    return 0 <= number_idx < len(lines) and _is_ask_price(lines[number_idx][1])


async def send_reply(event, reply: str, lang: str = lang.DEFAULT) -> None:
    """One message: item 1's photo/video (channel logo when the post has none), its card as
    caption and inline ◀/▶ buttons that edit it in place (#19).
    Falls back to plain text (no link preview) when there is no media or the send fails."""
    ids = post_ids_in(reply)
    if ids and len(reply) <= CAPTION_LIMIT:
        try:
            media = await _media_for(event.client, ids)
            if ids[0] in media:
                idx = 0
                msg = await event.client.send_file(
                    event.chat_id, media[ids[0]],
                    caption=carousel_caption(reply, idx, lang),
                    buttons=carousel_buttons(idx, ids, _ask_price_on(reply, idx), lang,
                                             first=_first_number(reply)),
                    reply_to=event.message.id)
                _remember_placeholder(media[ids[0]], msg)
                if msg is not None and getattr(msg, "id", None) is not None:
                    if len(_captions) >= 2000:  # bounded; older ones are rebuilt from the catalog
                        _captions.pop(next(iter(_captions)))
                    _captions[msg.id] = reply
                return
        except Exception as e:
            log.warning("carousel to chat %s failed for posts %s: %s", event.chat_id, ids, e)
    await event.reply(reply, link_preview=False)


def _rebuild_reply(ids: list[int]) -> str:
    """After a restart (_captions empty): the numbered list for `ids` straight from the catalog."""
    from shop_assistant import search, tools
    by_id = {p.id: p for p in search.PRODUCTS}
    return tools.format_products([by_id[i] for i in ids if i in by_id])


async def handle_callback(event) -> None:
    """Inline button press on a carousel message: 'c' → edit media/caption/buttons in place,
    'p' → escalate the current post's price to the owner. Owner presses are ignored.
    Texts in the chat's stored language (#24), so buttons stay localized after a restart."""
    code = lang.DEFAULT
    try:
        if event.sender_id == owner_id():
            return
        parsed = parse_carousel_data(event.data)
        if parsed is None:
            await event.answer()
            return
        kind, idx, ids = parsed
        code = _chat_lang(event.sender_id)
        if not 0 <= idx < len(ids):
            await event.answer()
            return
        if kind == "p":
            await escalate(event.sender_id, f"Narxi? https://t.me/{config.CHANNEL}/{ids[idx]}", [ids[idx]])
            log.info("price escalation from chat %s for post %s", event.sender_id, ids[idx])
            await event.answer(lang.TEXTS[code]["callback_toast"], alert=False)
            return
        media = await _media_for(event.client, ids)
        if ids[idx] not in media:
            await event.answer()
            return
        reply = _captions.get(event.message_id) or _rebuild_reply(ids)
        msg = await event.edit(carousel_caption(reply, idx, code), file=media[ids[idx]],
                               buttons=carousel_buttons(idx, ids, _ask_price_on(reply, idx), code,
                                                        first=_first_number(reply)),
                               link_preview=False)
        _remember_placeholder(media[ids[idx]], msg)
        await event.answer()
    except Exception:
        log.exception("handle_callback failed for chat %s", getattr(event, "sender_id", "?"))
        try:
            await event.answer(lang.TEXTS[code]["callback_error"])
        except Exception:
            log.exception("could not answer callback")


async def _lang_code(event) -> str | None:
    """Telegram sender.lang_code of the event; None when there is no sender (channels, anonymous)."""
    try:
        sender = getattr(event, "sender", None)
        if sender is None and hasattr(event, "get_sender"):
            sender = await event.get_sender()
        return getattr(sender, "lang_code", None)
    except Exception:
        log.warning("no sender for chat %s", getattr(event, "chat_id", "?"))
        return None


async def handle_customer(event) -> None:
    """Resolve the chat's language (#24); '/start' → localized greeting (no LLM);
    otherwise to_thread(run_agent, lang=…) → reply; log the turn."""
    code = lang.DEFAULT
    try:
        from shop_assistant import agent  # lazy: agent imports the LLM stack
        text = event.raw_text or "<media>"
        code = _chat_lang(event.chat_id)       # used by the error reply if resolving itself fails
        code = lang.resolve(event.chat_id, event.raw_text or "", await _lang_code(event))
        if _command(text) == "/start":
            await event.reply(lang.TEXTS[code]["greeting"])
            return
        t0 = time.monotonic()
        reply = await asyncio.to_thread(agent.run_agent, event.chat_id, text, lang=code)
        reply = strip_tool_lines(reply)       # #27: the tools' paging line is for the model only
        ms = int((time.monotonic() - t0) * 1000)
        await send_reply(event, reply, code)
        last = getattr(agent, "last_run", {}) or {}
        log_turn(event.chat_id, text, last.get("tools", []), reply,
                 last.get("escalated", False), ms, last.get("usd", 0.0),
                 llm_calls=llm_calls_for(last))
    except Exception:
        log.exception("handle_customer failed for chat %s", getattr(event, "chat_id", "?"))
        try:
            await event.reply(lang.TEXTS[code]["error_reply"])
        except Exception:
            log.exception("could not send error reply")


async def escalate(customer_id: int, question: str, post_ids: list[int]) -> None:
    """Message the owner (Uzbek); tell the customer 'the owner will answer soon' in their stored
    language (FR-19, #24)."""
    client = _client()
    links = [f"https://t.me/{config.CHANNEL}/{i}" for i in post_ids]
    await client.send_message(owner_id(), format_escalation(customer_id, question, links))
    await client.send_message(customer_id, lang.TEXTS[_chat_lang(customer_id)]["escalated_reply"])


def escalate_sync(question: str, post_ids: list[int]) -> None:
    """Blocking wrapper for the agent's worker thread (tools run under to_thread)."""
    from shop_assistant import agent
    customer_id = agent.current_chat_id.get()
    if main_loop is None:
        raise RuntimeError("bot.main_loop is not set — escalate_sync() must run while the bot is running")
    fut = asyncio.run_coroutine_threadsafe(escalate(customer_id, question, post_ids), main_loop)
    fut.result(timeout=15)


async def handle_owner_reply(event) -> None:
    """Owner reply-to '#esc' → forward text to the customer (FR-21), then save it to the FAQ (FR-22); hint if not a reply."""
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
    await _save_faq(quoted.raw_text, event.raw_text)


def esc_question(quoted: str) -> str:
    """The customer's question from a quoted '#esc' message: without the header line and the t.me link lines."""
    lines = (quoted or "").split("\n")[1:]
    return "\n".join(ln for ln in lines if "t.me/" not in ln).strip()


async def _save_faq(quoted: str, answer: str) -> None:
    """FR-22 (#23.7): the relayed owner answer goes to the FAQ store. Runs after the relay; never raises."""
    from shop_assistant import faq   # looked up at call time (tests replace faq.add)
    try:
        await asyncio.to_thread(faq.add, esc_question(quoted), answer, post_ids_in(quoted))
    except Exception:
        log.exception("owner answer relayed but not saved to the FAQ")


# ---------------------------------------------------------------- entry point (#15)

def run() -> None:
    """Start the bot client and register handlers."""
    global main_loop
    from telethon import events

    client = _client()
    owner = owner_id()

    async def dispatch(event) -> None:
        await route(event, owner)  # owner commands, owner replies, customers (#17)

    client.add_event_handler(dispatch, events.NewMessage(incoming=True))
    client.add_event_handler(handle_callback, events.CallbackQuery())
    client.start(bot_token=config.secret("TG_BOT_TOKEN"))
    main_loop = client.loop
    log.info("bot started; owner=%s", owner)
    client.run_until_disconnected()
