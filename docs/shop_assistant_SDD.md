# Software Design Description — Shop Assistant

Version 1.4 · 2026-09-23 · Status: draft · Implements: shop_assistant_SRS.md v0.5

## 1. Overview

Six stages, one Python module each, every module runnable on its own (NFR-7). Data flows left to right through files on disk; the bot process only reads them.

```
 offline (ingest.py, nightly timer)                   online (bot service)
 ─────────────────────────────                        ────────────────────
 @<channel>                                           customer ⇄ Telegram bot
      │ fetch.py (Telethon, user account)                        │
      ▼                                                          ▼ bot.py
 data/posts.jsonl        raw captions                      agent.py  (Gemini, manual loop) 
      │ extract.py (Gemini)                                     │ tools.py
      ▼                                                         ├─ find_products ─┐
 data/products.jsonl     structured records ◀───────────────────┤                 │ search.py
      │ index.py (Gemini embed)                                 ├─ semantic_search┘
      ▼                                                         ├─ latest_posts
 data/embeddings.npy + _ids.json + _meta.json ◀─────────────────┤
                                                                ├─ search_faq ──▶ faq.py ──▶ data/faq.jsonl + faq_embeddings*
                                                                └─ ask_owner ───▶ owner (same bot) ──reply──▶ customer
                                                                                   (the reply is also saved: faq.add)
```

### 1.1 Technology
| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | same as the rest of the repo |
| Telegram, channel history | Telethon **user account** | bots cannot read channel history; session pattern reused from `telegram_digest/tgclient.py` |
| Telegram, customers + owner | Telethon **bot account** (BotFather token) | customers must not talk to a personal account; bot can message the owner; pattern from `telegram_digest/approval.py` |
| LLM | **Gemini API, free tier** (hosted by Google), a Flash model (`config.GEMINI_MODEL`) with function calling, through the official `google-genai` SDK. One shared client in `llm.py`, created lazily from `GEMINI_API_KEY`. The Anthropic SDK cannot be used: Gemini has no Anthropic-compatible endpoint | C-2 v0.5: no local models; removes the cold model load (54 s) and the GPU dependency. Privacy trade-off: questions and captions go to Google (SRS C-2) |
| Embeddings | **Gemini embedding API** (`config.EMBED_MODEL = gemini-embedding-001`, `EMBED_DIM = 768`) through the same `llm.client()` and `GEMINI_API_KEY` (#23.5) | covers uz-Latin / uz-Cyrillic / ru (spike 2026-09-23: spelling variants ≥ 0.96 normalised); no local model left, so any host with Python can run the bot and the ingest |
| Vector store | `numpy` array + cosine similarity | ≤ a few thousand posts; a DB adds nothing to learn yet |
| Storage | JSONL files under `data/` | greppable, diffable, restart-safe (FR-6) |
| Secrets | `.env` via `python-dotenv` | NFR-6 |
| Deploy | rsync + user-level systemd, copy of `telegram_digest/deploy.sh` | NFR-5, C-3 |

## 2. Data Model

All files live in `shop_assistant/data/` (gitignored). One JSON object per line.

### 2.1 `posts.jsonl` — raw, written by fetch.py
```json
{"id": 1234, "date": "2026-09-10T14:02:00", "link": "https://t.me/example_shop/1234",
 "caption": "🍂Kuz mavsumi uchun🍂\n🔥Yangi model Dvoyka🔥\nRazmer:M.L.XL.2XL.3XL\nNarx:980.000ming\n...",
 "has_media": true}
```

### 2.2 `products.jsonl` — one record per post, written by extract.py
```json
{"id": 1234, "date": "2026-09-10", "link": "https://t.me/example_shop/1234",
 "name": "Dvoyka", "category": "kiyim",
 "price": 980000, "subscriber_price": null,
 "sizes": ["M","L","XL","2XL","3XL"], "colors": [],
 "keywords": ["dvoyka","двойка","костюм двойка","two-piece set","sport kostyum"],
 "season": "kuz", "body": "Yangi model Dvoyka Razmer M L XL 2XL 3XL"}
```
- `category` ∈ fixed list: `kiyim`, `poyabzal`, `aksessuar`, `boshqa` (extend when the channel shows more).
- `body` = caption with footer (FR-3a) and emoji stripped; this is the text that gets embedded.
- `keywords` are produced by the LLM in four scripts/languages (FR-4) and stored already **normalised** (§4.2).

### 2.3 `embeddings.npy` + `embeddings_ids.json`
`float32[N, D]` matrix (rows L2-normalised), row *i* belongs to post `ids[i]`. Next to them `embeddings_meta.json` = `{"model", "dim", "n"}` records which embedding model built the matrix. All three rewritten together by index.py; vectors from two models are never mixed (§3.5 load guard).

### 2.4 `faq.jsonl` — owner answers, appended by bot.py (FR-22)
```json
{"ts": "...", "question": "dastavka Samarqandga qancha?", "answer": "35 ming, 2 kun", "post_ids": [1234]}
```
Append-only, UTF-8, one line per relayed owner answer (`question` = the escalated customer question, `post_ids` = the post links of the `#esc` message). Written and read only by `faq.py` (§3.5a).

Next to it (#23.7): `faq_embeddings.npy` — `float32[n, EMBED_DIM]`, row *i* = `index.embed([normalise(question of line i)], kind="document")`; lines `n…` of `faq.jsonl` have no row yet ("pending") and are embedded by the next successful `add` / `search`. `faq_embeddings_meta.json` (`config.FAQ_META_PATH`) = `{"model", "dim", "n"}` of that matrix — the same guard as §2.3: an `.npy` without meta or built by another model / dimension is never used or extended until `python -m shop_assistant.faq` rebuilds it.

### 2.5 `state.json`
`{"last_post_id": 1234, "last_index_at": "2026-09-24T03:00:41"}` — drives incremental ingestion (FR-7) and `/stats` (FR-26). `last_post_id` is written by the fetch step; `last_index_at` only by a nightly run whose every step succeeded (a night with nothing new counts), other keys kept.

### 2.6 `log.jsonl` — one line per customer turn (FR-25)
`{"ts", "chat_id", "question", "tools": [{"name", "input", "n_results"}], "answer", "escalated": bool, "ms", "usd", "llm_calls": int}`
(`llm_calls` = Gemini requests that turn, LLM + query embeddings; older lines without it count 1.)

### 2.6a `gemini_ingest.jsonl` — one line per ingestion Gemini request, written by ingest.py (#18)
`{"ts": "2026-09-24T03:00:12", "kind": "extract" | "embed", "model": "gemini-3.5-flash", "n": 5}` — every HTTP request of the nightly run, retried and failed attempts included; `n` = posts / texts in that request (one embed request of 5 texts is one line, but it spends 5 of the 100 texts/min embed quota). Customer query embeddings are **not** written here (they count in `log.jsonl`). `/stats` adds today's lines to the day's Gemini total (#17).

### 2.7 `languages.json` — customer language per chat, written by lang.py (FR-13a)
One JSON object (not JSON lines): `{"<chat_id>": "uz_latn" | "uz_cyrl" | "ru"}`. Only chat id → language code, no message text (NFR-4). Written only when a chat's language changes, atomically (temp file in the same dir + rename). Missing or corrupt → one WARNING, empty store, languages are re-detected.

## 3. Components

### 3.1 `fetch.py` — FR-1, FR-2, FR-3, FR-7
- `fetch(channel, min_id, limit=500) -> list[Post]` via `tg.iter_messages(channel, min_id=min_id, limit=limit)`.
- Skip empty captions. Dedup on `normalise(caption)` keeping the newest id.
- Append to `posts.jsonl`, update `state.last_post_id`.
- CLI: `python -m shop_assistant.fetch [--full]`.

### 3.2 `extract.py` — FR-3a, FR-3b, FR-4
- `strip_footer(caption) -> body`: drop lines matching phone / `@handle` / `📍` / delivery boilerplate; strip emoji.
- `extract(body) -> Product` — one Gemini call (`config.GEMINI_EXTRACT_MODEL`) through `llm.client()` with the `record_products` function **forced** (function-calling mode `ANY`, allowed names = `record_products`) so the output is always a structured call. The schema is `extract._TOOL` (Anthropic-style), converted by `llm.tool_declarations`. Prompt gives the fixed category list, price notation examples (`980.000ming` → 980000), and asks for keywords in uz-Latin, uz-Cyrillic, ru, en.
- The deterministic guards (price notation, `_sane_price`, `product_name`, `is_announcement`) run on the model output unchanged: they do not depend on the model.
- Batches of `config.EXTRACT_BATCH` posts per call (NFR-3; fewer requests = less free quota). Timeout `llm.EXTRACT_TIMEOUT_S` per request.
- Errors: a 429 / 5xx / timeout is retried with exponential backoff; on the final failure the batch raises and `main()` logs an ERROR and stops, so no half batch is written and the posts stay un-extracted for the next run. `main()` returns `True` when every pending batch was written (or nothing was pending), `False` when it stopped — that is how `ingest.run()` learns the step failed.
- CLI: `python -m shop_assistant.extract` processes posts not yet in `products.jsonl`.

### 3.3 `index.py` — FR-5, FR-6
- `embed(texts, kind="document", *, retry=False) -> float32[N, EMBED_DIM]` — Gemini `models.embed_content` through `llm.client()` with `config.EMBED_MODEL` and `output_dimensionality = config.EMBED_DIM`, one request per batch of `config.EMBED_BATCH` (≤ 100 texts, the API limit). Task type: `RETRIEVAL_DOCUMENT` for products / FAQ answers (`kind="document"`), `RETRIEVAL_QUERY` for the customer's query (`kind="query"`). Rows are L2-normalised (reduced dimensions are not unit length).
- Errors: `retry=True` (index CLI only, offline) retries 429 / 5xx / timeouts up to `index.RETRIES` attempts per batch with growing sleeps; an invalid key is never retried. `retry=False` (query path) makes one attempt and lets the error propagate to `search.semantic_search`.
- **Embed the normalised text**: `embed([normalise(product_text(p)) …])`. Spike #3 measured the old local model cross-script raw at 0.62–0.67 (`krossovka`/`кроссовка`) but 0.87 after transliteration — so D-3 applies to embeddings too, not only to keywords. `scripts/spike_gemini_embed.py` (2026-09-23) repeats it for Gemini: see D-3.
- Embeds `name + " " + body + " " + " ".join(keywords)` per product. `reindex()` (the CLI) rewrites `embeddings.npy` / `embeddings_ids.json` / `embeddings_meta.json` for all products. If embedding fails, none of the files is touched.
- **Incremental update (#18, nightly):** `update_index() -> int` embeds (retry=True) only the products of `products.jsonl` whose ids are not yet in `embeddings_ids.json`, appends their rows after the existing ones (existing rows copied, never re-embedded) and rewrites the three files atomically (temp file + `os.replace`, meta last). Nothing new → 0, no request, no file touched (so the bot does not reload for nothing). No matrix yet → built from all products. A matrix from another model / dimension, or without `embeddings_meta.json`, raises `ModelMismatch` with no request and nothing written: a full re-embed spends the free quota the customers share, so it is only ever `python -m shop_assistant.index` by hand.
- Same for `faq.jsonl` → `faq_embeddings.npy`.
- CLI: `python -m shop_assistant.index`.

### 3.4 `textnorm.py` — FR-9
- `normalise(s) -> str`: lowercase → Cyrillic→Latin transliteration table (uz + ru letters) → `oʻ o' o` / `gʻ g' g` / `ў o` / `ғ g` / `ҳ h` / `қ q` folded → collapse whitespace.
- Used on keywords at extract time and on customer filter values at search time, so `krossovka`, `кроссовка`, `Krossovka` all match.

### 3.5 `search.py` — FR-8, FR-10, FR-11, FR-12
```python
def find_products(category=None, min_price=None, max_price=None, size=None, color=None, keywords=None, limit=5) -> list[Product]
def semantic_search(text, max_price=None, limit=5) -> list[Product]     # embed([normalise(text)], kind="query"), cosine, then price filter
def latest_posts(n=5) -> list[Product]
def search_faq(text, limit=3) -> list[FaqEntry]
```
- Filters are ANDed; `keywords` matches if **any** normalised customer keyword is a substring of any normalised product keyword or of `normalise(name)`.
- Every returned product carries `link`, `date`, and `stale: bool` (`date` older than `config.STALE_DAYS = 60`, FR-16).
- Loads `products.jsonl` + `.npy` once at import; `reload()` for the admin re-index command.
- **`reload_if_changed() -> bool` (#18):** `reload()` (and the import) records `_loaded_sig` = per file `(path, (mtime_ns, size))` from `os.stat` (`None` when missing) of `products.jsonl`, `embeddings.npy`, `embeddings_ids.json` and `embeddings_meta.json`, taken before reading. `find_products` (so also `latest_posts`) and `semantic_search` first stat those same files again and compare: same → nothing is read; different → `reload()`, so the first customer search after the nightly run sees tonight's posts without a restart. A reload that fails (e.g. a half-appended line) logs a WARNING and keeps the old catalog; the next call tries again.
- **Model guard (#23.5):** `_load_matrix()` uses the matrix only when `embeddings_meta.json` exists with `model == config.EMBED_MODEL`, `dim == config.EMBED_DIM` and the matrix is `N × EMBED_DIM`. Otherwise (e.g. an old 1024-d matrix without meta) it logs one ERROR naming the stored model / dimension and semantic search is off until re-index; filters keep working.
- **Fail fast on the query path:** if the query embedding fails (429, 5xx, timeout, invalid or missing key) `semantic_search` returns `[]` with one WARNING and never raises; the agent carries on as on any empty search (filters or `ask_owner`).
- CLI: `python -m shop_assistant.search "krosovka 42"` prints both filter and semantic results (FR-24).

### 3.5a `faq.py` — FR-22 (#23.7)
- `load()`, `add(question, answer, post_ids=(), ts=None)`, `search(query, k=3)`, `reindex()`; files §2.4, paths read from `config` at call time.
- `add` appends the line **first**, then embeds all pending questions in one `index.embed(..., kind="document", retry=False)` call and rewrites `.npy` + meta. Any embedding error is logged, never raised (it runs inside the bot, no sleeping retries); the line stays saved and is embedded later.
- `search` first tries to embed pending entries (failure → go on with the existing rows), then embeds `normalise(query)` with `kind="query"` and returns entries with cosine ≥ `faq.THRESHOLD`, best first, at most `k`. Query embedding fails → `[]` + one WARNING. No rows → `[]` without a request. Model / dimension mismatch → `[]` + one ERROR, no request.
- `THRESHOLD` measured with `gemini-embedding-001` on owner-style questions vs paraphrases (uz Latin / ru) and unrelated product questions — numbers next to the constant.
- CLI: `python -m shop_assistant.faq` = `reindex()` (all entries, retries allowed), the fix after an embedding-model change. `search.search_faq` / `index.reindex_faq` stay as unused stubs.

### 3.6 `tools.py` — the agent's interface
Thin wrappers around §3.5, declared with the plain `tools.tool` decorator (no SDK dependency): each `TOOLS` entry has `name`, `description` and `input_schema` built from the function signature and its docstring (`Args:` section), plus `call(args) -> str`. `llm.tool_declarations` turns them into Gemini function declarations. They return compact text (one line per product: `name · price · sizes · date · link · [eskirgan]`), plus:
- **Paging (#25).** `find_products_tool`, `semantic_search_tool` and `latest_posts_tool` take an optional `offset: int = 0` (None / negative → 0, a float from Gemini → int). A page is `config.MAX_RESULTS` items (`latest_posts_tool`: its clamped `n`), numbered from `offset + 1` (`format_products(page, start=)`), then the offer line when a shown item has no price, then ONE last range/total line: `ko'rsatildi 6–10, jami 23; keyingilari: shu filtrlar bilan offset=10` while more remain, `ko'rsatildi 21–23, jami 23; boshqa yo'q` on the last page. No match at all → `"no results"`; matches but `offset ≥ total` → `boshqa natija yo'q (jami 23, offset=30)`. Totals: `find_products_tool` = every match (`search.find_products(limit=None)`), `latest_posts_tool` = every sellable product (`search.latest_posts(None)`), `semantic_search_tool` = one `search.semantic_search(text, max_price, limit=20)` call paged (total ≤ 20).
- `ask_owner(question: str, post_ids: list[int]) -> str` — calls `bot.escalate(...)` through `run_coroutine_threadsafe` (same thread-bridge as `tgclient.run`). Returns `"forwarded"`.
- `search_faq_tool(text) -> str` — `faq.search(text)`; one line per hit `<question> — <owner_said>: <answer>` (`owner_said` = `lang.TEXTS[l]["owner_said"]` for the chat's stored language, e.g. "egasi aytgan"), never the product format (no numbering, no links); no hits → `"no faq entries"`.

### 3.7 `agent.py` — FR-13…FR-18, FR-20
- `run_agent(chat_id, text, history=None, lang="uz_latn") -> str`, blocking, run in a worker thread.
- Per-customer history: `dict[chat_id, list[message]]`, last 10 turns, in memory only (FR-17, NFR-4).
- System prompt (rules, kept short):
  1. One explicit line for the chat's language (`run_agent(..., lang=)`, from `lang.resolve`): "Answer in Russian" / "Answer in Uzbek, Latin script" / "Answer in Uzbek, Cyrillic script"; the fixed phrases the model copies (ask-price, offer line, stale note) come from `lang.TEXTS[lang]`.
  2. First `find_products`; if empty and the question has a descriptive part, `semantic_search`. For delivery/payment/other shop questions, `search_faq`.
  3. Never state price, size or availability not in tool output. Never guess stock.
  4. Escalate with `ask_owner` when: stock/availability asked, nothing relevant found, or question is outside the catalog.
  4a. **search_faq_tool before ask_owner** (#23.7) for delivery / payment / address / hours / other shop questions: an FAQ hit is answered as what the owner said (the `owner_said` label in the chat's language), never as catalog data, and is not escalated; only "no faq entries" leads to `ask_owner`.
  5. Max 5 products per reply; keep the tool's numbering (6., 7., … on a later page). Always include links. Mark stale posts with the "may be sold out" note. The offer line is copied when the tool result has it (just before the range line).
  6. **Paging (#25).** The tool's last line gives range and total. On a request for more (boshqalari, yana, boshqa bormi, ещё, другие) or a yes (ha, xa, да) to an offer of more, call the same tool with the same filters and the `offset` from that line. Never say the catalog has only the shown items unless the total says so; when the total is exhausted say so honestly and offer other filters.
- Manual function-calling loop over `llm.client().models.generate_content` (automatic function calling **off**): each model `function_call` runs the matching `TOOLS` entry, the model turn and the `function_response` are appended, repeat until the model answers with text or `config.MAX_ITERATIONS = 8` requests were made (then the polite apology). Tool declarations come from `TOOLS` via `llm.tool_declarations` (single source of truth). History keeps the `{"role": "user"|"assistant", "content": str}` format; assistant turns are sent to Gemini as role `model`. Timeout `llm.AGENT_TIMEOUT_S` (15 s) per request.
- `agent.last_run` = `{"tools": [{name, input, n_results}], "escalated", "usd": 0.0, "llm_calls"}` for `bot.py` / eval. For the three listing tools `n_results` counts only the numbered `N. ` lines (not the offer or range line).
- Errors (429 quota, 5xx, timeout, invalid/missing key): one ERROR log line, the customer gets `APOLOGY`, the failed turn is not stored. No paid fallback when the daily quota runs out (SRS NFR-2).

### 3.8 `bot.py` — FR-19, FR-21, FR-22, FR-26, C-5
- One Telethon bot client. Handlers:
  - `NewMessage(incoming, is_private, sender != owner)` → `asyncio.to_thread(run_agent, chat_id, text)` → reply. Groups are ignored (C-5).
  - **Carousel reply (S4, #19).** When the agent's text names posts, the customer gets ONE message: item 1's photo/video (the channel logo when the post has none), a card caption (`N. name` / `Narxi:` / `O'lcham:` / `Sana:` / stale warning) and inline buttons `◀ · N/n · ▶`, `Narxini so'rash` (only when the item has no price → `escalate` with that post id), `Kanalda ko'rish`. `CallbackQuery` edits the message in place; callback data `c:<idx>:<ids>` is self-contained (≤ 64 bytes) so navigation survives restarts. Text-only fallback (no link preview) when the send fails. The agent's numbered text still carries every link (FR-11/FR-14, eval parses ids) — it is the caption source, not shown as a list. On a later page (#25) the caption takes the idx-th numbered line by position and keeps its own number, and the counter starts at the first item's number (`6/10` … `10/10`); callback data stays positional.
  - **Customer language (#24, FR-13a).** Before each customer message `lang.resolve(chat_id, text, sender.lang_code)`: a clear detection from the text wins and is stored; otherwise the stored language; on first contact Telegram's `lang_code` (`ru` → ru, else uz_latn). `/start` is answered with `lang.TEXTS[l]["greeting"]` and no LLM call. Carousel labels, buttons, stale note, escalation/error replies and callback toasts use `lang.TEXTS` for the chat's language (callbacks read the store). Owner-facing texts (`#esc`, hints) stay Uzbek.
  - `escalate(customer_id, question, post_ids)` → message to `TG_OWNER_ID`: `"#esc <customer_id>\n<question>\n<links>"`. Tells the customer "Egasi tez orada javob beradi".
  - `NewMessage` from owner **that is a reply** to an `#esc` message → parse `customer_id` from the quoted text → forward owner's text to the customer → `faq.add(question, answer, post_ids)` (#23.7; `question` = the quoted `#esc` text without its header and link lines, `post_ids` = `post_ids_in(quoted)`, run in a worker thread). A `faq.add` failure is logged and never blocks the relay.
  - `/stats` from owner only → counts from `state.json` and today's `log.jsonl`.
- Logs every turn to `log.jsonl` (FR-25).

### 3.8a `lang.py` — FR-13, FR-13a
- `detect(text)` → `uz_latn` / `uz_cyrl` / `ru` / None: letter and word rules, no LLM (Uzbek Cyrillic letters/words first, then Russian letters/words, then Latin Uzbek markers; needs ≥ 2 words or 8 letters; the dominant script wins so one foreign word does not flip). Cyrillic words are matched through `textnorm`'s transliteration table.
- `LanguageStore` over `data/languages.json` (§2.7), `get_store()`, `resolve()`, `from_lang_code()`.
- `TEXTS[lang][key]`: every customer-facing fixed text (greeting, buttons, stale note, escalation/error replies, callback toasts, ask-price phrase, offer line, card labels).

### 3.9 `config.py`
`CHANNEL` (from env `TG_CHANNEL`, the shop channel username without @), `STALE_DAYS = 60`, `MAX_RESULTS = 5`, `CATEGORIES = [...]`, `FETCH_LIMIT = 500`, model names (`GEMINI_MODEL` for the agent, `GEMINI_EXTRACT_MODEL` for extraction — free-tier Flash models; measured limits written next to them; `EMBED_MODEL = "gemini-embedding-001"`, `EMBED_DIM = 768`, `EMBED_BATCH = 100`), `MAX_ITERATIONS = 8`, paths (incl. `EMBEDDINGS_META_PATH`, `FAQ_PATH`, `FAQ_EMBEDDINGS_PATH`, `FAQ_META_PATH`, `LANGUAGES_PATH = DATA_DIR / "languages.json"`). From env: `TG_CHANNEL`, `TG_API_ID`, `TG_API_HASH`, `TG_BOT_TOKEN`, `TG_OWNER_ID`, `GEMINI_API_KEY` (read lazily with `secret()`). `llm.py` holds the shared Gemini client, `tool_declarations()` and the timeouts.

### 3.10 `main.py`
Loads `.env`, starts the bot, runs forever. Ingestion is **not** in the service (D-7): it is `ingest.py` on its own timer (§3.11); the bot notices the new files by itself (`search.reload_if_changed`). `/reindex` (`search.reload()`) stays for a manual re-index.

### 3.11 `ingest.py` — FR-7, FR-23 (#18)
- `run() -> bool`: fetch (`fetch.main()`: `min_id = state.last_post_id`, new posts appended) → extract pending (`extract.main()`, whole batches) → `index.update_index()` → `state.last_index_at = now` (ISO, other keys kept). Stops at the first failing step (Telegram error, Gemini 429 / 5xx after retries, `ModelMismatch`, …), logs an ERROR and returns `False`; never raises. Files the failed step would have written stay unchanged; fetched posts are kept and are simply pending for the next night.
- Every Gemini request of the run (retries and failed attempts included) appends a line to `data/gemini_ingest.jsonl` (§2.6a) through a request hook that `run()` sets on `extract` and `index` and clears afterwards, so customer query embeddings are never logged there.
- Budget: 5 new posts = 1 extraction request (≤ `EXTRACT_BATCH`) + 1 embedding request (≤ `EMBED_BATCH`).
- `main() -> int`: 0 on success, 1 on failure (systemd marks the night failed). CLI: `python -m shop_assistant.ingest`.
- Deploy: `deploy/shop-assistant-ingest.service` (`Type=oneshot`, runs the CLI from `%h/shop_assistant`, never restarts the bot) + `deploy/shop-assistant-ingest.timer` (`OnCalendar=*-*-* 03:00:00` server time = Asia/Tashkent, `Persistent=true` so a night missed while the server was off runs at boot); `deploy.sh` installs both and enables the timer.

## 4. Key Design Decisions

| # | Decision | Alternative rejected | Reason |
|---|---|---|---|
| D-1 | Filters first, embeddings as fallback | embeddings only | numbers (size 42, ≤ 200k) embed badly; also the teaching point of the project |
| D-2 | Structured extraction with forced tool-use JSON | regex on captions | template drifts; the LLM handles "980.000ming", "Telegram obunachilariga narx", missing lines |
| D-3 | Normalise scripts at index *and* query time — for keywords **and** for the text that gets embedded | fuzzy matching at query time; raw-text embeddings | one cheap deterministic function, testable in isolation; spike #3: `bge-m3` cross-script similarity 0.62 raw → 0.87 normalised; still applies on Gemini (#23.5): `gemini-embedding-001` 768-d, spelling variants of krossovka 0.91–0.96 raw → 0.96–1.00 normalised |
| D-4 | Rewrite the whole embedding matrix on index | patch rows | N is small; correctness over cleverness |
| D-5 | Owner replies via Telegram "reply to" the escalation message | inline buttons / commands | zero UI to build; the quoted `#esc <id>` header carries the routing |
| D-6 | Conversation history in memory only | persist per customer | NFR-4 privacy; restart loses only the current chat context |
| D-8 | Customer replies are a one-message carousel with inline buttons (2026-09-21) | forward the channel posts / send an album / plain text list | forwards and albums made customers scroll through five screens; an album cannot carry buttons; Telegram's native "Show as carousel" is app-only until a Bot API layer exposes it |
| D-9 | `boshqa` records stay in `products.jsonl` but are never returned by search (`search.is_sellable`) | drop them at extraction | ids stay stable for the eval; announcements are visible for debugging |
| D-7 | Ingestion outside the bot process: a oneshot user unit on a nightly systemd **timer** (#18), not a thread or schedule inside the bot | live `on_channel_post` handler; scheduling in the bot | keeps the service simple; a failed night never touches the bot; the bot picks up the new files with `search.reload_if_changed` (no restart); live updates stay a v2 item |

## 5. Module Layout

```
shop_assistant/                   # repo root; run everything from here
  docs/shop_assistant_SRS.md, shop_assistant_SDD.md
  shop_assistant/                 # the package: `python -m shop_assistant.fetch`
    config.py  models.py  textnorm.py  fetch.py  extract.py  index.py  search.py
    llm.py     tools.py   agent.py    bot.py      main.py    lang.py    faq.py    ingest.py
  eval/questions.jsonl        # 20 questions, expected: {"posts":[ids]} or {"escalate":true}
  eval/run_eval.py            # runs agent offline (ask_owner stubbed), prints AC-2..AC-4
  tests/                      # only tests for merged work; a ticket's tests live on its branch until merged
  .github/                    # CI (pytest + tests/ untouched check), CODEOWNERS, PR template
  data/  session/             # gitignored
  requirements.txt  .env.example  deploy.sh  shop-assistant.service
  deploy/shop-assistant-ingest.service, shop-assistant-ingest.timer   # nightly ingestion (D-7, #18)
```

## 6. Traceability

| Requirement | Component |
|---|---|
| FR-1, 2, 3, 7 | fetch.py |
| FR-3a, 3b, 4 | extract.py |
| FR-5, 6 | index.py |
| FR-8, 9, 10, 11, 12 | search.py, textnorm.py |
| FR-13–18, 20 | agent.py (system prompt + history) |
| FR-13, 13a | lang.py (detection, per-chat store, fixed texts), bot.py, agent.py (answer-in line) |
| FR-19, 21, 22 | bot.py `escalate` + owner-reply handler, tools.ask_owner; faq.py + tools.search_faq_tool (FR-22) |
| FR-23, 24 | CLIs of fetch/extract/index/search; nightly `ingest.py` on a systemd timer |
| FR-25, 26 | bot.py logging, `/stats` |
| NFR-1, 2 | Gemini free-tier Flash model, max_iterations=8, compact tool output |
| NFR-3 | extract batching (`EXTRACT_BATCH`/call), embed batching (128) |
| NFR-4 | in-memory history, log stores question text only; `languages.json` holds only chat id → language code |
| NFR-5 | systemd `Restart=always` |
| NFR-6 | `.env`, `data/` and `session/` gitignored |
| NFR-7 | one module per stage, each with `__main__` |
| AC-1–4 | eval/ |
| AC-5 | manual test with owner account |
| AC-6 | restart service, confirm `search.py` loads from disk without network |

## 7. Risks
- Embeddings and the matrix must come from the same model: deploying new code without re-indexing (or the reverse) turns semantic search off (load guard, §3.5) → deploy code and the rebuilt `data/embeddings*` together.
- Embedding free quota (measured 2026-09-23): 100 embed requests/minute per model, and every text in a batch counts as one → a 134-product re-index needs ~2 minutes of retries, and while it runs customer query embeddings can hit 429 (semantic search then returns nothing for that turn; filters still work).
- The free-tier quota is per project per day and shared by customers, extraction, eval and development → the bot answers "try again later" when it runs out; watch `log.jsonl` for 429s; use a separate AI Studio key for development.
- On the free tier Google may use prompts to improve its products (SRS C-2 privacy trade-off).
- A stronger hosted model may be more eager to fill in numbers → watch the eval's `invented` count, not only `correct`.
- Category list too narrow → `boshqa` bucket; review after first extract run.
- Customer sends a photo/voice only → agent gets `<media>`; reply asking for text (v1), photo search is out of scope.
- Owner forgets to *reply* to the `#esc` message → bot answers the owner with a hint.
- FAQ search by cosine cannot tell "dastavka Toshkentga qancha?" from the stored Samarqand question (0.83, above `faq.THRESHOLD = 0.75`) → the tool line carries the stored question and the prompt uses an answer only when it fits; watch escalations vs FAQ answers in `log.jsonl`.

## 8. Change Log
| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-09-16 | Initial design against SRS v0.3 |
| 0.2 | 2026-09-16 | §5: code lives in a `shop_assistant/` package (so `python -m shop_assistant.x` works from repo root); `models.py` holds Post/Product/FaqEntry; scaffolding for tickets 1–15 |
| 0.3 | 2026-09-17 | §5: tests per ticket live on the ticket branch (senior-written), `main` keeps only merged tests; CI added |
| 0.5 | 2026-09-17 | Spike #3 results: embed normalised text (§3.3, §3.5, D-3); `max_tokens ≥ 1000` for extraction (§3.2) |
| 0.6 | 2026-09-21 | S4: §3.8 carousel reply (D-8); §3.2 product names = type + brand (`product_name` guard) and announcements → `boshqa` (`is_announcement`); §3.5 search excludes `boshqa` (D-9); tools number results, `narxi: so'rab beraman` + offer line (#21) |
| 0.8 | 2026-09-23 | SRS v0.5 C-2 (#23): no local models — agent and extraction on a Gemini free-tier model via `google-genai`; new `llm.py` (client, tool-schema conversion, timeouts); §1.1, §3.2, §3.7, §3.9, §7 updated; embeddings stay on Ollama until #23.5 |
| 0.9 | 2026-09-23 | #23 live results: two models — `GEMINI_MODEL = gemini-3.5-flash-lite` for the agent (15 req/min free, 0.8 s, 10/10 tool calls; flash models allow only 5 req/min), `GEMINI_EXTRACT_MODEL = gemini-3.5-flash` for extraction (fewer mis-categorised items than lite); `EXTRACT_BATCH = 20` (5/10/20 all valid); price regex needs one separator per number ("630.000 399.000" is two prices); eval 19/20, 0 invented, median 2.5 s |
| 0.7 | 2026-09-23 | #17: owner `/stats` (indexed posts, last index, questions/escalations today, `Gemini: N / GEMINI_DAILY_LIMIT today` from `log.jsonl` + `gemini_ingest.jsonl`) and `/reindex` (`search.reload()`); §2.6 `log.jsonl` gains `llm_calls`; `run()` dispatches through `bot.route` |
| 1.0 | 2026-09-23 | #23.5: embeddings on the Gemini API (`gemini-embedding-001`, 768-d, `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types, L2-normalised, batches ≤ 100); `embeddings_meta.json` + load guard against mixed models; query path fails fast (empty + WARNING), index CLI retries; tools are plain definitions (no Anthropic SDK); Ollama and `bge-m3` gone; §1, §1.1, §2.3, §3.3, §3.5, §3.6, §3.9, D-3, §7 updated |
| 1.1 | 2026-09-23 | SRS v0.5 FR-13a (#24): new `lang.py` (§3.8a: language detection, per-chat store, `TEXTS`), `data/languages.json` (§2.7), `config.LANGUAGES_PATH`; §3.7 explicit "Answer in …" line + `run_agent(lang=)`; §3.8 `/start` greeting without LLM, localized carousel/escalation/error texts |
| 1.2 | 2026-09-23 | R2 FAQ store (#23.7, FR-22): new `faq.py` (§3.5a) with `faq.jsonl` + `faq_embeddings.npy` + `faq_embeddings_meta.json` (§2.4, `config.FAQ_META_PATH`), model guard as for products, measured `faq.THRESHOLD`; §3.6 `search_faq_tool` uses `faq.search` and labels answers with `lang.TEXTS[l]["owner_said"]`; §3.7 rule "search_faq_tool before ask_owner"; §3.8 the owner's relayed answer is saved with `faq.add` (failure never blocks the relay) |
| 0.4 | 2026-09-17 | SRS C-2 v0.4: Claude + Voyage replaced by Ollama on the GPU server (`gemma4:31b`, `bge-m3`); §1.1, §3.2, §3.3, §3.7, §3.9, §7 updated; deploy target = the GPU server |
| 1.3 | 2026-09-23 | #18: nightly ingestion — new `ingest.py` (§3.11: fetch → extract pending → `index.update_index()` → `state.last_index_at`, `False`/exit 1 on a failed step) on a oneshot unit + `deploy/shop-assistant-ingest.timer` (03:00, `Persistent=true`); §3.3 incremental `update_index()` + `ModelMismatch`; §3.5 `search.reload_if_changed()` (os.stat signature, no bot restart); §2.6a `gemini_ingest.jsonl`; §3.2 `extract.main()` returns success; D-7 timer, not the bot |
| 1.4 | 2026-09-23 | #25: listing tools page with `offset` and end with a range/total line (`ko'rsatildi 6–10, jami 23; keyingilari: … offset=10` / `; boshqa yo'q`; past the end `boshqa natija yo'q (jami N, …)`); §3.6 paging, §3.7 rule 6 (paging, never "only these") and rule 5 without "ask to narrow down", `n_results` counts item lines; §3.8 carousel numbering/counter on later pages |
