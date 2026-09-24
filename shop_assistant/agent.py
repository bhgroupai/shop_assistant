"""Manual Gemini function-calling loop (google-genai) + per-customer history. SDD §3.7, FR-13…18/20. Tickets #10, #23, #26."""
import contextvars
import json
import logging
import re
import time

import httpx
from google.genai import errors, types

from shop_assistant import config, llm, tools
from shop_assistant.lang import DEFAULT as DEFAULT_LANG, TEXTS
from shop_assistant.tools import NO_FAQ, NO_RESULTS

log = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are the assistant of a clothing shop whose catalog is its Telegram channel. Rules:
1. {answer_line}.
2. If the question names a product type, size, price or color, call find_products_tool first; if it returns "no results", call semantic_search_tool. If the question only describes an occasion, season or feeling without naming a product type (e.g. "kuzda kiyishga mos narsa", "to'yga chiroyli narsa", "что-нибудь для холодной погоды"), call semantic_search_tool directly. For "what's new" questions (yangi, янги, новое, новинки) call latest_posts_tool. For delivery, payment, address, hours or other shop questions call search_faq_tool.
3. Never state a price, size, color or availability that is not in a tool result. Never guess stock.
4. Call ask_owner when: the customer asks whether an item is STILL available / in stock ("hali bormi", "ҳали борми", "ещё есть", "в наличии", "qolganmi") — availability is never in the catalog, so escalate even if search finds the product (you may still show it). A plain "bormi?" / "борми?" / "есть?" is an ordinary search question: answer from search results, do not escalate; both searches found nothing relevant; or the question is about orders, delivery, payment or anything outside the catalog and search_faq_tool had no answer (rule 8). After ask_owner, tell the customer the owner will reply soon.
5. Show at most 5 products per reply. Keep the tool's numbering (1., 2., ... or 6., 7., ... on a later page) and always include each product's link. Items whose price is "{tool_ask_price}" in the tool result have no known price: write "{ask_price}" as their price, never send the customer elsewhere to ask; end your reply with the one line "{offer_price}" only when the tool result has the line "{tool_offer_price}" (just before its last "ko'rsatildi ..." line), i.e. some shown item has no price; otherwise omit it. For products marked [eskirgan] keep the [eskirgan] tag on the item's line and add the note "{stale_note}".
6. If the message is "<media>" (a photo/voice without text), do not call tools: ask the customer to write the product name in text.
7. A message that is just a number, or "narxi N" / "N-chisi" / "N-си" / "цена N", refers to item N of your previous numbered reply. For a price question call ask_owner(question, post_ids=[that item's id from its t.me link]) and confirm the owner will reply with the price. For other follow-ups about item N (e.g. "2-chisi 43 bormi?") call find_products_tool with that item's name as keywords plus the asked filter (size/color/price).
8. For delivery, payment, address, hours, orders or other shop questions always call search_faq_tool before ask_owner. If it returns an answer that fits the question, give it as what the owner said ("{owner_said}: ..."), never as catalog data, and do not call ask_owner; call ask_owner only when it returns "no faq entries" or none of its answers fits.
9. Paging: the last line of a product tool result gives the shown range and the total, e.g. "ko'rsatildi 1–5, jami 23; keyingilari: shu filtrlar bilan offset=5". When more remain, tell the customer how many there are in total and offer to show the next ones. When the customer asks for more (boshqalari, yana, boshqa bormi, ещё, другие) or says yes (ha, xa, да) to your offer of more, call the SAME tool with the SAME filters as your previous search and offset = the number from that line; never repeat offset 0. Never say the catalog has only the shown items unless the tool's total says so; when the total is exhausted ("boshqa yo'q" / "boshqa natija yo'q"), say honestly that all of them have been shown and offer other filters.
10. The "ko'rsatildi ..." / "boshqa natija yo'q (...)" line of a tool result is for you only: never copy or show it (or its offset) to the customer; instead tell the customer the total in their language (e.g. "Jami 23 ta topildi", "Всего найдено 23").
Be short and friendly; no markdown tables."""

ANSWER_LINES = {
    "uz_latn": "Answer in Uzbek, Latin script",
    "uz_cyrl": "Answer in Uzbek, Cyrillic script",
    "ru": "Answer in Russian",
}


def system_prompt(lang: str = DEFAULT_LANG) -> str:
    """SYSTEM_PROMPT_TEMPLATE with one explicit answer-in line and the fixed phrases in `lang` (#24)."""
    lang = lang if lang in TEXTS else DEFAULT_LANG
    t, uz = TEXTS[lang], TEXTS["uz_latn"]
    return SYSTEM_PROMPT_TEMPLATE.format(
        answer_line=ANSWER_LINES[lang], ask_price=t["ask_price"], offer_price=t["offer_price"],
        stale_note=t["stale_note"], owner_said=t["owner_said"], tool_ask_price=uz["ask_price"], tool_offer_price=uz["offer_price"])


SYSTEM_PROMPT = system_prompt(DEFAULT_LANG)

_LISTING_TOOLS = {"find_products_tool", "semantic_search_tool", "latest_posts_tool"}
_ITEM_LINE = re.compile(r"^\d+\. ")


def _n_results(name: str, out: str) -> int:
    """How many results a tool returned: numbered product lines for the listing tools (not the offer or
    range line, #25); non-empty lines for the others; 0 for "no results" / "no faq entries"."""
    if out.strip() in (NO_RESULTS, NO_FAQ):
        return 0
    if name in _LISTING_TOOLS:
        return sum(1 for ln in out.splitlines() if _ITEM_LINE.match(ln))
    return sum(1 for ln in out.splitlines() if ln.strip())

APOLOGY = "Kechirasiz, texnik xatolik. Birozdan keyin qayta urinib ko'ring."

current_chat_id: contextvars.ContextVar[int] = contextvars.ContextVar("current_chat_id", default=0)

last_run: dict = {"tools": [], "escalated": False, "usd": 0.0, "llm_calls": 0}

# 429 quota / other API errors, 5xx, timeouts and transport errors, missing key (RuntimeError from llm.client()).
_API_ERRORS = (errors.APIError, httpx.HTTPError, RuntimeError)

# Ticket #26: transient failures worth one retry (same model, same contents). 429 / 400 are not.
_RETRY_CODES = frozenset({500, 502, 503, 504})
RETRY_PAUSE_S = 0.7


def _is_transient(e: Exception) -> bool:
    """True for a 500/502/503/504 server error or a client-side timeout (httpx / SDK)."""
    if isinstance(e, httpx.TimeoutException):
        return True
    return isinstance(e, errors.APIError) and getattr(e, "code", None) in _RETRY_CODES


class History:
    """Per-customer message history, last config.HISTORY_TURNS turns, in memory only (FR-17, NFR-4)."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._chats: dict[int, list[dict]] = {}

    def get(self, chat_id: int) -> list[dict]:
        return self._chats.get(chat_id, [])

    def append(self, chat_id: int, role: str, content) -> None:
        """Append and trim so at most max_turns user/assistant pairs remain."""
        msgs = self._chats.setdefault(chat_id, [])
        msgs.append({"role": role, "content": content})
        limit = 2 * self.max_turns
        if len(msgs) > limit:
            del msgs[: len(msgs) - limit]

    def clear(self, chat_id: int) -> None:
        self._chats.pop(chat_id, None)


_history = History(config.HISTORY_TURNS)


def _to_contents(messages: list[dict]) -> list[types.Content]:
    """Our history format ({"role": "user"|"assistant", "content": str}) -> Gemini contents."""
    out = []
    for m in messages:
        content = m["content"]
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        out.append(types.Content(role="model" if m["role"] == "assistant" else "user",
                                 parts=[types.Part(text=text)]))
    return out


def _request_config(lang: str = DEFAULT_LANG) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system_prompt(lang),
        tools=[types.Tool(function_declarations=llm.tool_declarations(tools.TOOLS))],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        # Client gives up after AGENT_TIMEOUT_S; the server deadline header must be >= 10 s (see llm.py).
        http_options=types.HttpOptions(timeout=llm.AGENT_TIMEOUT_S * 1000,
                                       headers={"X-Server-Timeout": str(llm.SERVER_DEADLINE_S)}),
    )


def _text_of(content) -> str:
    """Concatenate text parts; skips function calls and thought parts."""
    parts = (content.parts if content is not None else None) or []
    return "\n".join(p.text for p in parts if p.text and not p.thought).strip()


def _run_tool(name: str, args: dict) -> str:
    """Run the TOOLS entry called `name`; a failing tool becomes an error text the model can read."""
    tool = next((t for t in tools.TOOLS if t.name == name), None)
    if tool is None:
        return f"error: unknown tool {name}"
    try:
        out = tool.call(args)
    except Exception as e:                        # bad arguments or a tool bug: tell the model, keep going
        log.warning("tool %s failed: %s", name, e)
        return f"error: {e}"
    return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


def _generate(chat_id: int, contents: list, cfg, run: dict):
    """One model step: a generate_content request, retried ONCE on a transient failure (5xx / timeout)
    after RETRY_PAUSE_S, with the same model and contents (#26). Every attempt counts in run["llm_calls"].
    A second failure, or any non-transient error (429, 400, ...), propagates to run_agent's handler."""
    for attempt in (1, 2):
        run["llm_calls"] += 1
        try:
            return llm.client().models.generate_content(model=config.GEMINI_MODEL, contents=contents, config=cfg)
        except _API_ERRORS as e:
            if attempt == 2 or not _is_transient(e):
                raise
            code = getattr(e, "code", None)
            log.warning("chat %s: Gemini request failed (%s%s), retrying once in %.1f s", chat_id,
                        type(e).__name__, f" {code}" if code else "", RETRY_PAUSE_S)
            time.sleep(RETRY_PAUSE_S)


def run_agent(chat_id: int, text: str, history: History | None = None, lang: str = "uz_latn") -> str:
    """Blocking. Manual Gemini function-calling loop over TOOLS, max_iterations=8. Returns final reply text.
    Ticket #24: `lang` (uz_latn | uz_cyrl | ru, see lang.py) — the system prompt gets one explicit
    "Answer in ..." line and the fixed phrases the model copies in that language."""
    global last_run
    h = history or _history
    current_chat_id.set(chat_id)
    run: dict = {"tools": [], "escalated": False, "usd": 0.0, "llm_calls": 0}
    h.append(chat_id, "user", text)
    contents = _to_contents(h.get(chat_id))
    reply = ""
    # Uzbek Latin keeps the old APOLOGY; other languages get their error text (#24).
    apology = APOLOGY if lang == DEFAULT_LANG or lang not in TEXTS else TEXTS[lang]["error_reply"]
    try:
        cfg = _request_config(lang)
        while run["llm_calls"] < config.MAX_ITERATIONS:
            resp = _generate(chat_id, contents, cfg, run)
            content = resp.candidates[0].content if resp.candidates else None
            calls = [p.function_call for p in (content.parts if content is not None else None) or []
                     if p.function_call]
            if not calls:
                reply = _text_of(content)
                if not reply:
                    log.warning("chat %s: empty model response", chat_id)
                break
            if run["llm_calls"] >= config.MAX_ITERATIONS:
                # The model would never see these results; do not run them (ask_owner has side effects).
                log.warning("chat %s: stopped after %d model requests", chat_id, config.MAX_ITERATIONS)
                break
            contents.append(content)
            responses = []
            for fc in calls:
                args = dict(fc.args or {})
                print(f"  · {fc.name}({json.dumps(args, ensure_ascii=False)})")
                out = _run_tool(fc.name, args)
                n = _n_results(fc.name, out)
                run["tools"].append({"name": fc.name, "input": args, "n_results": n})
                if fc.name == "ask_owner":
                    run["escalated"] = True
                responses.append(types.Part.from_function_response(name=fc.name, response={"result": out}))
            contents.append(types.Content(role="user", parts=responses))
    except _API_ERRORS as e:
        code = getattr(e, "code", None)
        log.error("chat %s: Gemini request failed (%s%s): %s", chat_id, type(e).__name__,
                  f" {code}" if code else "", e)
        last_run = run
        msgs = h.get(chat_id)
        if msgs and msgs[-1] == {"role": "user", "content": text}:
            msgs.pop()          # keep history consistent; the apology is not stored either
        return apology
    last_run = run
    h.append(chat_id, "assistant", reply or apology)
    return reply or apology


if __name__ == "__main__":
    import sys
    if sys.argv[1:]:
        print(run_agent(0, " ".join(sys.argv[1:])))
    else:
        try:
            while (q := input("> ").strip()):
                print(run_agent(0, q))
        except (EOFError, KeyboardInterrupt):
            pass
