"""Tool Runner agent (Anthropic SDK → Ollama) + per-customer history. SDD §3.7, FR-13…18/20. Ticket #10."""
import contextvars
import json

import anthropic

from shop_assistant import config
from shop_assistant.tools import NO_FAQ, NO_RESULTS, TOOLS

SYSTEM_PROMPT = """You are the assistant of the Telegram shop @status_dokon. Rules:
1. Reply in the customer's language AND script: Uzbek Latin, Uzbek Cyrillic or Russian, exactly as they wrote.
2. If the question names a product type, size, price or color, call find_products_tool first; if it returns "no results", call semantic_search_tool. If the question only describes an occasion, season or feeling without naming a product type (e.g. "kuzda kiyishga mos narsa", "to'yga chiroyli narsa", "что-нибудь для холодной погоды"), call semantic_search_tool directly. For "what's new" questions (yangi, янги, новое, новинки) call latest_posts_tool. For delivery, payment, address, hours or other shop questions call search_faq_tool.
3. Never state a price, size, color or availability that is not in a tool result. Never guess stock.
4. Call ask_owner when: the customer asks whether an item is STILL available / in stock ("hali bormi", "ҳали борми", "ещё есть", "в наличии", "qolganmi") — availability is never in the catalog, so escalate even if search finds the product (you may still show it). A plain "bormi?" / "борми?" / "есть?" is an ordinary search question: answer from search results, do not escalate; both searches found nothing relevant; or the question is about orders, delivery, payment or anything outside the catalog. After ask_owner, tell the customer the owner will reply soon.
5. Show at most 5 products per reply; if there are more, ask the customer to narrow down. Always include each product's link. For products marked [eskirgan] add a note in the customer's language that it may be sold out and should be confirmed with the owner.
6. If the message is "<media>" (a photo/voice without text), do not call tools: ask the customer to write the product name in text.
Be short and friendly; no markdown tables."""

APOLOGY = "Kechirasiz, texnik xatolik. Birozdan keyin qayta urinib ko'ring."

client = anthropic.Anthropic()

current_chat_id: contextvars.ContextVar[int] = contextvars.ContextVar("current_chat_id", default=0)

last_run: dict = {"tools": [], "escalated": False, "usd": 0.0}


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


def _text_of(message) -> str:
    """Concatenate text blocks; skips thinking / tool_use blocks (gemma4 may emit `thinking`)."""
    return "\n".join(b.text for b in message.content if getattr(b, "type", None) == "text").strip()


def run_agent(chat_id: int, text: str, history: History | None = None) -> str:
    """Blocking. tool_runner with TOOLS, max_iterations=8, max_tokens=1024. Returns final reply text."""
    global last_run
    h = history or _history
    current_chat_id.set(chat_id)
    run: dict = {"tools": [], "escalated": False, "usd": 0.0}
    h.append(chat_id, "user", text)
    try:
        runner = client.beta.messages.tool_runner(
            model=config.MODEL,
            max_tokens=config.MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=list(h.get(chat_id)),
            max_iterations=config.MAX_ITERATIONS,
            thinking=config.THINKING,
        )
        last_message = None
        for message in runner:
            last_message = message
            tool_uses = [b for b in message.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                continue
            response = runner.generate_tool_call_response() or {"content": []}
            results = {r["tool_use_id"]: r.get("content", "") for r in response["content"]}
            for tu in tool_uses:
                print(f"  · {tu.name}({json.dumps(tu.input, ensure_ascii=False)})")
                out = results.get(tu.id, "")
                out = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
                n = 0 if out.strip() in (NO_RESULTS, NO_FAQ) else sum(1 for ln in out.splitlines() if ln.strip())
                run["tools"].append({"name": tu.name, "input": dict(tu.input), "n_results": n})
                if tu.name == "ask_owner":
                    run["escalated"] = True
    except anthropic.APIError as e:
        print(f"  ! APIError: {e}")
        last_run = run
        msgs = h.get(chat_id)
        if msgs and msgs[-1] == {"role": "user", "content": text}:
            msgs.pop()          # keep history consistent; the apology is not stored either
        return APOLOGY
    last_run = run
    reply = _text_of(last_message) if last_message is not None else ""
    h.append(chat_id, "assistant", reply or APOLOGY)
    return reply or APOLOGY


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
