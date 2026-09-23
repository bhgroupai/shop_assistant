"""Customer language: detection, per-chat store, fixed customer-facing texts. SDD §3.8, FR-13a. Ticket #24.

Contract (tests/test_lang.py, tests/test_bot_lang.py):

- LANGS = ("uz_latn", "uz_cyrl", "ru"); DEFAULT = "uz_latn".
- detect(text) -> "uz_latn" | "uz_cyrl" | "ru" | None. Pure (no LLM, no network). Rules in order:
  Uzbek-only Cyrillic letters (ў қ ғ ҳ) or Uzbek Cyrillic words (борми, нархи, қанча, ...) → uz_cyrl;
  other Cyrillic with Russian-only letters (ы щ э ъ) or Russian words (есть, сколько, размер, что, ...) → ru;
  Latin with Uzbek markers (o'/g'/oʻ, sh/ch, bormi, narxi, qancha, ...) → uz_latn;
  too short / unclear (emoji, a number, a /command, "ok", "") → None.
  A decision needs at least 2 words or 8 letters of evidence, so one foreign word inside a sentence
  does not flip the language. Cyrillic handling reuses the tables in textnorm.py.
- from_lang_code(code) -> str: Telegram sender.lang_code on first contact: "ru…" → "ru", anything else
  (uz, en, None, "") → "uz_latn".
- LanguageStore(path): chat id → language code, persisted as a JSON object {"<chat_id>": "<lang>"}.
  .path; .get(chat_id) -> str | None; .set(chat_id, lang) (ValueError for a lang not in LANGS; saves
  only when the value changes; atomic write via a temp file in the same dir + rename; creates the parent
  dir). A missing, corrupt or non-object file → one WARNING on this module's logger and an empty store.
- get_store() -> LanguageStore for config.LANGUAGES_PATH read at call time; the same object while the path
  is unchanged, a fresh one (read from disk) when config.LANGUAGES_PATH changes.
- resolve(chat_id, text, lang_code=None, store=None) -> str: a clear detect(text) wins and is saved;
  otherwise the stored language; otherwise (first contact) from_lang_code(lang_code). store defaults to
  get_store().
- TEXTS: {lang: {key: str}} with identical keys for every lang in LANGS, all non-empty:
  greeting, ask_price_button, view_in_channel_button, stale_note, escalated_reply, callback_toast,
  callback_error, error_reply, ask_price, offer_price, price_label, sizes_label, date_label.
  Owner-facing texts (bot.OWNER_HINT_*, the #esc header) are not here and stay Uzbek.
"""
import json
import logging
import os
import re
import tempfile
from pathlib import Path

from shop_assistant import config
from shop_assistant.textnorm import normalise

log = logging.getLogger(__name__)

LANGS = ("uz_latn", "uz_cyrl", "ru")
DEFAULT = "uz_latn"

# ---------------------------------------------------------------- fixed customer-facing texts
# uz_latn keeps the wording the bot used before #24. ru and uz_cyrl need a native speaker's check.

TEXTS: dict[str, dict[str, str]] = {
    "uz_latn": {
        "greeting": "Assalomu alaykum! 👋 Men do'kon yordamchisiman. Qanday mahsulot qidiryapsiz? "
                    "Nomini, o'lchamini yoki narxini yozing.",
        "ask_price_button": "Narxini so'rash",
        "view_in_channel_button": "Kanalda ko'rish",
        "stale_note": "⚠️ Eskirgan bo'lishi mumkin — egadan tasdiqlang",
        "escalated_reply": "Egasi tez orada javob beradi 🙏",
        "callback_toast": "Egaga yuborildi, javobini shu yerga yozaman",
        "callback_error": "Xatolik",
        "error_reply": "Kechirasiz, xatolik yuz berdi. Birozdan keyin qayta urinib ko'ring.",
        "ask_price": "narxi: so'rab beraman",
        "offer_price": "Narxini bilmoqchi bo'lsangiz raqamini yozing",
        "price_label": "Narxi:",
        "sizes_label": "O'lcham:",
        "date_label": "Sana:",
    },
    "uz_cyrl": {
        "greeting": "Ассалому алайкум! 👋 Мен дўкон ёрдамчисиман. Қандай маҳсулот қидиряпсиз? "
                    "Номини, ўлчамини ёки нархини ёзинг.",
        "ask_price_button": "Нархини сўраш",
        "view_in_channel_button": "Каналда кўриш",
        "stale_note": "⚠️ Эскирган бўлиши мумкин — эгадан тасдиқланг",
        "escalated_reply": "Эгаси тез орада жавоб беради 🙏",
        "callback_toast": "Эгага юборилди, жавобини шу ерга ёзаман",
        "callback_error": "Хатолик",
        "error_reply": "Кечирасиз, хатолик юз берди. Бироздан кейин қайта уриниб кўринг.",
        "ask_price": "нархи: сўраб бераман",
        "offer_price": "Нархини билмоқчи бўлсангиз рақамини ёзинг",
        "price_label": "Нархи:",
        "sizes_label": "Ўлчам:",
        "date_label": "Сана:",
    },
    "ru": {
        "greeting": "Здравствуйте! 👋 Я помощник магазина. Что вы ищете? "
                    "Напишите название, размер или цену.",
        "ask_price_button": "Узнать цену",
        "view_in_channel_button": "Смотреть в канале",
        "stale_note": "⚠️ Возможно, уже продано — уточните у владельца",
        "escalated_reply": "Владелец скоро ответит 🙏",
        "callback_toast": "Отправлено владельцу, ответ пришлю сюда",
        "callback_error": "Ошибка",
        "error_reply": "Извините, произошла ошибка. Попробуйте ещё раз чуть позже.",
        "ask_price": "цена: уточню у владельца",
        "offer_price": "Чтобы узнать цену, напишите номер товара",
        "price_label": "Цена:",
        "sizes_label": "Размер:",
        "date_label": "Дата:",
    },
}

CURRENCY = {"uz_latn": "so'm", "uz_cyrl": "сўм", "ru": "сум"}   # after a formatted price on the card

# ---------------------------------------------------------------- detection

_UZ_CYRL_LETTERS = set("ўқғҳ")
_RU_LETTERS = set("ыщэъ")
_APOS = "ʻ'ʼ’‘`´"
# Uzbek marker words in normalised Latin form (textnorm.normalise). Latin input is matched directly,
# Cyrillic input after transliteration through textnorm's table (борми → bormi, нархи → narxi).
_UZ_WORDS = {
    "bormi", "bor", "yoq", "narxi", "narx", "narxini", "qancha", "kancha", "qanaqa", "qanday", "qaysi",
    "qayerda", "qaerda", "yangi", "hali", "xali", "nima", "kerak", "kerakmi", "salom", "assalomu",
    "alaykum", "rahmat", "raxmat", "bilan", "uchun", "emas", "shu", "bu", "necha", "nechpul", "narsa",
    "narsalar", "kiyim", "qora", "oq", "rangi", "qishki", "yozgi", "kuzgi", "bormikan", "boladimi",
    "olsam", "qilsam", "manzil", "manzilingiz", "dastavka", "yetkazib", "tolov", "karta",
}
# Russian marker words (Cyrillic, lowercase). Checked after the Uzbek markers.
_RU_WORDS = {
    "есть", "сколько", "размер", "размера", "размеры", "что", "какие", "какой", "какая", "нужна",
    "нужен", "нужно", "мне", "для", "это", "эта", "этот", "где", "как", "стоит", "цена", "можно",
    "ещё", "еще", "нового", "новое", "новинки", "привет", "здравствуйте", "спасибо", "доставка",
    "наличии", "или", "да", "нет", "у", "вас", "а", "в", "и", "на", "с", "по",
}
_RU_ENDINGS = ("ть", "ться", "ая", "ые", "ого", "его")
_RU_MI = ("ами", "ими", "ьми", "ыми", "ями")      # Russian instrumental plural, not the Uzbek "-ми?"
_TOKEN = re.compile(rf"[^\W\d_](?:[^\W\d_]|[{_APOS}](?=[^\W\d_]))*")
_LATIN_APOS_MARK = re.compile(rf"[og][{_APOS}]")


def _is_cyr(ch: str) -> bool:
    return "Ѐ" <= ch <= "ӿ"


def _uz_word(latin: str) -> bool:
    """A normalised (Latin) word that marks Uzbek: a known word or a typical Uzbek suffix."""
    return (latin in _UZ_WORDS or latin.endswith(("ning", "ingiz", "imiz", "dagi"))
            or (len(latin) >= 5 and latin.endswith("mi")))


def _uz_cyrl_marker(word: str) -> bool:
    if _UZ_CYRL_LETTERS & set(word):
        return True
    if word in _RU_WORDS or word.endswith(_RU_MI):
        return False
    return _uz_word(normalise(word))


def _ru_marker(word: str) -> bool:
    return bool(_RU_LETTERS & set(word)) or word in _RU_WORDS or word.endswith(_RU_ENDINGS)


def _uz_latn_marker(word: str) -> bool:
    return (bool(_LATIN_APOS_MARK.search(word)) or "sh" in word or "ch" in word
            or ("q" in word and "qu" not in word) or _uz_word(normalise(word)))


def detect(text: str) -> str | None:
    """Language of `text` or None when there is not enough evidence (see module docstring)."""
    # Commands, links and @mentions are not language evidence.
    chunks = [c for c in (text or "").lower().split()
              if not c.startswith(("/", "@", "http")) and "t.me/" not in c]
    words = _TOKEN.findall(" ".join(chunks))
    cyr = [w for w in words if any(_is_cyr(ch) for ch in w)]
    lat = [w for w in words if not any(_is_cyr(ch) for ch in w)
           and all(ch.isascii() or ch in _APOS for ch in w)]
    # The dominant script decides, so one foreign word inside a sentence does not flip the language.
    group = cyr if sum(map(len, cyr)) > sum(map(len, lat)) else lat
    letters = sum(ch.isalpha() for w in group for ch in w)
    if not group or (len(group) < 2 and letters < 8):
        return None
    if group is cyr:
        if any(_uz_cyrl_marker(w) for w in cyr):
            return "uz_cyrl"
        if any(_ru_marker(w) for w in cyr):
            return "ru"
        return None
    return "uz_latn" if any(_uz_latn_marker(w) for w in lat) else None


def from_lang_code(code: str | None) -> str:
    """Telegram lang_code → our language code: ru → ru, anything else → uz_latn."""
    return "ru" if (code or "").strip().lower().startswith("ru") else DEFAULT


# ---------------------------------------------------------------- per-chat store

class LanguageStore:
    """chat id → language code, persisted to a small JSON file (atomic write, save on change only)."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._data: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            log.warning("language store %s missing; starting empty", self.path)
            return {}
        except (OSError, ValueError) as e:
            log.warning("language store %s unreadable (%s); starting empty", self.path, e)
            return {}
        if not isinstance(raw, dict):
            log.warning("language store %s is not a JSON object; starting empty", self.path)
            return {}
        return {str(k): v for k, v in raw.items() if v in LANGS}

    def get(self, chat_id: int) -> str | None:
        return self._data.get(str(chat_id))

    def set(self, chat_id: int, lang: str) -> None:
        if lang not in LANGS:
            raise ValueError(f"unknown language {lang!r}; expected one of {LANGS}")
        key = str(chat_id)
        if self._data.get(key) == lang:
            return
        self._data[key] = lang
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


_store: LanguageStore | None = None


def get_store() -> LanguageStore:
    """The process-wide store for config.LANGUAGES_PATH (re-created when that path changes)."""
    global _store
    path = Path(config.LANGUAGES_PATH)
    if _store is None or _store.path != path:
        _store = LanguageStore(path)
    return _store


def resolve(chat_id: int, text: str, lang_code: str | None = None, store: LanguageStore | None = None) -> str:
    """Language to answer `chat_id` in: clear detection (saved) > stored > from_lang_code(lang_code)."""
    store = store or get_store()
    found = detect(text)
    if found:
        store.set(chat_id, found)
        return found
    return store.get(chat_id) or from_lang_code(lang_code)
