"""Ticket #24 — customer language: detection, first-contact mapping, per-chat store, fixed texts.
The contract is in the shop_assistant/lang.py docstring. Written by the senior — do not edit."""
import json
import logging
import runpy

import pytest

from shop_assistant import config, lang

CHAT = 4242

# Real customer phrasings: eval/questions.jsonl first, then typical messages of this shop's customers.
UZ_LATN = [
    "dvoyka narxi qancha?",
    "600 ming gacha sviter bormi?",
    "42 razmer krossovka bormi",
    "kuzda kiyishga mos narsa bormi",
    "sovuqda kiyiladigan issiq narsa bormi",
    "to'yga kiyadigan chiroyli narsa",
    "yangi narsalar bormi?",
    "Stefano Ricci kiyim hali bormi?",
    "dastavka Samarqandga qancha?",
    "karta bilan to'lasa bo'ladimi?",
    "zakaz qilsam bo'ladimi, qanday olsam bo'ladi?",
    "krossovka 42 bormi?",
    "Qishki kurtka qora rangi bormi",
    "Assalomu alaykum, shu kurtkaning narxi qancha?",
]
UZ_CYRL = [
    "барсофка борми?",
    "пойабзал 42 размер борми",
    "янги нарса борми?",
    "кроссовка борми",
    "двойка нархи қанча?",
    "қишки куртка борми?",
    "42 размерли туфли борми",
    "ҳали борми?",
    "бу кўйлакнинг нархи қанча",
    "Самарқандга доставка борми?",
    "қора куртка борми",
    "манзилингиз қаерда?",
]
RU = [
    "туфли 43 размер есть?",
    "джинсовка есть?",
    "куртка есть?",
    "что-нибудь для холодной погоды",
    "что нового?",
    "сколько стоит двойка?",
    "есть кроссовки 42 размера?",
    "какие размеры есть?",
    "эта куртка ещё есть?",
    "а доставка в Самарканд есть?",
    "мне нужна зимняя куртка, что есть?",
    "сколько стоит доставка?",
]
NO_DECISION = ["/start", "42", "ok", "👍", "", "   ", "Ok", "ha", "да", "/start@example_shop_bot", "Nike Air 42"]


# ---------------------------------------------------------------- detect

@pytest.mark.parametrize("text", UZ_LATN)
def test_detect_uz_latin(text):
    assert lang.detect(text) == "uz_latn"


@pytest.mark.parametrize("text", UZ_CYRL)
def test_detect_uz_cyrillic(text):
    assert lang.detect(text) == "uz_cyrl"


@pytest.mark.parametrize("text", RU)
def test_detect_russian(text):
    assert lang.detect(text) == "ru"


def test_enough_real_phrasings_per_language():
    assert min(len(UZ_LATN), len(UZ_CYRL), len(RU)) >= 10


@pytest.mark.parametrize("text", NO_DECISION)
def test_short_or_unclear_is_no_decision(text):
    assert lang.detect(text) is None


def test_uzbek_cyrillic_without_special_letters_is_not_russian():
    """The hard case from the ticket notes: no ў қ ғ ҳ, still Uzbek because of 'борми'."""
    assert lang.detect("кроссовка борми") == "uz_cyrl"


def test_one_russian_word_in_uzbek_sentence_does_not_flip():
    assert lang.detect("dvoyka bormi, сколько?") != "ru"
    assert lang.detect("krossovka 42 razmer bormi, есть?") != "ru"


def test_detect_is_case_insensitive():
    assert lang.detect("KROSSOVKA 42 BORMI?") == "uz_latn"
    assert lang.detect("СКОЛЬКО СТОИТ ДВОЙКА?") == "ru"


def test_detect_accepts_other_apostrophes():
    assert lang.detect("toʻyga kiyadigan chiroyli narsa") == "uz_latn"
    assert lang.detect("to‘yga kiyadigan chiroyli narsa") == "uz_latn"


# ---------------------------------------------------------------- first contact: Telegram lang_code

@pytest.mark.parametrize("code, expected", [("ru", "ru"), ("uz", "uz_latn"), ("en", "uz_latn"),
                                            ("tr", "uz_latn"), ("", "uz_latn"), (None, "uz_latn")])
def test_from_lang_code(code, expected):
    assert lang.from_lang_code(code) == expected


def test_constants():
    assert lang.LANGS == ("uz_latn", "uz_cyrl", "ru")
    assert lang.DEFAULT == "uz_latn"


# ---------------------------------------------------------------- store

@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "languages.json"


def test_store_path_in_config():
    ns = runpy.run_path(config.__file__)     # unpatched module values (conftest patches the live one)
    assert ns["LANGUAGES_PATH"] == ns["DATA_DIR"] / "languages.json"


def test_store_round_trip_across_restart(store_path):
    s = lang.LanguageStore(store_path)
    assert s.get(CHAT) is None
    s.set(CHAT, "ru")
    s.set(CHAT + 1, "uz_cyrl")
    assert s.get(CHAT) == "ru"
    fresh = lang.LanguageStore(store_path)     # new process after a restart
    assert fresh.get(CHAT) == "ru"
    assert fresh.get(CHAT + 1) == "uz_cyrl"
    assert fresh.get(CHAT + 2) is None


def test_store_file_holds_only_ids_and_codes(store_path):
    """NFR-4: no message text on disk."""
    lang.LanguageStore(store_path).set(CHAT, "ru")
    assert json.loads(store_path.read_text(encoding="utf-8")) == {str(CHAT): "ru"}


def test_store_atomic_write_leaves_no_temp_file(store_path):
    s = lang.LanguageStore(store_path)
    s.set(CHAT, "ru")
    s.set(CHAT, "uz_latn")
    assert sorted(p.name for p in store_path.parent.iterdir()) == ["languages.json"]


def test_store_creates_parent_dir(tmp_path):
    path = tmp_path / "data" / "languages.json"
    lang.LanguageStore(path).set(CHAT, "uz_cyrl")
    assert lang.LanguageStore(path).get(CHAT) == "uz_cyrl"


def test_store_saves_only_on_change(store_path):
    s = lang.LanguageStore(store_path)
    s.set(CHAT, "ru")
    store_path.unlink()
    s.set(CHAT, "ru")                          # same value → no write
    assert not store_path.exists()
    s.set(CHAT, "uz_latn")                     # change → written
    assert json.loads(store_path.read_text(encoding="utf-8")) == {str(CHAT): "uz_latn"}


def test_store_rejects_unknown_language(store_path):
    with pytest.raises(ValueError):
        lang.LanguageStore(store_path).set(CHAT, "en")


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize("content", ["{not json", "", "[1, 2]", '"ru"'], ids=["broken", "empty", "list", "string"])
def test_corrupt_file_warns_once_and_works(store_path, caplog, content):
    store_path.write_text(content, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        s = lang.LanguageStore(store_path)
        assert s.get(CHAT) is None
    assert len(_warnings(caplog)) == 1
    s.set(CHAT, "ru")
    assert lang.LanguageStore(store_path).get(CHAT) == "ru"


def test_missing_file_warns_once_and_works(store_path, caplog):
    with caplog.at_level(logging.WARNING):
        s = lang.LanguageStore(store_path)
        assert s.get(CHAT) is None
        assert s.get(CHAT + 1) is None
    assert len(_warnings(caplog)) == 1
    s.set(CHAT, "uz_cyrl")
    assert lang.LanguageStore(store_path).get(CHAT) == "uz_cyrl"


def test_get_store_follows_config_path(tmp_path, monkeypatch):
    first = lang.get_store()
    assert first is lang.get_store()
    assert first.path == config.LANGUAGES_PATH
    other = tmp_path / "other" / "languages.json"
    other.parent.mkdir()
    other.write_text(json.dumps({str(CHAT): "ru"}), encoding="utf-8")
    monkeypatch.setattr(config, "LANGUAGES_PATH", other)
    assert lang.get_store().get(CHAT) == "ru"


# ---------------------------------------------------------------- resolve

def test_first_contact_without_text_uses_lang_code(store_path):
    s = lang.LanguageStore(store_path)
    assert lang.resolve(CHAT, "/start", "ru", store=s) == "ru"
    assert lang.resolve(CHAT + 1, "/start", "uz", store=s) == "uz_latn"
    assert lang.resolve(CHAT + 2, "/start", "en", store=s) == "uz_latn"
    assert lang.resolve(CHAT + 3, "👍", None, store=s) == "uz_latn"


def test_text_beats_lang_code(store_path):
    """Many Uzbek customers run Telegram in Russian."""
    s = lang.LanguageStore(store_path)
    assert lang.resolve(CHAT, "krossovka 42 bormi?", "ru", store=s) == "uz_latn"
    assert s.get(CHAT) == "uz_latn"
    assert lang.resolve(CHAT + 1, "туфли 43 размер есть?", "uz", store=s) == "ru"


def test_stored_language_wins_over_lang_code(store_path):
    s = lang.LanguageStore(store_path)
    s.set(CHAT, "uz_latn")
    assert lang.resolve(CHAT, "/start", "ru", store=s) == "uz_latn"


@pytest.mark.parametrize("text", ["42", "ok", "👍", "/start", ""])
def test_unclear_message_keeps_stored_language(store_path, text):
    s = lang.LanguageStore(store_path)
    s.set(CHAT, "uz_cyrl")
    assert lang.resolve(CHAT, text, "ru", store=s) == "uz_cyrl"
    assert s.get(CHAT) == "uz_cyrl"


def test_one_russian_word_does_not_flip_the_stored_language(store_path):
    s = lang.LanguageStore(store_path)
    s.set(CHAT, "uz_latn")
    assert lang.resolve(CHAT, "dvoyka bormi, сколько?", "ru", store=s) == "uz_latn"
    assert lang.LanguageStore(store_path).get(CHAT) == "uz_latn"


def test_clear_message_switches_and_persists(store_path):
    s = lang.LanguageStore(store_path)
    s.set(CHAT, "uz_latn")
    assert lang.resolve(CHAT, "сколько стоит двойка?", "uz", store=s) == "ru"
    assert lang.LanguageStore(store_path).get(CHAT) == "ru"


def test_resolve_uses_the_default_store(monkeypatch, tmp_path):
    path = tmp_path / "default" / "languages.json"
    monkeypatch.setattr(config, "LANGUAGES_PATH", path)
    assert lang.resolve(CHAT, "кроссовка борми", "ru") == "uz_cyrl"
    assert lang.LanguageStore(path).get(CHAT) == "uz_cyrl"


# ---------------------------------------------------------------- fixed texts

KEYS = {"greeting", "ask_price_button", "view_in_channel_button", "stale_note", "escalated_reply",
        "callback_toast", "callback_error", "error_reply", "ask_price", "offer_price",
        "price_label", "sizes_label", "date_label",
        "owner_said"}   # ticket #23.7: label on FAQ answers ("egasi aytgan")


def _has_cyrillic(s: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in s)


def _has_latin(s: str) -> bool:
    return any("a" <= ch.lower() <= "z" for ch in s)


def test_texts_have_identical_keys_for_all_languages():
    assert set(lang.TEXTS) == set(lang.LANGS)
    for code in lang.LANGS:
        assert set(lang.TEXTS[code]) == KEYS, code


def test_texts_are_non_empty_strings():
    for code in lang.LANGS:
        for key, value in lang.TEXTS[code].items():
            assert isinstance(value, str) and value.strip(), (code, key)


def test_texts_are_in_their_own_script():
    for key in KEYS:
        assert not _has_cyrillic(lang.TEXTS["uz_latn"][key]), key
        assert _has_cyrillic(lang.TEXTS["uz_cyrl"][key]) and not _has_latin(lang.TEXTS["uz_cyrl"][key]), key
        assert _has_cyrillic(lang.TEXTS["ru"][key]) and not _has_latin(lang.TEXTS["ru"][key]), key
        assert not any(ch in lang.TEXTS["ru"][key].lower() for ch in "ўқғҳ"), key


def test_russian_and_uzbek_cyrillic_texts_differ():
    for key in ("greeting", "escalated_reply", "error_reply", "ask_price_button"):
        assert lang.TEXTS["ru"][key] != lang.TEXTS["uz_cyrl"][key], key


def test_uz_latin_texts_keep_the_current_wording():
    """Existing Uzbek Latin texts (bot.py / tools.py) do not change."""
    t = lang.TEXTS["uz_latn"]
    assert t["ask_price_button"] == "Narxini so'rash"
    assert t["view_in_channel_button"] == "Kanalda ko'rish"
    assert t["escalated_reply"] == "Egasi tez orada javob beradi 🙏"
    assert t["ask_price"] == "narxi: so'rab beraman"
    assert t["offer_price"] == "Narxini bilmoqchi bo'lsangiz raqamini yozing"
