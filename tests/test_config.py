import pytest

from shop_assistant import config


def test_constants_match_srs():
    assert config.STALE_DAYS == 60          # FR-16
    assert config.MAX_RESULTS == 5          # FR-18
    assert config.FETCH_LIMIT == 500        # FR-1
    assert config.HISTORY_TURNS == 10       # FR-17
    assert set(config.CATEGORIES) == {"kiyim", "poyabzal", "aksessuar", "boshqa"}


def test_data_paths_are_gitignored():
    ignored = (config.ROOT / ".gitignore").read_text()
    assert "data/" in ignored and "session/" in ignored and ".env" in ignored   # NFR-6


def test_secret_missing_raises(monkeypatch):
    monkeypatch.delenv("TG_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TG_BOT_TOKEN"):
        config.secret("TG_BOT_TOKEN")


def test_secret_unknown_key():
    with pytest.raises(KeyError):
        config.secret("NOT_A_SECRET")


def test_secret_present(monkeypatch):
    monkeypatch.setenv("TG_BOT_TOKEN", "123:abc")
    assert config.secret("TG_BOT_TOKEN") == "123:abc"


def test_embeddings_on_gemini():                          # ticket #23.5
    assert isinstance(config.EMBED_MODEL, str) and "embedding" in config.EMBED_MODEL
    assert "bge" not in config.EMBED_MODEL.lower()
    assert isinstance(config.EMBED_DIM, int) and 128 <= config.EMBED_DIM <= 3072
    assert isinstance(config.EMBED_BATCH, int) and 1 <= config.EMBED_BATCH <= 100   # embed_content limit
    assert config.EMBEDDINGS_META_PATH == config.DATA_DIR / "embeddings_meta.json"


def test_no_ollama_or_anthropic_settings():              # ticket #23.5
    assert not hasattr(config, "OLLAMA_URL")
    assert not [k for k in vars(config) if k.upper().startswith(("ANTHROPIC", "OLLAMA"))]


def test_gemini_model_replaces_local_gemma():          # ticket #23
    model = getattr(config, "GEMINI_MODEL", None)
    assert isinstance(model, str) and model.startswith("gemini")
    assert not hasattr(config, "MODEL"), "MODEL (gemma4) is replaced by GEMINI_MODEL"
    assert not hasattr(config, "THINKING"), "THINKING was gemma4-specific"
    assert config.MAX_ITERATIONS == 8


def test_gemini_key_is_a_known_secret(monkeypatch):     # ticket #23
    assert "GEMINI_API_KEY" in config.ENV_KEYS
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-shop-assistant")
    assert config.secret("GEMINI_API_KEY") == "AIza-test-key-shop-assistant"


def test_gemini_key_missing_raises(monkeypatch):         # ticket #23
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        config.secret("GEMINI_API_KEY")


def test_no_real_key_in_env_example():                   # ticket #23: placeholder only
    example = (config.ROOT / ".env.example").read_text()
    assert "GEMINI_API_KEY" in example
    assert "AIza" not in example
