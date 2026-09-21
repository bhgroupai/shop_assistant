"""Load .env, start the bot, run forever. SDD §3.10. Ticket #15."""
import json
import logging

from dotenv import load_dotenv


def _startup_print() -> None:
    """One line so `journalctl` shows the index size the bot started with."""
    from shop_assistant import config, search
    n_products = len(getattr(search, "PRODUCTS", []))
    n_emb = 0
    try:
        if config.EMBEDDINGS_IDS_PATH.exists():
            n_emb = len(json.loads(config.EMBEDDINGS_IDS_PATH.read_text(encoding="utf-8")))
    except Exception:  # a broken ids file must not stop the bot
        pass
    print(f"loaded {n_products} products, {n_emb} embeddings", flush=True)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _startup_print()
    from shop_assistant import bot
    bot.run()


if __name__ == "__main__":
    main()
