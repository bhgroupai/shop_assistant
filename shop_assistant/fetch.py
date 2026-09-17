"""Dump channel posts to posts.jsonl. SDD §3.1, FR-1/2/3/7/23. Ticket #4."""
import asyncio
import json
from dataclasses import asdict

from telethon.sync import TelegramClient

from shop_assistant import config
from shop_assistant.models import Post
from shop_assistant.textnorm import normalise


def dedup(posts: list[Post]) -> list[Post]:
    """Drop empty captions (FR-2); among posts with equal normalise(caption) keep the newest id (FR-3)."""
    groups: dict[str, Post] = {}
    for post in posts:
        caption = (post.caption or "").strip()
        if not caption:
            continue
        norm_cap = normalise(caption)
        if not norm_cap:
            continue
        if norm_cap not in groups or post.id > groups[norm_cap].id:
            groups[norm_cap] = post
    return sorted(groups.values(), key=lambda p: p.id)


async def _fetch(channel: str, min_id: int, limit: int) -> list[Post]:
    config.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    session_path = str(config.SESSION_DIR / "user")
    api_id = int(config.secret("TG_API_ID"))
    api_hash = config.secret("TG_API_HASH")

    posts = []
    async with TelegramClient(session_path, api_id, api_hash) as client:
        async for message in client.iter_messages(channel, min_id=min_id, limit=limit):
            if message.text and message.text.strip():
                posts.append(Post(
                    id=message.id,
                    date=message.date.isoformat(),
                    link=f"https://t.me/{channel}/{message.id}",
                    caption=message.text,
                    has_media=bool(message.media)
                ))
    return dedup(posts)


def fetch(channel: str, min_id: int = 0, limit: int = 500) -> list[Post]:
    """Telethon user account, iter_messages(channel, min_id, limit) → Post list, already dedup'ed."""
    return asyncio.run(_fetch(channel, min_id, limit))


def main(full: bool = False) -> None:
    """CLI: append new posts to posts.jsonl and update state.last_post_id (FR-7)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    min_id = 0
    if not full and config.STATE_PATH.exists():
        try:
            with open(config.STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
                min_id = state.get("last_post_id", 0)
        except (json.JSONDecodeError, OSError):
            pass

    posts = fetch(config.CHANNEL, min_id, config.FETCH_LIMIT)
    
    if posts:
        with open(config.POSTS_PATH, "a", encoding="utf-8") as f:
            for post in posts:
                json.dump(asdict(post), f, ensure_ascii=False)
                f.write("\n")
                
        last_post_id = max(p.id for p in posts)
        
        state = {}
        if config.STATE_PATH.exists():
            try:
                with open(config.STATE_PATH, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        
        state["last_post_id"] = max(last_post_id, state.get("last_post_id", 0))
        with open(config.STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)
    else:
        last_post_id = min_id
        
    print(f"fetched {len(posts)} posts, last_post_id={last_post_id}")


if __name__ == "__main__":
    import sys
    main(full="--full" in sys.argv)
