# Data Pipeline: Fetch & Extract

Welcome to the second Deep Dive! Now that we know our data structures (the `Product`), let's see how they are actually created from the real world.

## Background
The shop owner posts pictures and text (captions) of new clothes on their Telegram channel. We need a way to take that messy human-readable text and turn it into the neat, structured `Product` dataclass.

This process happens in two distinct steps:
1. **Fetching (`fetch.py`)**: Scraping the channel for all the raw posts and saving them into a simple text file (`posts.jsonl`).
2. **Extracting (`extract.py`)**: Reading the raw posts, stripping away the useless noise (like emojis and phone numbers), and using an AI (LLM) to extract the exact `name`, `price`, `sizes`, and `colors`. The AI then outputs a `Product` which is saved to `products.jsonl`.

*Note: `.jsonl` stands for JSON Lines, meaning every single line in the file is a separate, valid JSON object. It's a great format for appending data one by one without loading a massive JSON array into memory!*

## Intuition
Imagine you are sorting incoming mail. 
**Fetching** is the mail carrier dropping the letters into a big box (`posts.jsonl`). They don't read them; they just drop them off. It's safe, simple, and you don't lose anything if you crash.
**Extracting** is you sitting down, opening the letters, throwing away the junk mail envelopes (emojis/links), and writing the important details (price, size) into an organized spreadsheet (`products.jsonl`). Since reading (using an LLM) is slow and expensive, we only want to do it *once* per post.

## Code Walkthrough
Let's look at how the code implements this.

### `fetch.py`
This script uses the `Telethon` library (a Telegram client) to connect as a user and scrape the channel history.
- `fetch()` connects and retrieves the raw posts.
- `dedup()` handles duplicates: it removes posts with empty captions and ensures if the exact same text is posted twice, only the latest ID is kept.
- `main()` appends only the *new* posts into `data/posts.jsonl`.

### `extract.py`
This is where the magic happens.
- `strip_footer()` uses Regular Expressions (Regex) to aggressively chop off delivery addresses, phone numbers, and emojis from the bottom of the caption. This saves LLM tokens and prevents the AI from getting confused.
- `_call_llm()` constructs a prompt and uses the Anthropic API (which is locally routed to an Ollama instance running `gemma4:31b`). We use **Tool Calling** (structured outputs). The LLM is given a tool called `record_products` and forced to "call" it with the exact fields we need (`price`, `category`, `sizes`, etc.).
- Because the LLM might be slow, `extract_batch()` processes posts in small chunks (e.g., 5 at a time).

## Concept Example
Run this standalone script to see how the fetching and extraction concepts map out in code:
[[02_fetch_extract_example.py]]

---

## Quizzes

**1. Why does `fetch.py` separate its job from `extract.py` instead of doing everything all at once?**

<details>
<summary>Option 1: Because Telethon cannot run at the same time as the Anthropic LLM API.</summary>

❌ Incorrect. You could run them together, but it's a bad design.
</details>

<details>
<summary>Option 2: Separation of concerns. Fetching is fast and cheap. Extracting via LLM is slow and prone to errors. Saving intermediate raw data to `posts.jsonl` prevents us from having to re-scrape the whole channel if the LLM crashes.</summary>

✅ Correct! This is a classic pipeline design. Save the raw data first, then process it.
</details>

**2. What is the main purpose of the `strip_footer()` function in `extract.py`?**

<details>
<summary>Option 1: To remove delivery boilerplate, phone numbers, and emojis to save LLM tokens and reduce AI confusion.</summary>

✅ Correct! The shop owner always pastes the same delivery instructions at the bottom. The LLM doesn't need to read that to know what the product is.
</details>

<details>
<summary>Option 2: To translate the text into English before sending it to the LLM.</summary>

❌ Incorrect. The LLM handles the Uzbek text directly!
</details>

---

## Your Mission 🎯

**Mission:** Open `shop_assistant/extract.py`. Look for the `parse_price(s: str)` function (around line 60). 
Read the regex and the logic inside that function. **Explain in your own words how it handles a string like `"Narxi: 350.000ming"` and turns it into an integer. What does it do with the word "ming"?**

Reply to me with your answer when you are ready!
