# System Overview & Models

Welcome to your first Deep Dive into the Shop Assistant codebase! We are going to look at the high-level architecture and the core data structures that tie everything together.

## Background
When building an AI assistant for a shop (specifically over Telegram), there are generally two major phases:
1. **Offline/Background Data Pipeline**: Scraping products from a Telegram channel, understanding the messy unstructured text (like extracting price, color, size), and indexing them so we can search them fast.
2. **Online Customer Bot**: A chat interface where customers ask questions ("do you have warm winter jackets?"), an LLM figures out what they mean, searches the indexed data, and replies.

This project implements both phases. It uses an offline pipeline (`fetch.py` -> `extract.py` -> `index.py`) to build a database of products, and an online bot (`bot.py` -> `agent.py` -> `tools.py` -> `search.py`) to serve customers.

To make sure all these different scripts can talk to each other without confusion, they share common data structures. In Python, `dataclasses` are perfect for this.

## Intuition
Think of a dataclass as a well-labeled storage box. Instead of passing around random dictionaries like `{"id": 1, "n": "Jacket", "p": 500}` where you have to guess what `"n"` and `"p"` mean, you pass around a `Product` object. If a script needs to know the price, it just asks for `product.price`. If a script writes data, it is forced to provide all the required fields of a `Product`.

In this codebase, `[[models.py]]` is the source of truth for these storage boxes. Every stage of the pipeline inputs and outputs these specific models.

## Code Walkthrough (`models.py`)
Let's look at the actual code in `[[models.py]]`. There are three main classes:

1. **`Post`**: Represents a raw message fetched straight from the shop's Telegram channel. It contains unstructured `caption` (the text of the post).
2. **`Product`**: This is the heart of the system. The AI `extract.py` script reads a `Post` and spits out a `Product`. It breaks down the messy text into clean fields like `name`, `category`, `price`, `sizes`, and `colors`.
3. **`FaqEntry`**: Represents a question and answer snippet (useful when customers ask "how do you deliver?").

Notice that these are `@dataclass(frozen=True)`. "Frozen" means immutable — once you create a `Product`, you cannot change its price by doing `product.price = 10`. This prevents hard-to-track bugs where one script accidentally modifies a product that another script is reading.

## Concept Example
I have created a small, runnable Python script to show you how these dataclasses work in practice. Check out the example here:
[[01_models_example.py]]

---

## Quizzes

Test your understanding of what we just covered!

**1. Why does `models.py` use `@dataclass(frozen=True)` instead of regular dictionaries?**

<details>
<summary>Option 1: Because dictionaries cannot store integers and strings together.</summary>

❌ Incorrect. Dictionaries can store any mix of types.
</details>

<details>
<summary>Option 2: Because frozen dataclasses provide clear, documented structure and prevent accidental modifications to the data across different scripts.</summary>

✅ Correct! Dataclasses make it obvious what fields exist (like `price` vs `subscriber_price`), and being frozen stops bugs where data is mutated unexpectedly.
</details>


**2. In the context of this system, what is the difference between a `Post` and a `Product`?**

<details>
<summary>Option 1: A `Post` is unstructured text scraped from Telegram, while a `Product` is the clean, structured data extracted from it.</summary>

✅ Correct! The `Post` just has the raw `caption` text, whereas the `Product` breaks it down into `price`, `sizes`, `category`, etc.
</details>

<details>
<summary>Option 2: A `Product` is fetched from Telegram, and a `Post` is what the bot replies to the customer.</summary>

❌ Incorrect. The bot replies with plain text strings. A `Post` is the raw fetched data.
</details>

---

## Your Mission 🎯
Now it's time for you to look at the code!

**Mission:** Open `shop_assistant/models.py`. Look at the `Product` dataclass. Notice the `stale` field. Read the comment next to it. Based on your intuition of a shop system, **why do you think a product would need a `stale` flag, and how might that affect the search results shown to a customer?**

Reply to me with your answer!
