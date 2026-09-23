"""Captions in the real channel format (SDD §2.1) as fixtures. Never 'foo'."""
import pytest

from shop_assistant import config
from shop_assistant.models import Post, Product


@pytest.fixture(autouse=True)
def _channel(monkeypatch):
    monkeypatch.setattr(config, "CHANNEL", "example_shop")


@pytest.fixture(autouse=True)
def _languages_file(monkeypatch, tmp_path):
    """Ticket #24: the per-chat language store never touches the real data/languages.json in tests;
    each test starts with an empty store at its own path."""
    monkeypatch.setattr(config, "LANGUAGES_PATH", tmp_path / "languages.json", raising=False)


@pytest.fixture(autouse=True)
def _faq_files(monkeypatch, tmp_path):
    """Ticket #23.7: the FAQ store never touches the real data/faq*.{jsonl,npy,json} in tests;
    each test starts with an empty store under its own tmp dir."""
    monkeypatch.setattr(config, "FAQ_PATH", tmp_path / "faq" / "faq.jsonl")
    monkeypatch.setattr(config, "FAQ_EMBEDDINGS_PATH", tmp_path / "faq" / "faq_embeddings.npy")
    monkeypatch.setattr(config, "FAQ_META_PATH", tmp_path / "faq" / "faq_embeddings_meta.json", raising=False)


FOOTER = "\n\n📍Manzil: Shahar markazi\n📞 +998 00 000 00 00\n@example_shop\n🚚 Dastavka bor"

CAPTION_DVOYKA = (
    "🍂Kuz mavsumi uchun🍂\n🔥Yangi model Dvoyka🔥\nRazmer:M.L.XL.2XL.3XL\nNarx:980.000ming" + FOOTER
)
CAPTION_KROSSOVKA = (
    "👟Krossovka Nike Air👟\nRazmer: 40 41 42 43\nNarx: 350.000\n"
    "Telegram obunachilariga narx: 320.000" + FOOTER
)
CAPTION_KURTKA = "🧥Qishki kurtka\nRang: qora, kok\nRazmer: L XL\nNarx: 1.200.000" + FOOTER

@pytest.fixture
def posts() -> list[Post]:
    return [
        Post(id=1234, date="2026-09-10T14:02:00", link="https://t.me/example_shop/1234",
             caption=CAPTION_DVOYKA),
        Post(id=1300, date="2026-09-12T10:00:00", link="https://t.me/example_shop/1300",
             caption=CAPTION_KROSSOVKA),
        Post(id=1301, date="2026-09-13T09:00:00", link="https://t.me/example_shop/1301",
             caption=CAPTION_DVOYKA),      # repost of 1234 — FR-3 keeps this one
        Post(id=1302, date="2026-09-13T09:05:00", link="https://t.me/example_shop/1302",
             caption=""),                  # FR-2 skipped
    ]


@pytest.fixture
def products() -> list[Product]:
    return [
        Product(id=1234, date="2026-09-10", link="https://t.me/example_shop/1234",
                name="Dvoyka", category="kiyim", price=980000, subscriber_price=None,
                sizes=("M", "L", "XL", "2XL", "3XL"), colors=(),
                keywords=("dvoyka", "dvoyka", "kostyum dvoyka", "two-piece set", "sport kostyum"),
                season="kuz", body="Yangi model Dvoyka Razmer M L XL 2XL 3XL"),
        Product(id=1300, date="2026-09-12", link="https://t.me/example_shop/1300",
                name="Krossovka Nike Air", category="poyabzal", price=350000, subscriber_price=320000,
                sizes=("40", "41", "42", "43"), colors=(),
                keywords=("krossovka", "krossovki", "sneakers", "nike"),
                body="Krossovka Nike Air Razmer 40 41 42 43"),
        Product(id=900, date="2026-06-01", link="https://t.me/example_shop/900",
                name="Qishki kurtka", category="kiyim", price=1200000, subscriber_price=None,
                sizes=("L", "XL"), colors=("qora", "kok"),
                keywords=("kurtka", "kurtka", "jacket", "qishki kurtka"),
                season="qish", body="Qishki kurtka Rang qora kok Razmer L XL"),   # > 60 days → stale
    ]


# ---- ticket #25: a catalog big enough to page through (23 size-42 shoes, as in the live bug)

_SHOE_MODELS = [
    ("Krossovka Nike Air Max 90", ("krossovka", "nike", "sneakers")),
    ("Krossovka Adidas Samba", ("krossovka", "adidas", "sneakers")),
    ("Krossovka New Balance 574", ("krossovka", "new balance", "sneakers")),
    ("Krossovka Puma Suede", ("krossovka", "puma", "sneakers")),
    ("Krossovka Asics Gel-Lyte", ("krossovka", "asics", "sneakers")),
    ("Krossovka Nike Air Force 1", ("krossovka", "nike", "air force")),
    ("Krossovka Adidas Superstar", ("krossovka", "adidas", "superstar")),
    ("Krossovka Reebok Classic", ("krossovka", "reebok", "sneakers")),
    ("Kedi Converse Chuck 70", ("kedi", "converse", "kedy")),
    ("Kedi Vans Old Skool", ("kedi", "vans", "kedy")),
    ("Tufli klassik charm", ("tufli", "klassik", "charm tufli")),
    ("Botinka Timberland", ("botinka", "timberland", "boots")),
    ("Botinka Dr. Martens 1460", ("botinka", "dr martens", "boots")),
    ("Mokasin charm", ("mokasin", "charm", "loafers")),
    ("Krossovka Nike Dunk Low", ("krossovka", "nike", "dunk")),
    ("Krossovka Adidas Gazelle", ("krossovka", "adidas", "gazelle")),
    ("Krossovka Salomon XT-6", ("krossovka", "salomon", "sneakers")),
    ("Krossovka On Cloud 5", ("krossovka", "on cloud", "running")),
    ("Krossovka Hoka Clifton 9", ("krossovka", "hoka", "running")),
    ("Krossovka Skechers Go Walk", ("krossovka", "skechers", "walking")),
    ("Kedi Nike Blazer Mid", ("kedi", "nike", "blazer")),
    ("Botinka qishki mo'ynali", ("botinka", "qishki", "winter boots")),
    ("Krossovka Jordan 1 Low", ("krossovka", "jordan", "nike")),
]


@pytest.fixture
def shoe_catalog() -> list[Product]:
    """35 sellable products + 1 announcement. 23 shoes (poyabzal) carry size 42, one post per day
    2026-08-01 … 2026-08-23 (id 2001 … 2023, so newest first = 2023, 2022, …); prices 250 000 … 910 000
    (step 30 000), two of them without a price (ids 2004 and 2017). 6 more shoes without size 42,
    5 clothes, 1 accessory, 1 `boshqa` announcement (never shown)."""
    out: list[Product] = []
    for i, (name, kws) in enumerate(_SHOE_MODELS, start=1):
        pid = 2000 + i
        price = None if pid in (2004, 2017) else 250000 + 30000 * (i - 1)
        sizes = ("40", "41", "42", "43") if i % 2 else ("41", "42", "43", "44")
        out.append(Product(id=pid, date=f"2026-08-{i:02d}", link=f"https://t.me/example_shop/{pid}",
                           name=name, category="poyabzal", price=price, subscriber_price=None,
                           sizes=sizes, colors=("qora",) if i % 3 == 0 else ("oq",),
                           keywords=kws, body=f"{name} Razmer {' '.join(sizes)}"))
    for j, (name, sizes) in enumerate([("Krossovka bolalar Nike", ("31", "32", "33")),
                                       ("Krossovka ayollar Adidas", ("36", "37", "38")),
                                       ("Tufli ayollar", ("36", "37")),
                                       ("Botinka Timberland 44", ("44", "45")),
                                       ("Shippak Adidas Adilette", ("39", "40", "41")),
                                       ("Krossovka Puma RS-X", ("43", "44"))], start=1):
        pid = 2100 + j
        out.append(Product(id=pid, date=f"2026-09-{j:02d}", link=f"https://t.me/example_shop/{pid}",
                           name=name, category="poyabzal", price=300000 + 10000 * j, subscriber_price=None,
                           sizes=sizes, colors=(), keywords=("krossovka",) if "Krossovka" in name else ("oyoq kiyim",),
                           body=f"{name} Razmer {' '.join(sizes)}"))
    for k, (name, kws) in enumerate([("Dvoyka sport kostyum", ("dvoyka", "sport kostyum")),
                                     ("Qishki kurtka", ("kurtka", "jacket")),
                                     ("Futbolka oversize", ("futbolka", "t-shirt")),
                                     ("Jinsi shim", ("jinsi", "shim", "jeans")),
                                     ("Xudi Nike", ("xudi", "hoodie"))], start=1):
        pid = 2200 + k
        out.append(Product(id=pid, date=f"2026-09-{10 + k:02d}", link=f"https://t.me/example_shop/{pid}",
                           name=name, category="kiyim", price=200000 + 100000 * k, subscriber_price=None,
                           sizes=("M", "L", "XL"), colors=("qora",), keywords=kws, body=name))
    out.append(Product(id=2300, date="2026-09-16", link="https://t.me/example_shop/2300",
                       name="Ryukzak Nike", category="aksessuar", price=250000, subscriber_price=None,
                       sizes=(), colors=("qora",), keywords=("ryukzak", "backpack"), body="Ryukzak Nike"))
    out.append(Product(id=2400, date="2026-09-20", link="https://t.me/example_shop/2400",
                       name="Boshqa", category="boshqa", price=None, subscriber_price=None,
                       sizes=(), colors=(), keywords=("yangi kolleksiya",), body="Yangi kolleksiya tez orada"))
    return out
