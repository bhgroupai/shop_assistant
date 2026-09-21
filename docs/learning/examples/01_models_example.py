"""
Example 01: Working with Data Models
Run this file directly to see how it works!
"""

from dataclasses import dataclass

# Here is a simplified version of our models from models.py
@dataclass(frozen=True)
class Post:
    id: int
    caption: str

@dataclass(frozen=True)
class Product:
    id: int
    name: str
    price: int | None
    sizes: tuple[str, ...]

def main():
    # 1. Fetching phase (Simulated)
    # We get a messy post from Telegram
    raw_post = Post(
        id=101, 
        caption="🔥 Yangi krossovkalar!\nNarxi: 350.000 so'm\nRazmerlar: 40, 41, 42\nDastavka bor 📍"
    )
    print(f"RAW POST TEXT:\n{raw_post.caption}\n")

    # 2. Extraction phase (Simulated)
    # Our AI reads the raw_post and structures it into a Product
    structured_product = Product(
        id=raw_post.id,
        name="krossovkalar",
        price=350000,
        sizes=("40", "41", "42")
    )
    
    # Now the rest of the system can easily read the data!
    print(f"STRUCTURED PRODUCT:")
    print(f"Name:  {structured_product.name}")
    print(f"Price: {structured_product.price} UZS")
    print(f"Sizes: {', '.join(structured_product.sizes)}")
    
    # 3. Why frozen=True is good:
    try:
        # If a script accidentally tries to modify the data...
        structured_product.price = 999999
    except Exception as e:
        print(f"\nERROR PREVENTED: {e}")
        print("Because the class is frozen, we are protected from accidental modifications!")

if __name__ == "__main__":
    main()
