"""
Example 02: Simulating the Data Pipeline
Run this file directly!
"""
import re

# 1. Simulating strip_footer from extract.py
def strip_footer(caption: str) -> str:
    """Removes common boilerplate like phone numbers and emojis."""
    # A simple regex for phone numbers
    phone_re = re.compile(r"\+998|\d{9}")
    
    clean_lines = []
    for line in caption.splitlines():
        # If it has a phone number, skip the line entirely
        if phone_re.search(line):
            continue
        # If it has boilerplate words, skip it
        if "dastavka" in line.lower() or "📍" in line:
            continue
            
        clean_lines.append(line.strip())
        
    return "\n".join(clean_lines)

def main():
    raw_post = """
🔥 Yangi kurtkalar keldi!
Juda issiq, qish uchun.
Razmerlar: S, M, L
Narxi: 450.000 so'm

====================
Dastavka O'zbekiston bo'ylab bor 📍
Murojaat uchun: +998901234567
@example_shop_admin
    """.strip()
    
    print("--- RAW POST ---")
    print(raw_post)
    
    print("\n--- AFTER STRIP_FOOTER ---")
    cleaned = strip_footer(raw_post)
    print(cleaned)
    
    print("\nNotice how the LLM only has to read the important stuff now!")

if __name__ == "__main__":
    main()
