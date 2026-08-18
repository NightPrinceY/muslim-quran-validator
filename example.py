"""
Minimal usage example for the Quran recitation validator.

Run with: python3 example.py
"""
import asyncio
import sys
from pathlib import Path

VALIDATOR_DIR = Path(__file__).parent / "validator"
sys.path.insert(0, str(VALIDATOR_DIR))

from validator_mcp import validate_recitation


async def main():
    # A correct recitation of Surah Al-Ikhlas, verse 1.
    result = await validate_recitation("قل هو الله احد")
    print("Feedback:", result["feedback"])
    print("Matched verse:", result.get("verse_key"))
    print("WER:", result.get("wer"))

    # The same verse typed in Uthmani script, exercising the
    # Uthmani-to-Standard mapping (validator/data/uthmani_standard_map.json).
    result = await validate_recitation("قُلْ هُوَ اللَّهُ أَحَدٌ")
    print("\nUthmani input feedback:", result["feedback"])


if __name__ == "__main__":
    asyncio.run(main())
