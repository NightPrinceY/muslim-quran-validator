"""
Complete Arabic text normalizer for Quran recitation validation.

Handles:
 - All tashkeel / harakat (U+064B–U+065F, U+0670, U+06D6–U+06FC)
 - Uthmani-specific chars: alef wasla (U+0671), Farsi yeh (U+06CC),
   small high marks, Quranic annotation marks
 - Alef variants: أإآٱ → ا
 - Hamza variants: ؤئ → ء
 - Final yeh / alef maqsura: ى → ي
 - Tatweel (kashida U+0640)
 - Zero-width / directional chars
 - Unicode normalization (NFC)
 - Word-level Uthmani→standard map (data/uthmani_standard_map.json)
   Provides 2017 exact word-level mappings derived from the full Quran corpus,
   covering all ٰ (U+0670) variants with 100% accuracy from aligned data.
"""
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

# ── Harakat ranges ──────────────────────────────────────────────────────────
#  U+064B  ARABIC FATHATAN
#  U+064C  ARABIC DAMMATAN
#  U+064D  ARABIC KASRATAN
#  U+064E  ARABIC FATHA
#  U+064F  ARABIC DAMMA
#  U+0650  ARABIC KASRA
#  U+0651  ARABIC SHADDA
#  U+0652  ARABIC SUKUN
#  U+0653  ARABIC MADDAH ABOVE
#  U+0654  ARABIC HAMZA ABOVE
#  U+0655  ARABIC HAMZA BELOW
#  U+0656–U+065F  other Arabic signs
#  U+0670  ARABIC LETTER SUPERSCRIPT ALEF  ← handled by word-level map first
#  U+06D6–U+06DC  Quranic annotation marks
#  U+06DF–U+06E8  more Quranic marks
#  U+06EA–U+06FC  more Quranic/Extended Arabic marks

_RE_TASHKEEL = re.compile(
    r'[\u064B-\u065F'                 # standard harakat (NOT U+0670 — handled by map/fallback)
    r'\u06D6-\u06DC\u06DF-\u06E8'    # Quranic annotation marks
    r'\u06EA-\u06FC'                  # more Quranic/Extended marks
    r'\u06E1'                         # small high dotless head of khah (Uthmani)
    r']'
)

# Fallback rule for U+0670 not covered by word-level map:
# Replace ٰ → ا UNLESS preceded by ى/ذ/ه/ل (where standard omits the alef).
_RE_SUPERSCRIPT_ALEF_REPLACE = re.compile(r'(?<![ىذهل])\u0670')

# Uthmani / Farsi / non-standard chars that map to standard Arabic letters
_UTHMANI_MAP: List[Tuple[str, str]] = [
    ('\u0671', '\u0627'),  # ARABIC LETTER ALEF WASLA → ALEF
    ('\u06CC', '\u064A'),  # ARABIC LETTER FARSI YEH → YEH
    ('\u0640', ''),        # ARABIC TATWEEL (kashida) → remove
]

_RE_ALEF = re.compile(r'[\u0623\u0625\u0622\u0671\u0627]')  # أإآٱا → ا
_RE_HAMZA = re.compile(r'[\u0624\u0626]')                    # ؤئ → ء
_RE_ZERO_WIDTH = re.compile(r'[\u200B-\u200F\u202A-\u202E\uFEFF]')
_RE_NON_ARABIC = re.compile(r'[^\u0621-\u064A\s]')
_RE_MULTI_SPACE = re.compile(r'\s+')


# ── Word-level Uthmani→Standard map ─────────────────────────────────────────
# Loaded at import time from data/uthmani_standard_map.json.
# Key:   Uthmani word with tashkeel stripped but U+0670 preserved (e.g. 'الكتٰب')
# Value: Standard Arabic word, normalized (e.g. 'الكتاب')
# Only applied to words that contain U+0670 — others use character-level rules.

def _build_uthmani_key(word: str) -> str:
    """Produce the map lookup key: strip tashkeel/marks but keep ٰ (U+0670)."""
    w = word.replace('\u0671', '\u0627').replace('\u06CC', '\u064A').replace('\u0640', '')
    w = re.sub(r'[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06FC\u06E1]', '', w)
    w = re.sub(r'[^\u0621-\u064A\u0670]', '', w)  # keep Arabic letters + ٰ only
    return w


def _load_uthmani_map() -> dict:
    map_path = Path(__file__).parent.parent / 'data' / 'uthmani_standard_map.json'
    if map_path.exists():
        with open(map_path, encoding='utf-8') as f:
            return json.load(f)
    return {}


_UTHMANI_WORD_MAP: dict = _load_uthmani_map()


def _apply_word_map(text: str) -> str:
    """
    Word-level pass: for each whitespace-separated token that contains ٰ (U+0670),
    look it up in the corpus-derived map and replace with the standard form.
    Tokens without ٰ are left unchanged (they don't need this pass).
    """
    if '\u0670' not in text:
        return text
    words = text.split(' ')
    out = []
    for w in words:
        if '\u0670' in w:
            key = _build_uthmani_key(w)
            mapped = _UTHMANI_WORD_MAP.get(key)
            if mapped is not None:
                out.append(mapped)
                continue
        out.append(w)
    return ' '.join(out)


# ── Individual harakat for tashkeel validation ──────────────────────────────

# Harakat grouped by function (for readable error messages)
HARAKAT_NAMES = {
    '\u064B': 'تنوين فتح',
    '\u064C': 'تنوين ضم',
    '\u064D': 'تنوين كسر',
    '\u064E': 'فتحة',
    '\u064F': 'ضمة',
    '\u0650': 'كسرة',
    '\u0651': 'شدة',
    '\u0652': 'سكون',
    '\u0653': 'مدة',
    '\u0670': 'ألف خنجرية',
}

# Harakat characters (for extraction)
HARAKAT_CHARS = set(HARAKAT_NAMES.keys())


def remove_tashkeel(text: str) -> str:
    """Remove all diacritical marks including Quranic annotation marks."""
    for src, dst in _UTHMANI_MAP:
        text = text.replace(src, dst)
    return _RE_TASHKEEL.sub('', text)


def normalize_arabic(text: str) -> str:
    """
    Full canonical normalization for search/comparison.

    Pipeline:
      1. Unicode NFC
      2. Word-level Uthmani→standard map (corpus-derived, 2017 entries)
         — replaces ٰ-containing words with their exact standard forms
      3. Remove all tashkeel and Quranic marks (U+0670 fallback handled next)
      4. Fallback: contextual U+0670 → ا (unless preceded by ى/ذ/ه/ل)
      5. Alef variants → bare alef (أإآٱ → ا)
      6. Hamza variants → bare hamza (ؤئ → ء)
      7. Uthmani word-initial ءا → ا  (ءَاتَ = standard آتَ = normalized اتَ)
      8. Final yeh → yeh (ى → ي)
      9. Remove zero-width and directional chars
     10. Remove any remaining non-Arabic-letter chars
     11. Collapse whitespace
    """
    text = unicodedata.normalize('NFC', text)
    text = _apply_word_map(text)             # step 2: exact word-level map
    text = remove_tashkeel(text)             # step 3: strip tashkeel; ٰ may remain
    text = _RE_SUPERSCRIPT_ALEF_REPLACE.sub('\u0627', text)  # step 4a: ٰ → ا
    text = text.replace('\u0670', '')        # step 4b: remove remaining ٰ
    text = _RE_ALEF.sub('\u0627', text)      # step 5: أإآٱ → ا
    text = _RE_HAMZA.sub('\u0621', text)     # step 6: ؤئ → ء
    # Step 7: Uthmani word-initial ءا = standard آ (normalized to ا)
    text = re.sub(r'(^| )\u0621(?=\u0627)', r'\1', text)
    text = text.replace('\u0649', '\u064A')  # step 8: ى → ي
    text = _RE_ZERO_WIDTH.sub('', text)
    text = _RE_NON_ARABIC.sub('', text)
    text = _RE_MULTI_SPACE.sub(' ', text)
    return text.strip()


def normalize_keep_tashkeel(text: str) -> str:
    """
    Normalization that KEEPS tashkeel — used for tashkeel validation.
    Only applies structural substitutions (alef variants, hamza, yeh).
    Removes Uthmani-only marks not part of standard tashkeel.
    """
    text = unicodedata.normalize('NFC', text)
    for src, dst in _UTHMANI_MAP:
        text = text.replace(src, dst)
    # Remove Quranic annotation marks (not standard tashkeel)
    text = re.sub(r'[\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06FC]', '', text)
    text = _RE_ALEF.sub('\u0627', text)
    text = _RE_HAMZA.sub('\u0621', text)
    text = text.replace('\u0649', '\u064A')
    text = _RE_ZERO_WIDTH.sub('', text)
    text = _RE_MULTI_SPACE.sub(' ', text)
    return text.strip()


def extract_harakat(word: str) -> str:
    """
    Extract only the harakat characters from a word (preserves order/position).
    Returns a string of harakat chars, e.g. '\u064E\u0651\u0650'.
    """
    return ''.join(c for c in word if c in HARAKAT_CHARS or c == '\u0651')


def word_skeleton(word: str) -> str:
    """Strip all harakat; return bare letter skeleton."""
    return normalize_arabic(word)


def has_tashkeel(text: str) -> bool:
    """Return True if the text contains any tashkeel marks."""
    return bool(re.search(r'[\u064B-\u0652\u0670]', text))
