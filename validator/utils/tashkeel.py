"""
Tashkeel (harakat/diacritics) validator.

Validates the user's tashkeel against the standard_full Quran text.
Only runs when the user's input actually contains harakat marks.

Harakat compared:
  - Fatha (فتحة)       U+064E
  - Damma (ضمة)        U+064F
  - Kasra (كسرة)       U+0650
  - Shadda (شدة)       U+0651
  - Sukun (سكون)       U+0652
  - Tanwin fath        U+064B
  - Tanwin damm        U+064C
  - Tanwin kasr        U+064D
  - Alef khanjariya    U+0670
"""
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from utils.normalizer import (
    normalize_arabic,
    normalize_keep_tashkeel,
    has_tashkeel,
    HARAKAT_NAMES,
    HARAKAT_CHARS,
)


# ── Tashkeel extraction helpers ──────────────────────────────────────────────

_HARAKAT_RE = re.compile(r'[\u064B-\u0652\u0670]')


def _extract_word_harakat(word_with_harakat: str) -> List[Tuple[str, str]]:
    """
    For each letter in a word, extract its associated harakat.

    Returns list of (letter, harakat_string) pairs.
    E.g. 'بِسْمِ' → [('ب', 'ِ'), ('س', 'ْ'), ('م', 'ِ')]
    Shadda is combined with the following harakat.
    """
    result: List[Tuple[str, str]] = []
    current_letter: Optional[str] = None
    current_marks: List[str] = []

    for char in word_with_harakat:
        if '\u0621' <= char <= '\u064A' or char == '\u0671':
            # It's an Arabic letter
            if current_letter is not None:
                result.append((current_letter, ''.join(current_marks)))
            current_letter = char
            current_marks = []
        elif char in HARAKAT_CHARS or char == '\u0651':
            if current_letter is not None:
                current_marks.append(char)
            # else: leading harakat before a letter — skip

    if current_letter is not None:
        result.append((current_letter, ''.join(current_marks)))

    return result


def _harakat_of_word(word: str) -> str:
    """Extract just the harakat sequence from a word (for quick comparison)."""
    return ''.join(c for c in word if c in HARAKAT_CHARS or c == '\u0651')


def _describe_harakat(h: str) -> str:
    """Convert harakat string to human-readable Arabic description."""
    if not h:
        return 'بدون حركة'
    parts = []
    for c in h:
        parts.append(HARAKAT_NAMES.get(c, c))
    return ' + '.join(parts)


# ── Word-level tashkeel comparison ──────────────────────────────────────────

def compare_word_tashkeel(recited_word: str, reference_word: str) -> Optional[Dict]:
    """
    Compare harakat of two words (same skeleton assumed).

    Returns None if tashkeel is identical, else a dict:
      {
        word:     the bare word,
        recited:  harakat from user,
        expected: harakat from reference,
        recited_desc:  human-readable Arabic,
        expected_desc: human-readable Arabic,
      }
    """
    r_harakat = _harakat_of_word(recited_word)
    e_harakat = _harakat_of_word(reference_word)

    if r_harakat == e_harakat:
        return None

    skeleton = normalize_arabic(recited_word)
    return {
        "word": skeleton,
        "recited_harakat": r_harakat,
        "expected_harakat": e_harakat,
        "recited_desc": _describe_harakat(r_harakat),
        "expected_desc": _describe_harakat(e_harakat),
    }


# ── Verse-level tashkeel validation ─────────────────────────────────────────

def validate_tashkeel(recited_text: str, reference_standard_full: str) -> Dict:
    """
    Validate harakat in recited_text against reference_standard_full.

    Only runs if recited_text contains harakat — otherwise returns
    {has_tashkeel: False}.

    Returns:
      has_tashkeel:   bool — user provided harakat
      is_correct:     bool — all harakat match
      errors:         list of per-word error dicts
      error_count:    int
      total_words:    int (words that were compared)
    """
    if not has_tashkeel(recited_text):
        return {"has_tashkeel": False, "is_correct": True, "errors": [], "error_count": 0}

    # Normalize structurally but keep harakat
    rec_norm = normalize_keep_tashkeel(recited_text)
    ref_norm = normalize_keep_tashkeel(reference_standard_full)

    rec_words = [w for w in rec_norm.split() if w]
    ref_words = [w for w in ref_norm.split() if w]

    # Align words skeleton-by-skeleton
    rec_skeletons = [normalize_arabic(w) for w in rec_words]
    ref_skeletons = [normalize_arabic(w) for w in ref_words]

    sm = SequenceMatcher(None, rec_skeletons, ref_skeletons, autojunk=False)
    opcodes = sm.get_opcodes()

    errors: List[Dict] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Same words — compare harakat
            for i, j in zip(range(i1, i2), range(j1, j2)):
                err = compare_word_tashkeel(rec_words[i], ref_words[j])
                if err:
                    err["position"] = j + 1
                    errors.append(err)
        elif tag == 'replace':
            # Words differ — compare what we can
            for k in range(min(i2 - i1, j2 - j1)):
                err = compare_word_tashkeel(rec_words[i1 + k], ref_words[j1 + k])
                if err:
                    err["position"] = j1 + k + 1
                    errors.append(err)
        # insert/delete — word itself is wrong, no point comparing harakat

    return {
        "has_tashkeel": True,
        "is_correct": len(errors) == 0,
        "errors": errors,
        "error_count": len(errors),
        "total_words": len(ref_words),
    }
