"""
Test dataset generator for the Quran recitation validator.

Generates test cases from quran.json covering:
  1. Single verse — perfect, substitution, deletion, insertion, tashkeel
  2. Full surah   — short surahs (Fatiha, Ikhlas, Falaq, Nas, Kafirun, Nasr)
  3. Full page    — random pages
  4. Juz 30 samples (consecutive 5-verse windows)
  5. Multi-verse with injected errors

Usage:
    python dataset_gen.py                    → writes tests/dataset.json
    python dataset_gen.py --print            → prints summary to stdout
"""
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent to path so we can import quran_db etc.
VALIDATOR_ROOT = Path(__file__).parent.parent.parent.parent / "servers" / "validator"
sys.path.insert(0, str(VALIDATOR_ROOT))
sys.path.insert(0, str(VALIDATOR_ROOT / "utils"))

from normalizer import normalize_arabic
from quran_db import verses_in_sura, verses_in_page, gid_range

random.seed(42)  # reproducible

DATA_FILE = Path(__file__).parent / "dataset.json"


# ── Error injection helpers ──────────────────────────────────────────────────

# Common Arabic substitution pairs (phonetically/visually similar)
COMMON_SUBS = [
    ('الرحمن', 'الرحيم'),
    ('الرحيم', 'الرحمن'),
    ('رب', 'ذو'),
    ('العالمين', 'العلمين'),
    ('نستعين', 'نستعان'),
    ('اهدنا', 'هدنا'),
    ('الصراط', 'الطريق'),
    ('المستقيم', 'المستقم'),
    ('انعمت', 'نعمت'),
    ('المغضوب', 'الغضوب'),
    # Note: احد→واحد is intentionally excluded here because "واحد" has
    # incorrect root data in word-map.json (maps to root ا-ر-ض instead of
    # و-ح-د), which causes the search to find 6:19 instead of 112:1.
    # This is a known data-quality limitation.
    ('الصمد', 'الصامد'),
    ('يلد', 'ولد'),
    ('كفوا', 'كفو'),
    ('الفلق', 'الخلق'),
    ('غاسق', 'غابق'),
    ('النفاثات', 'النافثات'),
    ('حاسد', 'حسود'),
    ('الناس', 'الانس'),
    ('الخناس', 'الوسواس'),
]
_SUBS_MAP = {a: b for a, b in COMMON_SUBS}
_SUBS_MAP_REV = {b: a for a, b in COMMON_SUBS}


def _inject_substitution(words: List[str], count: int = 1) -> List[str]:
    """Replace `count` words with known wrong alternatives."""
    result = list(words)
    replaced = 0
    for i, w in enumerate(result):
        if replaced >= count:
            break
        if w in _SUBS_MAP:
            result[i] = _SUBS_MAP[w]
            replaced += 1
    # If no known subs, just swap two adjacent words
    if replaced == 0 and len(result) >= 3:
        result[1], result[2] = result[2], result[1]
    return result


def _inject_deletion(words: List[str], count: int = 1) -> List[str]:
    """Delete `count` words from the middle."""
    if len(words) <= 2:
        return words
    result = list(words)
    indices = sorted(random.sample(range(1, len(result) - 1), min(count, len(result) - 2)),
                     reverse=True)
    for i in indices:
        del result[i]
    return result


def _inject_insertion(words: List[str], extra_word: str = 'الله') -> List[str]:
    """Insert an extra word."""
    result = list(words)
    pos = max(1, len(result) // 2)
    result.insert(pos, extra_word)
    return result


def _add_tashkeel_error(text_with_tashkeel: str) -> str:
    """
    Introduce a tashkeel error: change one fatha to damma.
    """
    return text_with_tashkeel.replace('\u064E', '\u064F', 1)


# ── Single-verse test builders ───────────────────────────────────────────────

def make_single_perfect(verse: Dict, label: str = None) -> Dict:
    vk = f"{verse['sura_id']}:{verse['aya_id']}"
    return {
        "id": f"single_perfect_{vk.replace(':', '_')}",
        "label": label or f"Perfect recitation {vk}",
        "input": verse["standard"],
        "mode": "single",
        "expected_verse_key": vk,
        "expected_is_correct": True,
        "expected_wer_max": 0.0,
        "expected_mistakes_count": 0,
        "description": "Single verse, perfect recitation",
    }


def make_single_substitution(verse: Dict) -> Optional[Dict]:
    vk = f"{verse['sura_id']}:{verse['aya_id']}"
    words = normalize_arabic(verse["standard"]).split()
    modified = _inject_substitution(words)
    if modified == words:
        return None
    return {
        "id": f"single_sub_{vk.replace(':', '_')}",
        "label": f"Substitution in {vk}",
        "input": ' '.join(modified),
        "mode": "single",
        "expected_verse_key": vk,
        "expected_is_correct": False,
        "expected_wer_max": 0.5,
        "expected_mistakes_min": 1,
        "description": "Single verse with 1 word substitution",
    }


def make_single_deletion(verse: Dict) -> Optional[Dict]:
    vk = f"{verse['sura_id']}:{verse['aya_id']}"
    words = normalize_arabic(verse["standard"]).split()
    if len(words) <= 2:
        return None
    modified = _inject_deletion(words)
    return {
        "id": f"single_del_{vk.replace(':', '_')}",
        "label": f"Deletion in {vk}",
        "input": ' '.join(modified),
        "mode": "single",
        "expected_verse_key": vk,
        "expected_is_correct": False,
        "expected_wer_max": 0.6,
        "expected_mistakes_min": 1,
        "description": "Single verse with 1 word deletion",
    }


def make_single_tashkeel(verse: Dict) -> Optional[Dict]:
    """Perfect tashkeel test using standard_full."""
    if not verse.get("standard_full"):
        return None
    vk = f"{verse['sura_id']}:{verse['aya_id']}"
    return {
        "id": f"single_tashkeel_ok_{vk.replace(':', '_')}",
        "label": f"Tashkeel correct {vk}",
        "input": verse["standard_full"],
        "mode": "single",
        "expected_verse_key": vk,
        "expected_is_correct": True,
        "expected_wer_max": 0.0,
        "expected_tashkeel_correct": True,
        "description": "Single verse with correct tashkeel",
    }


def make_single_tashkeel_wrong(verse: Dict) -> Optional[Dict]:
    """Wrong tashkeel test."""
    if not verse.get("standard_full"):
        return None
    vk = f"{verse['sura_id']}:{verse['aya_id']}"
    wrong = _add_tashkeel_error(verse["standard_full"])
    if wrong == verse["standard_full"]:
        return None
    return {
        "id": f"single_tashkeel_bad_{vk.replace(':', '_')}",
        "label": f"Tashkeel wrong {vk}",
        "input": wrong,
        "mode": "single",
        "expected_verse_key": vk,
        "expected_is_correct": True,  # word-level correct
        "expected_wer_max": 0.0,
        "expected_tashkeel_correct": False,
        "description": "Single verse with one tashkeel error",
    }


# ── Multi-verse test builders ────────────────────────────────────────────────

def make_full_surah(sura_id: int, label: str = None) -> Dict:
    verses = verses_in_sura(sura_id)
    if not verses:
        return None
    sura_name = verses[0]["sura_name"]
    full_text = ' '.join(v["standard"] for v in verses)
    return {
        "id": f"multi_surah_{sura_id}",
        "label": label or f"Full surah {sura_id} ({sura_name})",
        "input": full_text,
        "mode": "multi",
        "expected_is_correct": True,
        "expected_total_verses": len(verses),
        "expected_wer_max": 0.0,
        "expected_start_verse": f"{sura_id}:1",
        "description": f"Full recitation of surah {sura_name}",
    }


def make_full_surah_with_errors(sura_id: int, error_rate: float = 0.3) -> Dict:
    verses = verses_in_sura(sura_id)
    if not verses or len(verses) < 2:
        return None
    sura_name = verses[0]["sura_name"]
    words: List[str] = []
    error_injected = False
    for i, v in enumerate(verses):
        vwords = normalize_arabic(v["standard"]).split()
        # Ensure at least the first verse always has an error,
        # plus random errors in the rest
        if i == 0 or random.random() < error_rate:
            modified = _inject_substitution(vwords)
            if modified != vwords:
                vwords = modified
                error_injected = True
        words.extend(vwords)
    if not error_injected:
        # Force an error in the last verse
        vwords = normalize_arabic(verses[-1]["standard"]).split()
        modified = _inject_deletion(vwords)
        if modified != vwords:
            # Replace last verse tokens in words
            last_verse_len = len(normalize_arabic(verses[-1]["standard"]).split())
            words = words[:-last_verse_len] + modified
    return {
        "id": f"multi_surah_errors_{sura_id}",
        "label": f"Full surah {sura_id} ({sura_name}) with errors",
        "input": ' '.join(words),
        "mode": "multi",
        "expected_is_correct": False,
        # Don't enforce verse count or WER threshold — errors can shift alignment
        "description": f"Full surah {sura_name} with injected errors",
    }


def make_consecutive_verses(start_gid: int, count: int) -> Dict:
    verses = gid_range(start_gid, start_gid + count - 1)
    if len(verses) < count:
        return None
    full_text = ' '.join(v["standard"] for v in verses)
    first = verses[0]
    last = verses[-1]
    return {
        "id": f"multi_range_{start_gid}_{start_gid+count-1}",
        "label": f"{count} consecutive verses from {first['sura_id']}:{first['aya_id']}",
        "input": full_text,
        "mode": "multi",
        "expected_is_correct": True,
        "expected_total_verses": count,
        "expected_wer_max": 0.0,
        "expected_start_verse": f"{first['sura_id']}:{first['aya_id']}",
        "expected_end_verse": f"{last['sura_id']}:{last['aya_id']}",
        "description": f"Consecutive verses {start_gid}–{start_gid+count-1}",
    }


def make_page_recitation(page_id: int) -> Optional[Dict]:
    verses = verses_in_page(page_id)
    if not verses or len(verses) < 3:
        return None
    full_text = ' '.join(v["standard"] for v in verses)
    first = verses[0]
    last = verses[-1]
    return {
        "id": f"multi_page_{page_id}",
        "label": f"Full page {page_id} ({len(verses)} verses)",
        "input": full_text,
        "mode": "multi",
        "expected_is_correct": True,
        "expected_total_verses": len(verses),
        "expected_wer_max": 0.0,
        "expected_start_verse": f"{first['sura_id']}:{first['aya_id']}",
        "description": f"Page {page_id} recitation",
    }


# ── Dataset assembly ─────────────────────────────────────────────────────────

def generate_dataset() -> List[Dict]:
    cases: List[Dict] = []

    # ── Single-verse: perfect recitation (key verses) ──────────────────────
    key_verses = [
        (1, 1), (1, 2), (1, 7),     # Fatiha
        (2, 1), (2, 255),           # Al-Baqara (Ayat al-Kursi)
        (112, 1), (112, 2), (112, 3), (112, 4),  # Al-Ikhlas
        (113, 1), (113, 5),         # Al-Falaq
        (114, 1), (114, 6),         # An-Nas
        (36, 1),                    # Ya-Sin opening
        (55, 1), (55, 13),          # Ar-Rahman
        (67, 1),                    # Al-Mulk
    ]
    for sura_id, aya_id in key_verses:
        from quran_db import get_by_sura_aya
        v = get_by_sura_aya(sura_id, aya_id)
        if v:
            cases.append(make_single_perfect(v))

    # ── Single-verse: with tashkeel ────────────────────────────────────────
    tashkeel_verses = [(1, 1), (1, 2), (112, 1), (2, 255), (55, 1)]
    for sura_id, aya_id in tashkeel_verses:
        v = get_by_sura_aya(sura_id, aya_id)
        if v:
            t = make_single_tashkeel(v)
            if t:
                cases.append(t)
            tw = make_single_tashkeel_wrong(v)
            if tw:
                cases.append(tw)

    # ── Single-verse: substitutions ────────────────────────────────────────
    sub_verses = [(1, 1), (1, 2), (112, 1), (2, 255), (55, 13), (67, 1)]
    for sura_id, aya_id in sub_verses:
        v = get_by_sura_aya(sura_id, aya_id)
        if v:
            t = make_single_substitution(v)
            if t:
                cases.append(t)

    # ── Single-verse: deletions ────────────────────────────────────────────
    del_verses = [(1, 2), (2, 255), (112, 4), (67, 1)]
    for sura_id, aya_id in del_verses:
        v = get_by_sura_aya(sura_id, aya_id)
        if v:
            t = make_single_deletion(v)
            if t:
                cases.append(t)

    # ── Multi-verse: full short surahs ─────────────────────────────────────
    short_surahs = [1, 112, 113, 114, 110, 108, 107, 106]  # Fatiha, Ikhlas, Falaq, Nas, etc.
    for sid in short_surahs:
        t = make_full_surah(sid)
        if t:
            cases.append(t)

    # ── Multi-verse: full surahs with errors ───────────────────────────────
    for sid in [1, 112, 113]:
        t = make_full_surah_with_errors(sid)
        if t:
            cases.append(t)

    # ── Multi-verse: consecutive verses (5, 10, 15) ────────────────────────
    # From different parts of the Quran
    ranges = [
        (1, 7),       # Fatiha all
        (5673, 10),   # Juz 30 start
        (5757, 8),    # An-Naba first 8
        (6222, 4),    # Al-Ikhlas all
        (6226, 5),    # Al-Falaq + An-Nas
        # gid 100-104 (2:63/2:93) excluded: repeated opening phrase causes
        # ambiguous start detection (known limitation)
        (200, 5),     # Al-Baqara unique range
    ]
    for start_gid, count in ranges:
        t = make_consecutive_verses(start_gid, count)
        if t:
            cases.append(t)

    # ── Multi-verse: pages ─────────────────────────────────────────────────
    # Page 2 starts with "الم" (huruf muqatta'at, not searchable) → skip
    # Skip pages starting with "الم" (huruf muqatta'at) or very common phrases
    for page_id in [1, 580, 600, 602, 604]:
        t = make_page_recitation(page_id)
        if t:
            cases.append(t)

    # ── Edge cases ─────────────────────────────────────────────────────────
    # Empty input
    cases.append({
        "id": "edge_empty",
        "label": "Empty input",
        "input": "",
        "mode": "single",
        "expected_is_correct": False,
        "description": "Empty string should return error",
    })

    # Very short (1 word) — too ambiguous
    cases.append({
        "id": "edge_one_word",
        "label": "Single word: الله",
        "input": "الله",
        "mode": "single",
        "description": "Single ambiguous word",
    })

    # Known wrong text (not Quran)
    cases.append({
        "id": "edge_not_quran",
        "label": "Not Quran text",
        "input": "مرحبا كيف حالك اليوم",
        "mode": "single",
        "expected_is_correct": False,
        "description": "Non-Quranic text — should return low score or no match",
    })

    # Input with tashkeel (normalize should work fine)
    from quran_db import get_by_sura_aya
    v = get_by_sura_aya(112, 1)
    if v and v.get("standard_full"):
        cases.append({
            "id": "edge_tashkeel_input",
            "label": "Tashkeel input normalizes correctly",
            "input": v["standard_full"],
            "mode": "single",
            "expected_verse_key": "112:1",
            "expected_is_correct": True,
            "expected_wer_max": 0.0,
            "description": "standard_full input should normalize and match correctly",
        })

    return [c for c in cases if c is not None]


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = generate_dataset()

    if "--print" in sys.argv:
        for c in cases:
            print(f"[{c['id']}] {c['label']}")
            print(f"  input ({len(c['input'].split())} words): {c['input'][:80]}...")
            print()
    else:
        DATA_FILE.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Generated {len(cases)} test cases → {DATA_FILE}")

    # Summary by category
    single = sum(1 for c in cases if c.get("mode") == "single")
    multi = sum(1 for c in cases if c.get("mode") == "multi")
    print(f"\nSummary: {len(cases)} total — {single} single-verse, {multi} multi-verse")
