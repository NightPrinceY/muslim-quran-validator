"""
Multi-verse recitation validator.

Handles recitation of:
  - Full surah (e.g., Al-Fatiha 1:1–1:7)
  - Full juz (e.g., Juz 30)
  - Consecutive pages or arbitrary consecutive verse ranges
  - Any sequence of ≥2 consecutive verses

Algorithm:
  1. Detect multi-verse input (word count suggests > 1 verse)
  2. Find the starting verse (first 6–8 words)
  3. Forward greedy alignment: for each consecutive GID, consume
     the best-matching slice of remaining user tokens
  4. Per-verse word_diff → mistakes, WER
  5. Aggregate: total WER, global is_correct, per-verse breakdown

Boundary detection strategy:
  Given user tokens T and reference words R for a verse:
  - Take a window W = T[pos : pos + max(len(R) * 2, len(R) + 10)]
  - Run SequenceMatcher(W, R) to get opcodes
  - Find the "consumed" position: the highest i2 in any opcode that
    accounts for at least one reference word (j2 > 0)
  - If match ratio < MIN_VERSE_RATIO, stop — user has ended recitation
"""
import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from utils.normalizer import normalize_arabic, has_tashkeel
from utils.quran_db import get as db_get, gid_range
from utils.quran_search import find_best_verse
from utils.tashkeel import validate_tashkeel

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Minimum SequenceMatcher ratio to accept a verse boundary as valid
MIN_VERSE_RATIO = 0.25

# Average Quran verse length in words (used for multi-verse detection)
AVG_VERSE_WORDS = 12.5

# Word count above which we consider multi-verse possible.
# Set to 8 — even a very short surah like Al-Kawthar (3 verses, ~11 words)
# or Al-Ikhlas (4 verses, ~15 words) will trigger multi-verse mode.
# If multi-verse finds only 1 verse, validator_mcp.py falls back to single.
MULTI_VERSE_THRESHOLD = 8

# How many first words to use for start-verse detection
START_PROBE_WORDS = 7

# Max verses to try aligning (safety limit)
MAX_VERSES = 120


# ── Word-level diff (same as validator_mcp but here for independence) ────────

def _word_diff(recited: List[str], reference: List[str]) -> Dict:
    sm = SequenceMatcher(None, recited, reference, autojunk=False)
    opcodes = sm.get_opcodes()

    mistakes: List[Dict] = []
    subs = dels = ins = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            continue
        elif tag == 'replace':
            r_chunk = recited[i1:i2]
            e_chunk = reference[j1:j2]
            for k in range(max(len(r_chunk), len(e_chunk))):
                rw = r_chunk[k] if k < len(r_chunk) else None
                ew = e_chunk[k] if k < len(e_chunk) else None
                if rw and ew:
                    mistakes.append({"type": "substitution", "position": j1+k+1,
                                     "recited": rw, "expected": ew})
                    subs += 1
                elif ew:
                    mistakes.append({"type": "deletion", "position": j1+k+1, "expected": ew})
                    dels += 1
                elif rw:
                    mistakes.append({"type": "insertion", "position": i1+k+1, "recited": rw})
                    ins += 1
        elif tag == 'delete':
            for k, w in enumerate(recited[i1:i2]):
                mistakes.append({"type": "insertion", "position": i1+k+1, "recited": w})
                ins += 1
        elif tag == 'insert':
            for k, w in enumerate(reference[j1:j2]):
                mistakes.append({"type": "deletion", "position": j1+k+1, "expected": w})
                dels += 1

    n_ref = len(reference)
    wer = (subs + dels + ins) / max(n_ref, 1)
    return {
        "mistakes": mistakes,
        "wer": round(wer, 4),
        "matched_ratio": round(sm.ratio(), 4),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "recited_word_count": len(recited),
        "reference_word_count": n_ref,
    }


# ── Boundary detection ────────────────────────────────────────────────────────

def _find_verse_boundary(
    user_tokens: List[str],
    start_pos: int,
    ref_words: List[str],
) -> Tuple[int, float]:
    """
    Determine how many tokens (starting at start_pos) belong to this verse.

    Uses a direct word-by-word scanner to avoid SequenceMatcher's tendency
    to "skip ahead" to a later match when there are substitutions.

    Algorithm:
      For each reference word r[j], find the best matching user token in a
      small look-ahead window (insertion tolerance). Advance user position
      after each reference word — either aligned to the matched user token
      or left in place (deletion, user skipped the ref word).

    Returns (end_pos_exclusive, match_ratio).
    """
    remaining = user_tokens[start_pos:]
    if not remaining:
        return start_pos, 0.0

    n_ref = len(ref_words)

    # Maximum tokens we will look ahead per reference word
    # (handles insertions the user added that aren't in the verse)
    MAX_INSERTION_LOOKAHEAD = 2

    user_pos = 0          # position within `remaining`
    matched_words = 0     # count of exact matches

    for ref_word in ref_words:
        # Search for ref_word in remaining[user_pos : user_pos + 1 + lookahead]
        found = False
        for lookahead in range(MAX_INSERTION_LOOKAHEAD + 1):
            idx = user_pos + lookahead
            if idx >= len(remaining):
                break
            if remaining[idx] == ref_word:
                # Exact match found (possibly after skipping inserted words)
                user_pos = idx + 1
                matched_words += 1
                found = True
                break

        if not found:
            # No exact match in window — substitution or deletion.
            # For substitution: consume the next user token (it's "for" this ref word)
            # For deletion: don't advance (ref word was skipped entirely)
            if user_pos < len(remaining):
                # Assume substitution — consume 1 user token for this ref word
                user_pos += 1
            # else: deletion at end of input, user_pos stays

    cut = start_pos + user_pos

    # Compute match ratio for quality check
    total = n_ref + max(0, user_pos - matched_words)  # avoid div by zero
    ratio = (2.0 * matched_words) / max(total, 1)

    return cut, ratio


# ── Start verse detection ────────────────────────────────────────────────────

def _find_start_verse(tokens: List[str]) -> Optional[Dict]:
    """
    Identify the starting verse using a short prefix probe.

    Key insight: using too many tokens (>5) spans multiple verses and
    confuses the search (no single verse contains tokens from two verses).
    We use the shortest probe that reliably identifies the first verse
    (typically 4–5 tokens = the exact length of the first verse).

    We also validate: the found verse must plausibly be a start — we
    prefer verses whose word set overlaps with our first few tokens.
    """
    # Try progressively shorter probes; stop at the first confident hit
    for probe_len in [4, 5, 3, 6]:
        probe = ' '.join(tokens[:min(probe_len, len(tokens))])
        verse = find_best_verse(probe, min_score=0.4)
        if verse:
            # Quick sanity check: at least 2 of our first 6 tokens should
            # appear as words in the identified verse
            verse_norm = normalize_arabic(verse.get("standard", ""))
            verse_words = set(verse_norm.split())
            probe_tokens = set(tokens[:6])
            overlap = len(probe_tokens & verse_words)
            if overlap >= 2:
                return verse
    # Last resort: lowest bar
    for probe_len in [4, 5, 3]:
        probe = ' '.join(tokens[:min(probe_len, len(tokens))])
        verse = find_best_verse(probe, min_score=0.2)
        if verse:
            return verse
    return None


# ── Forward alignment ────────────────────────────────────────────────────────

def _forward_align(
    tokens: List[str],
    start_gid: int,
) -> List[Dict]:
    """
    Greedily align tokens to consecutive verses starting at start_gid.

    Returns list of per-verse result dicts.
    """
    pos = 0
    gid = start_gid
    verse_results: List[Dict] = []
    stop_streak = 0  # consecutive verses that failed to match

    while pos < len(tokens) and gid <= 6236 and len(verse_results) < MAX_VERSES:
        verse = db_get(gid)
        if not verse:
            break

        ref_norm = normalize_arabic(verse["standard"])
        ref_words = [w for w in ref_norm.split() if w]
        if not ref_words:
            gid += 1
            continue

        cut, ratio = _find_verse_boundary(tokens, pos, ref_words)

        if ratio < MIN_VERSE_RATIO:
            # This verse doesn't match — might be end of recitation
            stop_streak += 1
            if stop_streak >= 2:
                break
            gid += 1
            continue

        stop_streak = 0
        verse_tokens = tokens[pos:cut] if cut > pos else []

        # Even if user provided zero tokens for this verse (deletion), record it
        diff = _word_diff(verse_tokens, ref_words)

        verse_results.append({
            "gid": gid,
            "verse_key": f"{verse['sura_id']}:{verse['aya_id']}",
            "surah_name": verse["sura_name"],
            "matched_verse": verse.get("uthmani", verse.get("standard", "")),
            "standard": verse["standard"],
            "standard_full": verse.get("standard_full", ""),
            "diff": diff,
            "is_correct": diff["wer"] == 0.0,
            "wer": diff["wer"],
            "corrections": diff["mistakes"],
            "recited_tokens": verse_tokens,
        })

        pos = cut if cut > pos else pos + 1
        gid += 1

    return verse_results


# ── Aggregate result builder ─────────────────────────────────────────────────

def _aggregate(verse_results: List[Dict], original_text: str) -> Dict:
    """Build the top-level multi-verse result from per-verse results."""
    if not verse_results:
        return {
            "mode": "multi",
            "is_correct": False,
            "total_verses": 0,
            "total_wer": 1.0,
            "feedback": "لم أتمكن من التعرف على الآيات المتلوة.",
            "verses": [],
        }

    total_ref_words = sum(v["diff"]["reference_word_count"] for v in verse_results)
    total_errors = sum(
        v["diff"]["substitutions"] + v["diff"]["deletions"] + v["diff"]["insertions"]
        for v in verse_results
    )
    total_wer = round(total_errors / max(total_ref_words, 1), 4)
    is_correct = all(v["is_correct"] for v in verse_results)
    n_correct = sum(1 for v in verse_results if v["is_correct"])
    n_total = len(verse_results)

    # Summary range
    first = verse_results[0]
    last = verse_results[-1]
    range_desc = (
        f"{first['surah_name']} {first['verse_key']}"
        if n_total == 1
        else f"من {first['surah_name']} ({first['verse_key']}) إلى ({last['verse_key']})"
    )

    # Build feedback
    if is_correct:
        feedback = f"أحسنت! تلاوتك صحيحة تماماً. {range_desc} — {n_total} آية."
    elif total_wer <= 0.10:
        feedback = (f"تلاوتك ممتازة تقريباً ({n_correct}/{n_total} آيات صحيحة). "
                    f"WER={total_wer:.1%}. {range_desc}")
    elif total_wer <= 0.30:
        feedback = (f"تلاوتك قريبة مع بعض الأخطاء ({n_correct}/{n_total} آيات صحيحة). "
                    f"WER={total_wer:.1%}. {range_desc}")
    else:
        feedback = (f"تلاوتك تحتاج مراجعة ({n_correct}/{n_total} آيات صحيحة). "
                    f"WER={total_wer:.1%}. {range_desc}")

    # Per-verse clean output (drop internal diff key)
    verses_out = []
    for v in verse_results:
        verses_out.append({
            "verse_key": v["verse_key"],
            "surah_name": v["surah_name"],
            "matched_verse": v["matched_verse"],
            "is_correct": v["is_correct"],
            "wer": v["wer"],
            "corrections": v["corrections"],
        })

    return {
        "mode": "multi",
        "is_correct": is_correct,
        "total_verses": n_total,
        "correct_verses": n_correct,
        "total_wer": total_wer,
        "feedback": feedback,
        "verses": verses_out,
        "range": range_desc,
        "start_verse": first["verse_key"],
        "end_verse": last["verse_key"],
        "start_surah": first["surah_name"],
    }


# ── Detection heuristic ───────────────────────────────────────────────────────

def looks_like_multi_verse(text: str) -> bool:
    """
    Quick heuristic: does this text likely span multiple verses?

    True if word count > MULTI_VERSE_THRESHOLD (default 8).
    The caller can force single or multi mode, but this provides a default.
    """
    tokens = [w for w in normalize_arabic(text).split() if w]
    return len(tokens) > MULTI_VERSE_THRESHOLD


# ── Main entry point ─────────────────────────────────────────────────────────

def validate_multi_verse(text: str) -> Dict:
    """
    Validate a multi-verse recitation.

    1. Normalize text
    2. Find starting verse
    3. Forward align
    4. Validate tashkeel if user provided harakat
    5. Return aggregated result
    """
    norm = normalize_arabic(text)
    tokens = [w for w in norm.split() if w]

    if not tokens:
        return {
            "mode": "multi",
            "is_correct": False,
            "total_verses": 0,
            "total_wer": 1.0,
            "feedback": "النص فارغ.",
            "verses": [],
        }

    # Find starting verse
    start_verse = _find_start_verse(tokens)
    if not start_verse:
        return {
            "mode": "multi",
            "is_correct": False,
            "total_verses": 0,
            "total_wer": 1.0,
            "feedback": "لم أتمكن من التعرف على بداية التلاوة.",
            "verses": [],
        }

    start_gid = start_verse["gid"]
    logger.info("Multi-verse: start_gid=%d (%s:%s), tokens=%d",
                start_gid, start_verse.get("sura_id"), start_verse.get("aya_id"), len(tokens))

    # Forward alignment
    verse_results = _forward_align(tokens, start_gid)

    # Aggregate
    result = _aggregate(verse_results, text)

    # Tashkeel validation (per verse if user provided harakat)
    if has_tashkeel(text) and verse_results:
        tashkeel_errors: List[Dict] = []
        for i, vr in enumerate(verse_results):
            # Find corresponding recited text for this verse
            recited_slice = ' '.join(vr["recited_tokens"])
            tashkeel_check = validate_tashkeel(recited_slice, vr["standard_full"])
            if tashkeel_check.get("has_tashkeel") and not tashkeel_check["is_correct"]:
                tashkeel_errors.append({
                    "verse_key": vr["verse_key"],
                    "errors": tashkeel_check["errors"],
                })
                # Inject tashkeel errors into verse result
                result["verses"][i]["tashkeel_errors"] = tashkeel_check["errors"]
                result["verses"][i]["tashkeel_correct"] = False
        result["tashkeel_errors"] = tashkeel_errors
        result["has_tashkeel"] = True
    else:
        result["has_tashkeel"] = False

    return result
