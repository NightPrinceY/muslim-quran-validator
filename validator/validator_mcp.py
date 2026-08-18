"""
MCP Tool: Quran Recitation Validator
Pure algorithmic — no LLM required.

Supports:
  - Single verse recitation (default for short input ≤ MULTI_VERSE_THRESHOLD words)
  - Multi-verse recitation (full surah / juz / page / consecutive verses)
  - Tashkeel (harakat) validation when user provides diacritics

Pipeline (single verse):
  1. Identify verse via 4-layer search (exact/lemma/root/fuzzy)
  2. Word-level diff (SequenceMatcher) between recited and reference
  3. Compute WER
  4. Optional tashkeel validation

Pipeline (multi-verse):
  1. Find starting verse (first ~7 tokens)
  2. Forward greedy alignment across consecutive verses
  3. Per-verse word diff + WER
  4. Aggregate total WER, per-verse breakdown
  5. Optional tashkeel validation per verse
"""
import logging
from difflib import SequenceMatcher
from typing import Dict, List

from utils.normalizer import normalize_arabic, has_tashkeel
from utils.quran_search import find_best_verse
from utils.multi_verse import validate_multi_verse, looks_like_multi_verse
from utils.tashkeel import validate_tashkeel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Word-Level Diff ───────────────────────────────────────────────────────────

def _word_diff(recited_words: List[str], reference_words: List[str]) -> Dict:
    """
    Align recited words against reference words using SequenceMatcher opcodes.

    Returns:
      mistakes:       list of {type, position, recited?, expected?}
      wer:            Word Error Rate = (subs + dels + ins) / len(reference)
      matched_ratio:  fraction of matching words
    """
    sm = SequenceMatcher(None, recited_words, reference_words, autojunk=False)
    opcodes = sm.get_opcodes()

    mistakes: List[Dict] = []
    subs = dels = ins = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            continue
        elif tag == 'replace':
            r_chunk = recited_words[i1:i2]
            e_chunk = reference_words[j1:j2]
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
            for k, w in enumerate(recited_words[i1:i2]):
                mistakes.append({"type": "insertion", "position": i1+k+1, "recited": w})
                ins += 1
        elif tag == 'insert':
            for k, w in enumerate(reference_words[j1:j2]):
                mistakes.append({"type": "deletion", "position": j1+k+1, "expected": w})
                dels += 1

    n_ref = len(reference_words)
    wer = (subs + dels + ins) / max(n_ref, 1)
    return {
        "mistakes": mistakes,
        "wer": round(wer, 4),
        "matched_ratio": round(sm.ratio(), 4),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "recited_word_count": len(recited_words),
        "reference_word_count": n_ref,
    }


# ─── Arabic Feedback Templates ─────────────────────────────────────────────────

def _build_feedback(diff: Dict, verse: Dict) -> str:
    wer = diff["wer"]
    mistakes = diff["mistakes"]
    surah = verse.get("sura_name", "")
    verse_key = f"{verse.get('sura_id', '')}:{verse.get('aya_id', '')}"
    uthmani = verse.get("uthmani", "")

    lines: List[str] = []

    if wer == 0.0:
        lines.append(f"مُمْتَاز! تلاوتك صحيحة تماماً. هذه الآية {verse_key} من سورة {surah}.")
        return lines[0]

    if wer <= 0.10:
        lines.append("تلاوتك ممتازة تقريباً، مع خطأ بسيط جداً.")
    elif wer <= 0.30:
        lines.append("تلاوتك قريبة من الصحيح مع بعض الأخطاء.")
    elif wer <= 0.60:
        lines.append("تلاوتك تحتاج إلى مراجعة في عدة مواضع.")
    else:
        lines.append("تلاوتك تختلف كثيراً عن الآية الصحيحة.")

    lines.append(f"الآية {verse_key} من سورة {surah}.")

    shown = 0
    for m in mistakes:
        if shown >= 6:
            remaining = len(mistakes) - shown
            if remaining > 0:
                lines.append(f"وهناك {remaining} أخطاء أخرى.")
            break
        if m["type"] == "substitution":
            lines.append(f"• قلتَ «{m['recited']}» والصواب «{m['expected']}».")
        elif m["type"] == "deletion":
            lines.append(f"• كلمة «{m['expected']}» مفقودة من تلاوتك.")
        elif m["type"] == "insertion":
            lines.append(f"• كلمة «{m['recited']}» زيادة ليست في الآية.")
        shown += 1

    if uthmani:
        lines.append(f"الآية الكاملة: {uthmani}")

    return "\n".join(lines)


# ─── Single-Verse Validator ───────────────────────────────────────────────────

async def _validate_single(text: str) -> Dict:
    """Validate a single-verse recitation."""
    verse = find_best_verse(text)
    if not verse:
        return _empty_result("لم أتمكن من التعرف على الآية المتلوة.")

    recited_norm = normalize_arabic(text)
    reference_norm = normalize_arabic(verse.get("standard", ""))
    recited_words = [w for w in recited_norm.split() if w]
    reference_words = [w for w in reference_norm.split() if w]

    diff = _word_diff(recited_words, reference_words)
    feedback = _build_feedback(diff, verse)
    verse_key = f"{verse.get('sura_id', '')}:{verse.get('aya_id', '')}"

    result = {
        "mode": "single",
        "is_correct": diff["wer"] == 0.0,
        "feedback": feedback,
        "corrections": diff["mistakes"],
        "matched_verse": verse.get("uthmani", verse.get("standard", "")),
        "surah_name": verse.get("sura_name", ""),
        "verse_key": verse_key,
        "wer": diff["wer"],
        "matched_ratio": diff["matched_ratio"],
        "match_type": verse.get("_match_type", ""),
        "match_score": verse.get("_score", 0.0),
    }

    # Tashkeel validation if user provided harakat
    if has_tashkeel(text):
        std_full = verse.get("standard_full", "")
        tashkeel = validate_tashkeel(text, std_full)
        result["tashkeel"] = tashkeel
        # Tashkeel errors don't affect is_correct (word-level), but add them to feedback
        if tashkeel.get("has_tashkeel") and not tashkeel["is_correct"]:
            n_err = tashkeel["error_count"]
            result["feedback"] += f"\nكما يوجد {n_err} خطأ في الحركات (التشكيل)."

    return result


# ─── Main Entry Point ─────────────────────────────────────────────────────────

async def validate_recitation(text: str) -> Dict:
    """
    Validate a Quran recitation — automatically routes to single or multi-verse.

    Single-verse (≤ 8 words by default):
      Returns: mode, is_correct, feedback, corrections, matched_verse,
               surah_name, verse_key, wer, matched_ratio, match_type, match_score,
               tashkeel? (if harakat provided)

    Multi-verse (> 8 words):
      Returns: mode, is_correct, total_verses, correct_verses, total_wer,
               feedback, verses[], range, start_verse, end_verse, start_surah,
               tashkeel_errors? (if harakat provided)
    """
    if not text or not text.strip():
        return _empty_result("عَفْواً، النَّصُ فَارِغٌ.")

    try:
        if looks_like_multi_verse(text):
            logger.info("Multi-verse mode: %d tokens", len(text.split()))
            result = validate_multi_verse(text)
            # If multi-verse found only 1 verse, enrich with search metadata
            if result.get("total_verses", 0) <= 1:
                # Try single-verse for better metadata
                single = await _validate_single(text)
                if single.get("verse_key"):
                    return single
            return result
        else:
            return await _validate_single(text)

    except Exception as e:
        logger.error("Validation error: %s", e, exc_info=True)
        return _empty_result("عَفْواً، حَدَثَ خَطَأٌ. يَرْجَى الْمُحَاوَلَةُ مَرَّةً أُخْرَى.")


def _empty_result(feedback: str) -> Dict:
    return {
        "mode": "single",
        "is_correct": False,
        "feedback": feedback,
        "corrections": [],
        "matched_verse": "",
        "surah_name": "",
        "verse_key": "",
        "wer": 1.0,
        "matched_ratio": 0.0,
        "match_type": "",
        "match_score": 0.0,
    }
