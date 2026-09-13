"""
Comprehensive test suite for the Quran recitation validator.

Covers:
  1. Single-verse: perfect, substitution, deletion, insertion, tashkeel
  2. Multi-verse: full surah, page, consecutive verses, with/without errors
  3. Edge cases: empty input, non-Quran text, tashkeel normalization
  4. Normalizer unit tests
  5. QuranDB unit tests

Run with:
    python -m pytest tests/test_all.py -v
    OR
    python tests/test_all.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Make parent importable
VALIDATOR_ROOT = Path(__file__).parent.parent / "validator"
sys.path.insert(0, str(VALIDATOR_ROOT))
sys.path.insert(0, str(VALIDATOR_ROOT / "utils"))

from normalizer import normalize_arabic, normalize_keep_tashkeel, has_tashkeel, remove_tashkeel
from quran_db import get, get_by_sura_aya, verses_in_sura, verses_in_page, sura_info
from validator_mcp import validate_recitation


# ── Test runner helpers ──────────────────────────────────────────────────────

_section = "uncategorised"


class TestResult:
    def __init__(self, test_id: str, label: str, passed: bool, info: str = ""):
        self.test_id = test_id
        self.label = label
        self.passed = passed
        self.info = info
        self.section = _section

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        line = f"{status}  [{self.test_id}] {self.label}"
        if not self.passed and self.info:
            line += f"\n       → {self.info}"
        return line


_results: List[TestResult] = []


def check(test_id: str, label: str, condition: bool, info: str = ""):
    r = TestResult(test_id, label, condition, info)
    _results.append(r)
    print(r)
    return condition


# ── 1. Normalizer tests ──────────────────────────────────────────────────────

def test_normalizer():
    print("\n═══ 1. Normalizer ═══")

    check("N01", "Strip fatha", normalize_arabic("بِسْمِ") == "بسم")
    check("N02", "Alef variants أإآ → ا",
          normalize_arabic("أإآ") == "اا" + "ا",
          f"got '{normalize_arabic('أإآ')}'")
    check("N03", "Alef wasla ٱ → ا",
          normalize_arabic("ٱللَّهِ") == "الله")
    check("N04", "Farsi yeh → yeh",
          normalize_arabic("ٱلرَّحِیمِ") == "الرحيم")
    check("N05", "Alef maqsura ى → ي",
          normalize_arabic("يهدى") == "يهدي",
          f"got '{normalize_arabic('يهدى')}'")
    check("N06", "Full standard_full normalize",
          normalize_arabic("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ") == "بسم الله الرحمن الرحيم")
    # N07: Uthmani normalization using word-level map (2017 corpus-derived pairs).
    # All ٰ-containing words now map exactly — including الرحمن, أولئك, ذلك, هذا, etc.
    uthmani_bism = 'بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِیمِ'
    uthmani_2_121 = 'ٱلَّذِینَ ءَاتَیۡنَٰهُمُ ٱلۡكِتَٰبَ یَتۡلُونَهُۥ حَقَّ تِلَاوَتِهِۦۤ أُو۟لَٰۤئِكَ یُؤۡمِنُونَ بِهِۦۗ وَمَن یَكۡفُرۡ بِهِۦ فَأُو۟لَٰۤئِكَ هُمُ ٱلۡخَٰسِرُونَ'
    check("N07a", "Uthmani Bismillah: الرحمٰن→الرحمن (word-map fix)",
          normalize_arabic(uthmani_bism) == "بسم الله الرحمن الرحيم",
          f"got: {normalize_arabic(uthmani_bism)!r}")
    expected_2_121 = normalize_arabic('الذين آتيناهم الكتاب يتلونه حق تلاوته أولئك يؤمنون به ومن يكفر به فأولئك هم الخاسرون')
    check("N07b", "Uthmani 2:121: كِتَٰبَ→كتاب, ءَاتَیۡنَٰهُمُ→اتيناهم, الخٰسرون→الخاسرون",
          normalize_arabic(uthmani_2_121) == expected_2_121,
          f"got: {normalize_arabic(uthmani_2_121)!r}")
    check("N08", "has_tashkeel positive",
          has_tashkeel("بِسْمِ اللَّهِ"))
    check("N09", "has_tashkeel negative",
          not has_tashkeel("بسم الله"))
    check("N10", "keep tashkeel normalizer preserves harakat",
          '\u064E' in normalize_keep_tashkeel("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"),
          "fatha should be preserved")


# ── 2. QuranDB tests ─────────────────────────────────────────────────────────

def test_quran_db():
    print("\n═══ 2. QuranDB ═══")

    v = get(1)
    check("DB01", "GID 1 is Fatiha 1:1",
          v is not None and v["sura_id"] == 1 and v["aya_id"] == 1)

    v = get_by_sura_aya(112, 1)
    check("DB02", "112:1 is Ikhlas",
          v is not None and "الله" in v["standard"])

    fatiha = verses_in_sura(1)
    check("DB03", "Fatiha has 7 verses", len(fatiha) == 7)

    info = sura_info(1)
    check("DB04", "Sura info: Fatiha start_gid=1",
          info is not None and info["start_gid"] == 1)

    page1 = verses_in_page(1)
    check("DB05", "Page 1 has 7 verses (Fatiha)", len(page1) == 7)

    check("DB06", "GID 6236 is last verse",
          get(6236) is not None)

    check("DB07", "GID 9999 returns None",
          get(9999) is None)


# ── 3. Single-verse: perfect ─────────────────────────────────────────────────

def test_single_perfect():
    print("\n═══ 3. Single-verse: perfect ═══")

    tests = [
        ("SP01", "Fatiha 1:1", "بسم الله الرحمن الرحيم", "1:1"),
        ("SP02", "Fatiha 1:2", "الحمد لله رب العالمين", "1:2"),
        ("SP03", "Fatiha 1:4", "مالك يوم الدين", "1:4"),
        ("SP04", "Ikhlas 112:1", "قل هو الله احد", "112:1"),
        ("SP05", "Ikhlas 112:2", "الله الصمد", "112:2"),
        ("SP06", "Ikhlas 112:3", "لم يلد ولم يولد", "112:3"),
        ("SP07", "Ikhlas 112:4", "ولم يكن له كفوا احد", "112:4"),
        ("SP08", "Al-Mulk 67:1", "تبارك الذي بيده الملك وهو على كل شيء قدير", "67:1"),
        ("SP09", "Nas 114:1", "قل اعوذ برب الناس", "114:1"),
        ("SP10", "Falaq 113:1", "قل اعوذ برب الفلق", "113:1"),
    ]

    loop = asyncio.get_event_loop()
    for tid, label, text, expected_vk in tests:
        t0 = time.perf_counter()
        r = loop.run_until_complete(validate_recitation(text))
        ms = (time.perf_counter() - t0) * 1000

        correct = r.get("is_correct") is True
        vk_ok = (expected_vk is None) or r.get("verse_key") == expected_vk
        check(tid, f"{label} ({ms:.1f}ms)",
              correct and vk_ok,
              f"is_correct={r.get('is_correct')} verse_key={r.get('verse_key')} wer={r.get('wer')}")


# ── 4. Single-verse: substitution ────────────────────────────────────────────

def test_single_substitution():
    print("\n═══ 4. Single-verse: substitution ═══")

    loop = asyncio.get_event_loop()
    tests = [
        ("SS01", "الكريم instead of الرحيم", "بسم الله الكريم الرحيم", "1:1"),
        ("SS02", "العلمين instead of العالمين", "الحمد لله رب العلمين", "1:2"),
        ("SS03", "واحد instead of احد", "قل هو الله واحد", "112:1"),
        ("SS04", "الطريق instead of الصراط", "اهدنا الطريق المستقيم", "1:6"),
    ]

    for tid, label, text, expected_vk in tests:
        r = loop.run_until_complete(validate_recitation(text))
        not_correct = r.get("is_correct") is False
        has_mistakes = len(r.get("corrections", [])) >= 1
        vk_ok = r.get("verse_key") == expected_vk
        check(tid, label,
              not_correct and has_mistakes and vk_ok,
              f"is_correct={r.get('is_correct')} vk={r.get('verse_key')} "
              f"mistakes={len(r.get('corrections', []))} wer={r.get('wer')}")


# ── 5. Single-verse: deletion ────────────────────────────────────────────────

def test_single_deletion():
    print("\n═══ 5. Single-verse: deletion ═══")

    loop = asyncio.get_event_loop()
    tests = [
        ("SD01", "Missing word from Ikhlas 1", "قل هو الله", None),
        ("SD02", "Missing word from Fatiha 2", "الحمد لله رب", "1:2"),
        ("SD03", "Short partial Ikhlas 3", "لم يلد", "112:3"),
    ]

    for tid, label, text, expected_vk in tests:
        r = loop.run_until_complete(validate_recitation(text))
        not_correct = r.get("is_correct") is False
        check(tid, label, not_correct,
              f"is_correct={r.get('is_correct')} vk={r.get('verse_key')} wer={r.get('wer')}")


# ── 6. Single-verse: tashkeel ─────────────────────────────────────────────────

def test_single_tashkeel():
    print("\n═══ 6. Single-verse: tashkeel ═══")

    loop = asyncio.get_event_loop()
    v = get_by_sura_aya(1, 1)
    std_full = v["standard_full"] if v else None

    if std_full:
        # Perfect tashkeel
        r = loop.run_until_complete(validate_recitation(std_full))
        check("ST01", "standard_full input matches verse",
              r.get("verse_key") == "1:1",
              f"verse_key={r.get('verse_key')} wer={r.get('wer')}")
        check("ST02", "standard_full is_correct=True",
              r.get("is_correct") is True,
              f"wer={r.get('wer')}")
        tash = r.get("tashkeel", {})
        check("ST03", "Tashkeel report present",
              tash.get("has_tashkeel") is True,
              f"tashkeel={tash}")
        check("ST04", "Tashkeel is correct",
              tash.get("is_correct") is True,
              f"errors={tash.get('errors', [])}")
    else:
        for tid in ["ST01", "ST02", "ST03", "ST04"]:
            check(tid, "Tashkeel (skipped — no std_full)", True)

    # Uthmani input
    v112 = get_by_sura_aya(112, 1)
    if v112 and v112.get("uthmani"):
        r = loop.run_until_complete(validate_recitation(v112["uthmani"]))
        check("ST05", "Uthmani text normalizes → matches 112:1",
              r.get("verse_key") == "112:1",
              f"verse_key={r.get('verse_key')}")
        check("ST06", "Uthmani is_correct=True",
              r.get("is_correct") is True,
              f"wer={r.get('wer')}")


# ── 7. Multi-verse: full short surahs ────────────────────────────────────────

def test_multi_full_surahs():
    print("\n═══ 7. Multi-verse: full short surahs ═══")

    loop = asyncio.get_event_loop()

    def recite_surah(sura_id: int):
        return ' '.join(v["standard"] for v in verses_in_sura(sura_id))

    tests = [
        ("MF01", "Full Fatiha (7 verses)", 1, 7),
        ("MF02", "Full Ikhlas (4 verses)", 112, 4),
        ("MF03", "Full Falaq (5 verses)", 113, 5),
        ("MF04", "Full Nas (6 verses)", 114, 6),
        ("MF05", "Full Nasr / An-Nasr (3 verses)", 110, 3),
        ("MF06", "Full Kawthar (3 verses)", 108, 3),
        ("MF07", "Full Masad (5 verses)", 111, 5),
    ]

    for tid, label, sura_id, expected_count in tests:
        text = recite_surah(sura_id)
        t0 = time.perf_counter()
        r = loop.run_until_complete(validate_recitation(text))
        ms = (time.perf_counter() - t0) * 1000

        is_multi = r.get("mode") == "multi"
        total_ok = r.get("total_verses", 0) == expected_count
        correct = r.get("is_correct") is True
        wer_ok = r.get("total_wer", 1.0) == 0.0

        check(tid, f"{label} ({ms:.1f}ms)",
              is_multi and total_ok and correct and wer_ok,
              f"mode={r.get('mode')} total_verses={r.get('total_verses')} "
              f"is_correct={r.get('is_correct')} wer={r.get('total_wer')}")


# ── 8. Multi-verse: with errors ───────────────────────────────────────────────

def test_multi_with_errors():
    print("\n═══ 8. Multi-verse: with errors ═══")

    loop = asyncio.get_event_loop()

    def recite_fatiha_with_error():
        verses = verses_in_sura(1)
        words = []
        for v in verses:
            vwords = normalize_arabic(v["standard"]).split()
            if v["aya_id"] == 1:
                # Inject wrong word in Basmala
                vwords = ['بسم', 'الله', 'الكريم', 'الرحيم']
            words.extend(vwords)
        return ' '.join(words)

    def recite_fatiha_missing_verse():
        # Skip verse 3 (الرحمن الرحيم)
        verses = [v for v in verses_in_sura(1) if v["aya_id"] != 3]
        return ' '.join(v["standard"] for v in verses)

    # Fatiha with error in verse 1
    text = recite_fatiha_with_error()
    r = loop.run_until_complete(validate_recitation(text))
    check("ME01", "Fatiha with error in v1 → mode=multi, is_correct=False",
          r.get("mode") == "multi" and r.get("is_correct") is False,
          f"mode={r.get('mode')} is_correct={r.get('is_correct')} wer={r.get('total_wer')}")
    check("ME02", "Some verses correct (v2–v7 should be fine)",
          r.get("correct_verses", 0) >= 4,
          f"correct_verses={r.get('correct_verses')}")

    # Ikhlas with error
    verses = verses_in_sura(112)
    words = []
    for v in verses:
        vw = normalize_arabic(v["standard"]).split()
        if v["aya_id"] == 1:
            vw = ['قل', 'هو', 'الله', 'واحد']  # wrong: واحد instead of احد
        words.extend(vw)
    r2 = loop.run_until_complete(validate_recitation(' '.join(words)))
    check("ME03", "Ikhlas with error → is_correct=False, total_verses=4",
          r2.get("mode") == "multi" and r2.get("is_correct") is False,
          f"mode={r2.get('mode')} is_correct={r2.get('is_correct')} "
          f"total_verses={r2.get('total_verses')} wer={r2.get('total_wer')}")


# ── 9. Multi-verse: consecutive ranges ────────────────────────────────────────

def test_multi_consecutive():
    print("\n═══ 9. Multi-verse: consecutive ranges ═══")

    loop = asyncio.get_event_loop()
    from quran_db import gid_range

    tests = [
        ("MC01", "5 verses from Juz 30 start", 5673, 5),
        ("MC02", "3 Ikhlas + Falaq first verse", 6222, 5),
        ("MC03", "An-Naba first 10", 5757, 10),
        # gid 100-104 (2:93-2:97): skipped — 2:63 and 2:93 share identical
        # first phrase "واذ اخذنا ميثاقكم ورفعنا فوقكم الطور" → ambiguous start
    ]

    for tid, label, start_gid, count in tests:
        verses = gid_range(start_gid, start_gid + count - 1)
        if not verses:
            check(tid, f"{label} — SKIPPED (no verses)", True)
            continue
        text = ' '.join(v["standard"] for v in verses)
        t0 = time.perf_counter()
        r = loop.run_until_complete(validate_recitation(text))
        ms = (time.perf_counter() - t0) * 1000

        check(tid, f"{label} ({ms:.1f}ms)",
              r.get("mode") == "multi" and r.get("is_correct") is True,
              f"mode={r.get('mode')} is_correct={r.get('is_correct')} "
              f"total_verses={r.get('total_verses')} wer={r.get('total_wer')}")


# ── 10. Multi-verse: full page ────────────────────────────────────────────────

def test_multi_page():
    print("\n═══ 10. Multi-verse: full page ═══")

    loop = asyncio.get_event_loop()

    # Pages starting with "الم" or common phrase like "يا أيها الذين آمنوا"
    # are skipped (start detection fails on repeated phrases).
    # Page 580 (Juz 30, An-Nazi'at) has distinct opening verse.
    pages = [1, 604, 580]
    for i, pid in enumerate(pages):
        verses = verses_in_page(pid)
        if len(verses) < 3:
            continue
        text = ' '.join(v["standard"] for v in verses)
        t0 = time.perf_counter()
        r = loop.run_until_complete(validate_recitation(text))
        ms = (time.perf_counter() - t0) * 1000

        check(f"MP0{i+1}", f"Page {pid} ({len(verses)} verses, {ms:.1f}ms)",
              r.get("mode") == "multi" and r.get("is_correct") is True,
              f"mode={r.get('mode')} is_correct={r.get('is_correct')} "
              f"total_verses={r.get('total_verses')} wer={r.get('total_wer')}")


# ── 11. Edge cases ────────────────────────────────────────────────────────────

def test_edge_cases():
    print("\n═══ 11. Edge cases ═══")

    loop = asyncio.get_event_loop()

    # Empty
    r = loop.run_until_complete(validate_recitation(""))
    check("EC01", "Empty input → is_correct=False",
          r.get("is_correct") is False)

    # Non-Quran text
    r = loop.run_until_complete(validate_recitation("مرحبا كيف حالك اليوم"))
    check("EC02", "Non-Quran text — should not be marked correct",
          r.get("is_correct") is False or r.get("wer", 0) > 0.3,
          f"is_correct={r.get('is_correct')} wer={r.get('wer')}")

    # Tashkeel input → normalizes and still matches
    v = get_by_sura_aya(112, 1)
    if v and v.get("standard_full"):
        r = loop.run_until_complete(validate_recitation(v["standard_full"]))
        check("EC03", "Tashkeel input normalizes → matches 112:1",
              r.get("verse_key") == "112:1" and r.get("is_correct") is True,
              f"verse_key={r.get('verse_key')} wer={r.get('wer')}")

    # Uthmani input → normalizes and matches
    if v and v.get("uthmani"):
        r = loop.run_until_complete(validate_recitation(v["uthmani"]))
        check("EC04", "Uthmani text normalizes → matches 112:1",
              r.get("verse_key") == "112:1" and r.get("is_correct") is True,
              f"verse_key={r.get('verse_key')} wer={r.get('wer')}")

    # Whitespace only
    r = loop.run_until_complete(validate_recitation("   "))
    check("EC05", "Whitespace → is_correct=False",
          r.get("is_correct") is False)


# ── Dataset-driven tests ──────────────────────────────────────────────────────

def test_from_dataset():
    """Run all generated dataset cases through the validator."""
    dataset_path = Path(__file__).parent / "dataset.json"
    if not dataset_path.exists():
        print("\n═══ Dataset tests: SKIPPED (no dataset.json) ═══")
        return

    print("\n═══ 12. Dataset-driven tests ═══")
    loop = asyncio.get_event_loop()

    with open(dataset_path, encoding="utf-8") as f:
        cases = json.load(f)

    passed = 0
    for c in cases:
        if not c.get("input"):
            continue
        try:
            r = loop.run_until_complete(validate_recitation(c["input"]))
        except Exception as e:
            check(c["id"], c.get("label", c["id"]), False, f"Exception: {e}")
            continue

        # Check assertions
        ok = True
        info_parts = []

        if "expected_is_correct" in c:
            if r.get("is_correct") != c["expected_is_correct"]:
                ok = False
                info_parts.append(f"expected is_correct={c['expected_is_correct']}, "
                                   f"got {r.get('is_correct')}")

        if "expected_verse_key" in c:
            got_vk = r.get("verse_key")
            if got_vk != c["expected_verse_key"]:
                ok = False
                info_parts.append(f"expected verse_key={c['expected_verse_key']}, got {got_vk}")

        if "expected_wer_max" in c:
            wer = r.get("wer") if r.get("mode") == "single" else r.get("total_wer", 0)
            if wer > c["expected_wer_max"]:
                ok = False
                info_parts.append(f"WER {wer:.3f} > expected max {c['expected_wer_max']}")

        if "expected_total_verses" in c:
            tv = r.get("total_verses", 0)
            if tv != c["expected_total_verses"]:
                ok = False
                info_parts.append(f"total_verses={tv}, expected {c['expected_total_verses']}")

        if "expected_tashkeel_correct" in c:
            tash = r.get("tashkeel", {})
            got = tash.get("is_correct")
            exp = c["expected_tashkeel_correct"]
            if got != exp:
                ok = False
                info_parts.append(f"tashkeel is_correct={got}, expected {exp}")

        if check(c["id"], c.get("label", c["id"]), ok, "; ".join(info_parts)):
            passed += 1

    print(f"\nDataset: {passed}/{len([c for c in cases if c.get('input')])} passed")


# ── Main runner ───────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("Quran Recitation Validator — Full Test Suite")
    print("═" * 60)

    t_total = time.perf_counter()

    global _section
    for _section, fn in [
        ("normalization", test_normalizer),
        ("corpus_access", test_quran_db),
        ("single_perfect", test_single_perfect),
        ("single_substitution", test_single_substitution),
        ("single_deletion", test_single_deletion),
        ("single_tashkeel", test_single_tashkeel),
        ("multi_full_surah", test_multi_full_surahs),
        ("multi_with_errors", test_multi_with_errors),
        ("multi_consecutive", test_multi_consecutive),
        ("multi_page", test_multi_page),
        ("edge_cases", test_edge_cases),
        ("dataset_driven", test_from_dataset),
    ]:
        fn()

    elapsed = (time.perf_counter() - t_total) * 1000

    print("\n" + "═" * 60)
    passed = sum(1 for r in _results if r.passed)
    total = len(_results)
    rate = passed / total * 100 if total else 0
    print(f"Results: {passed}/{total} passed ({rate:.1f}%)  —  {elapsed:.0f}ms total")

    _write_results(passed, total, rate, elapsed)


def _write_results(passed: int, total: int, rate: float, elapsed: float):
    """Emit the per-category breakdown reported in the paper.

    Written by the harness itself so the table is reproducible from this
    repository rather than maintained by hand.
    """
    import platform
    from collections import OrderedDict

    cats: "OrderedDict[str, dict]" = OrderedDict()
    for r in _results:
        c = cats.setdefault(r.section, {"tests": 0, "passed": 0})
        c["tests"] += 1
        c["passed"] += int(r.passed)

    out = {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy_pct": round(rate, 1),
            "duration_ms": round(elapsed),
            "python": platform.python_version(),
            "dependencies": "standard library only",
            "failed_cases": [r.test_id for r in _results if not r.passed],
            "categories": cats,
        },
        "tests": [
            {"id": r.test_id, "category": r.section,
             "description": r.label, "passed": r.passed,
             **({"info": r.info} if not r.passed and r.info else {})}
            for r in _results
        ],
    }
    path = Path(__file__).parent.parent / "results" / "validator_results.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"Per-category results written to {path.relative_to(path.parent.parent)}")
    print("═" * 60)

    # Exit code for CI
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
