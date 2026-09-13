#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the Uthmani -> Standard word map from the aligned Quran corpus.

Five passes, as described in the paper:

  1. Direct alignment      — verses whose Uthmani and Standard forms have the
                             same word count are aligned positionally.
  2. Ornamental marks      — U+06DE (rub el-hizb) has no Standard counterpart
                             and is stripped before alignment.
  3. Structural mismatch   — where word counts differ, difflib.SequenceMatcher
                             aligns the longest common subsequence of words;
                             equal-length runs are harvested positionally and
                             unequal runs are resolved by span alignment (4),
                             which recovers the one-to-many splits.
  3b. Span alignment       — inside an unequal run, each Uthmani word is aligned
                             to a span of one to three consecutive Standard
                             words by dynamic programming over character
                             similarity. This is what recovers the vocative
                             split (Uthmani writes "yā qawmi" as one word,
                             Standard as two), so those entries are derived
                             rather than hand-written.
  4. Majority vote         — an Uthmani key seen with several Standard forms
                             keeps the most frequent one.
  5. Ambiguity report      — keys whose majority is not unique are reported
                             rather than guessed.

Only words containing U+0670 (superscript alef) become keys: words without it
need no mapping and are handled by the character-level normalisation rules.

Usage:
    python3 tools/build_map.py            # write validator/data/uthmani_standard_map.json
    python3 tools/build_map.py --report   # print statistics only, write nothing
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "validator" / "data" / "quran.json"
OUT = ROOT / "validator" / "data" / "uthmani_standard_map.json"

SUPERSCRIPT_ALEF = "ٰ"
RUB_EL_HIZB = "۞"

# Marks removed from an Uthmani word before it is used as a key. U+0670 is
# deliberately NOT in this set: it is the character the map exists to resolve.
_MARKS = (set(range(0x064B, 0x0660)) | set(range(0x06D6, 0x06FD))) - {ord(SUPERSCRIPT_ALEF)}

# Letter variants that differ only by the font/encoding convention of the
# source text, not by orthography.
_LETTERS = {0x06CC: 0x064A, 0x06A9: 0x0643, 0x0671: 0x0627}


def key_form(word: str) -> str:
    """Uthmani word reduced to its map-key form: marks dropped, U+0670 kept."""
    return "".join(
        chr(_LETTERS.get(ord(c), ord(c))) for c in word if ord(c) not in _MARKS
    )


def _flat(word: str) -> str:
    """Key form with U+0670 realised as alef, for similarity scoring only."""
    return key_form(word).replace(SUPERSCRIPT_ALEF, "ا")


def span_align(u_words: list[str], s_words: list[str], max_span: int = 3):
    """
    Align each Uthmani word to a span of consecutive Standard words.

    Dynamic programming over Ratcliff/Obershelp similarity; deterministic and
    dependency-free. Returns [(uthmani_word, "standard words")] or [] when no
    alignment covers both sides.
    """
    n, m = len(u_words), len(s_words)
    if not n or not m or m < n:
        return []

    NEG = float("-inf")
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(1, n + 1):
        flat = _flat(u_words[i - 1])
        for j in range(1, m + 1):
            for k in range(1, min(max_span, j) + 1):
                if best[i - 1][j - k] == NEG:
                    continue
                joined = "".join(s_words[j - k:j])
                score = difflib.SequenceMatcher(None, flat, joined, autojunk=False).ratio()
                cand = best[i - 1][j - k] + score
                if cand > best[i][j]:
                    best[i][j] = cand
                    back[i][j] = k
    if best[n][m] == NEG:
        return []

    pairs, i, j = [], n, m
    while i > 0:
        k = back[i][j]
        if not k:
            return []
        pairs.append((u_words[i - 1], " ".join(s_words[j - k:j])))
        i, j = i - 1, j - k
    return list(reversed(pairs))


def build(verses: list[dict]) -> tuple[dict[str, str], dict]:
    pairs: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    stats = {
        "verses": len(verses),
        "ornamental_verses": 0,
        "equal_word_count": 0,
        "structural_mismatch": 0,
        "aligned_pairs": 0,
        "span_aligned_pairs": 0,
        "split_entries": 0,
    }

    for verse in verses:
        uthmani, standard = verse["uthmani"], verse["standard"]

        if RUB_EL_HIZB in uthmani:                                    # pass 2
            stats["ornamental_verses"] += 1
            uthmani = uthmani.replace(RUB_EL_HIZB, " ")

        u_words, s_words = uthmani.split(), standard.split()

        def harvest(us, ss):
            for u, s in zip(us, ss):
                k = key_form(u)
                if SUPERSCRIPT_ALEF in k:
                    pairs[k][s] += 1
                    stats["aligned_pairs"] += 1

        if len(u_words) == len(s_words):                              # pass 1
            stats["equal_word_count"] += 1
            harvest(u_words, s_words)
        else:                                                         # pass 3
            stats["structural_mismatch"] += 1
            matcher = difflib.SequenceMatcher(
                None, [key_form(w) for w in u_words], s_words, autojunk=False
            )
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if (i2 - i1) == (j2 - j1):
                    harvest(u_words[i1:i2], s_words[j1:j2])
                elif tag != "equal":
                    for u, s_span in span_align(u_words[i1:i2], s_words[j1:j2]):
                        k = key_form(u)
                        if SUPERSCRIPT_ALEF in k:
                            pairs[k][s_span] += 1
                            stats["span_aligned_pairs"] += 1

    mapping: dict[str, str] = {}
    conflicts, ambiguous = [], []
    for k, counter in pairs.items():                                  # passes 4-5
        ranked = counter.most_common()
        if len(ranked) > 1:
            conflicts.append((k, dict(counter)))
            if ranked[0][1] == ranked[1][1]:
                ambiguous.append((k, dict(counter)))
                continue
        mapping[k] = ranked[0][0]

    stats["split_entries"] = sum(1 for v in mapping.values() if " " in v)
    stats.update(
        distinct_keys=len(pairs),
        conflicted_keys=len(conflicts),
        ambiguous_keys=len(ambiguous),
        entries=len(mapping),
        conflicts=sorted(conflicts)[:20],
        ambiguous=sorted(ambiguous),
    )
    return dict(sorted(mapping.items())), stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="print statistics, write nothing")
    args = ap.parse_args()

    verses = json.loads(CORPUS.read_text(encoding="utf-8"))
    mapping, stats = build(verses)

    print(f"verses read                : {stats['verses']}")
    print(f"  ornamental (U+06DE)      : {stats['ornamental_verses']}")
    print(f"  equal word count         : {stats['equal_word_count']}")
    print(f"  structural mismatch      : {stats['structural_mismatch']}")
    print(f"aligned U+0670 occurrences : {stats['aligned_pairs']}")
    print(f"  recovered by span align  : {stats['span_aligned_pairs']}")
    print(f"distinct U+0670 keys       : {stats['distinct_keys']}")
    print(f"  with >1 Standard form    : {stats['conflicted_keys']}  (majority vote)")
    print(f"  unresolved ties          : {stats['ambiguous_keys']}  (omitted)")
    print(f"  one-to-many (split)      : {stats['split_entries']}")
    print(f"entries written            : {stats['entries']}")
    for k, forms in stats["conflicts"]:
        print(f"    conflict {k} -> {forms}")
    for k, forms in stats["ambiguous"]:
        print(f"    TIE      {k} -> {forms}")

    if not args.report:
        OUT.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(mapping)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
