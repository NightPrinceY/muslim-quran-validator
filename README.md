# muslim-quran-validator

A corpus-aligned **Uthmani-to-Standard Quranic word mapping** (2,290 pairs) and a
**deterministic, LLM-free Quranic recitation validator** built on it.

Quranic text is distributed in two orthographic forms that are byte-level distinct even
when phonetically identical: the **Uthmani** script used in every printed mushaf, and the
**Standard** (*Imla'i*) Arabic form that mainstream Arabic NLP tooling, ASR models, and
text corpora are built for. The gap is concentrated in one Unicode character, **U+0670**
(Arabic Superscript Alef), which marks a long *a* vowel in words like *al-rahman* and
*al-rahim* — among the most frequently recited words in the Quran — and is silently
mishandled (stripped or left untreated) by general-purpose Arabic normalizers. This
repository releases a word-level mapping that closes that gap, and a recitation validator
that uses it.

This is the released research artifact accompanying a resource paper (citation to be
added on publication — see [CITATION.cff](CITATION.cff)).

## What's here

```
validator/
├── validator_mcp.py          Main entry point: validate_recitation(text)
├── utils/
│   ├── normalizer.py         7-step Arabic text normalization pipeline
│   ├── quran_search.py       4-layer verse search (exact/lemma/root/relaxed/fuzzy)
│   ├── multi_verse.py        Multi-verse (surah/juz/page/range) alignment
│   ├── quran_db.py           Quran text index (6,236 verses)
│   └── tashkeel.py           Optional diacritic-level validation
└── data/
    ├── uthmani_standard_map.json   The 2,290-pair mapping — the primary dataset
    ├── quran.json                   Reference Quran text (Uthmani + Standard)
    ├── morphology.json              Per-verse lemma/root annotation
    └── word-map.json                Per-token lemma/root lookup

tools/
└── build_map.py                Rebuilds the mapping from quran.json (4 passes)

tests/
├── test_all.py                124-case evaluation suite
├── dataset.json                62 of the 124 cases, machine-generated
└── dataset_gen.py              Generator for the synthetic portion (seeded)

results/
└── validator_results.json      Written by test_all.py on every run

example.py                      Minimal usage example
```

## Quick start

No external dependencies — pure Python standard library (`json`, `re`, `unicodedata`,
`difflib`, `pathlib`).

```bash
python3 example.py
python3 tests/test_all.py       # 122/124 (98.4%), writes results/validator_results.json
python3 tools/build_map.py --report   # rebuild statistics for the mapping
```

```python
import asyncio, sys
sys.path.insert(0, "validator")
from validator_mcp import validate_recitation

result = asyncio.run(validate_recitation("قل هو الله احد"))
print(result["feedback"])       # Arabic feedback
print(result["verse_key"])      # "112:1"
print(result["wer"])            # 0.0
```

## The mapping

`validator/data/uthmani_standard_map.json` is built by `tools/build_map.py` from
`validator/data/quran.json`, so it can be regenerated and checked rather than taken on
trust. Four passes over the 6,236-verse corpus: positional alignment for the 5,813 verses
whose two forms have equal word counts; the ornamental mark U+06DE (199 verses) stripped
first; `difflib.SequenceMatcher` for the 423 verses where the counts differ, with
unequal runs resolved by a dynamic program that aligns each Uthmani word to a span of one
to three Standard words; and majority vote for the two keys that occur with more than one
Standard form. Nothing is hand-curated.

That span alignment is what recovers the **one-to-many** entries (58 of them), where
Uthmani joins the vocative particle to its noun and Standard separates it — `يٰقوم` →
`يا قوم`. Those entries matter: without them, Uthmani-script input silently *loses* the
`يا`, including in `يا أيها الذين آمنوا`.

The corpus holds 2,291 distinct Uthmani word forms carrying U+0670 and the map covers
2,290 of them. The one omission is a form whose verse admits no consistent alignment; it
is left out rather than guessed.

## The validator

Both the query and every reference verse pass through the same 7-step normalization
pipeline before comparison (Unicode NFC → Uthmani-to-Standard mapping → tashkeel removal
→ contextual U+0670 resolution → alef/hamza unification → word-initial rules → non-Arabic
removal), so that a phonetically correct recitation matches regardless of the orthography
it arrived in. How far that actually holds is measurable: normalizing both forms of all
6,236 verses, **5,669 (90.9%)** reduce to identical strings. The remaining 567 are not
U+0670 failures but other Uthmani-Standard divergences (the assimilated lam of `الليل`,
`داوود` spelled with one waw, medial hamza in `يسألونك`) that the character rules do not
yet cover. Verse identification then runs a 4-layer
search (exact token match → morphological root/lemma expansion → relaxed 60% coverage →
fuzzy character-level match), and a Word Error Rate against the matched verse drives a
5-tier Arabic feedback system (perfect / minor errors / noticeable / significant /
not recognized).

On the 124-case suite in `tests/`, the validator scores **98.4% (122/124)**. The
per-category breakdown is written by the harness itself into
`results/validator_results.json` on every run, so it cannot drift from the code.

Both failures are the same failure. `SS03` recites the isolated word *wahid* where the
verse has *ahad*; the validator returns 6:19, which **literally contains** *wahid* and is
therefore an exact match at layer 1, while the intended 112:1 is not. `ME03` shows this
is not limited to one-word inputs: the full four-verse Surah Al-Ikhlas with the same
substitution anchors the multi-verse aligner on 6:19 and reports the whole recitation as
verse 112:4 with ten insertions. Layer 1 has no notion of how *surprising* a match is —
fixing it means scoring candidates against their alternatives instead of accepting the
first exact hit, which is left as future work rather than patched against these two cases.

Raising the fuzzy threshold from 0.45 to 0.65 scores 123/124. It has not been adopted:
the only evidence for it is the suite it would then be scored on.

## What's not here

This repository is the research artifact — the normalization/matching algorithm and the
dataset. It does not include the MCP server wrapper, caching/rate-limiting middleware, or
any of the production voice-platform integration those are part of in the deployed system
this work originated from; those are specific to that system's infrastructure and not
needed to use or evaluate this resource.

## License

- **Code** (`validator/*.py`, `tests/*.py`, `example.py`): [MIT](LICENSE).
- **Data** (`validator/data/*.json`, `tests/dataset.json`, `results/*.json`):
  [CC-BY 4.0](DATA_LICENSE).
- `validator/data/quran.json` and `validator/data/morphology.json` derive from the
  [Tanzil](https://tanzil.net) Quran text project and the
  [Quranic Arabic Corpus](https://corpus.quran.com), both openly licensed for research
  use; see those projects for their own terms on the underlying text.

## Citing this work

The accompanying resource paper is on arXiv:

> Yahya Mohamed Elnawasany. *A Corpus-Aligned Uthmani-to-Standard Quranic Word Mapping
> and a Deterministic Recitation Validator.* arXiv:2609.14967 [cs.CL], 2026.
> <https://arxiv.org/abs/2609.14967>

```bibtex
@misc{elnawasany2026uthmani,
  title         = {A Corpus-Aligned Uthmani-to-Standard Quranic Word Mapping
                   and a Deterministic Recitation Validator},
  author        = {Elnawasany, Yahya Mohamed},
  year          = {2026},
  eprint        = {2609.14967},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2609.14967},
  url           = {https://arxiv.org/abs/2609.14967}
}
```

Every number in that paper is reproduced by this repository — see
[Quick start](#quick-start) — except the toolkit comparison, which also needs PyArabic and
CAMeL Tools, and the deployment measurement, whose transcripts are not public.
See also [CITATION.cff](CITATION.cff).
