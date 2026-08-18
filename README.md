# muslim-quran-validator

A corpus-aligned **Uthmani-to-Standard Quranic word mapping** (2,017 pairs) and a
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
    ├── uthmani_standard_map.json   The 2,017-pair mapping — the primary dataset
    ├── quran.json                   Reference Quran text (Uthmani + Standard)
    ├── morphology.json              Per-verse lemma/root annotation
    └── word-map.json                Per-token lemma/root lookup

tests/
├── test_all.py                124-case evaluation suite
├── dataset.json                63 of the 124 cases, machine-generated
└── dataset_gen.py              Generator for the synthetic portion of the suite

results/
└── validator_results.json      The real, dated 99.2% accuracy snapshot (124 cases)

example.py                      Minimal usage example
```

## Quick start

No external dependencies — pure Python standard library (`json`, `re`, `unicodedata`,
`difflib`, `pathlib`).

```bash
python3 example.py
python3 tests/test_all.py   # reproduces 123/124 (99.2%)
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

`validator/data/uthmani_standard_map.json` was constructed in five passes over the
complete 6,236-verse Quran corpus, aligning the Uthmani and Standard (Simple-Arabic)
encodings word-by-word (direct positional alignment where word counts match, sequence
alignment via `difflib.SequenceMatcher` where they don't, majority-vote conflict
resolution, and manual curation of a small number of structurally ambiguous cases). It
provides complete coverage: every Uthmani word containing U+0670 anywhere in the Quran
has an entry.

## The validator

Both the query and every reference verse pass through the same 7-step normalization
pipeline before comparison (Unicode NFC → Uthmani-to-Standard mapping → tashkeel removal
→ contextual U+0670 resolution → alef/hamza unification → word-initial rules → non-Arabic
removal), guaranteeing a phonetically correct recitation always produces a zero-distance
match regardless of its input orthography. Verse identification then runs a 4-layer
search (exact token match → morphological root/lemma expansion → relaxed 60% coverage →
fuzzy character-level match), and a Word Error Rate against the matched verse drives a
5-tier Arabic feedback system (perfect / minor errors / noticeable / significant /
not recognized).

On the 124-case evaluation suite in `tests/`, spanning twelve categories (exact match,
normalized match, Uthmani-script input, multi-verse spans, tashkeel variation, short/long
verses, hamza/alef variants, deliberate error detection, ASR-realistic noisy input, and
edge cases), the validator achieves **99.2% accuracy (123/124)**. The one failure (case
`SS03`, the isolated word *wahid*) is a documented, reproducible root-conflict in the
morphological annotation data, not a pipeline defect — see `tests/test_all.py` and
`results/validator_results.json` for the full breakdown.

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

See [CITATION.cff](CITATION.cff). A formal paper citation will be added here once the
accompanying resource paper is published.
