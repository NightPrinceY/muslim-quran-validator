"""
Quran search engine — Python port of quran-search-engine (TypeScript).
4-layer pipeline: exact word match → lemma/root → relaxed exact → fuzzy.
All data loaded once at startup and held in memory (~8.5 MB).
"""
import re
import json
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# ─── Arabic Normalization ──────────────────────────────────────────────────────

_RE_TASHKEEL = re.compile(
    r'[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06FC]'
)
_RE_ALEF_VARIANTS = re.compile(r'[إأآٱ\u0671]')
_RE_HAMZA_VARIANTS = re.compile(r'[ؤئ]')
_RE_NON_ARABIC = re.compile(r'[^\u0621-\u064A\s]')
_RE_MULTI_SPACE = re.compile(r'\s+')


def remove_tashkeel(text: str) -> str:
    text = text.replace('\u0671', '\u0627')  # alef wasla → alef
    return _RE_TASHKEEL.sub('', text)


def normalize_arabic(text: str) -> str:
    """Canonical normalization: strip diacritics, unify alef/hamza/alif-maqsura."""
    text = remove_tashkeel(text)
    text = _RE_ALEF_VARIANTS.sub('\u0627', text)   # إأآٱ → ا
    text = _RE_HAMZA_VARIANTS.sub('\u0621', text)  # ؤئ → ء
    text = text.replace('\u0649', '\u064A')         # ى → ي
    text = _RE_NON_ARABIC.sub('', text)
    text = _RE_MULTI_SPACE.sub(' ', text)
    return text.strip()


# ─── Data (loaded once, held in memory) ───────────────────────────────────────

_quran_data: Optional[List[Dict]] = None
_morphology_map: Optional[Dict[int, Dict]] = None
_word_map: Optional[Dict[str, Dict]] = None
_verse_normalized: Optional[Dict[int, str]] = None   # gid → normalized text
_verse_word_sets: Optional[Dict[int, set]] = None    # gid → set of words


def _load():
    global _quran_data, _morphology_map, _word_map, _verse_normalized, _verse_word_sets
    if _quran_data is not None:
        return

    with open(DATA_DIR / "quran.json", encoding="utf-8") as f:
        _quran_data = json.load(f)

    with open(DATA_DIR / "morphology.json", encoding="utf-8") as f:
        _morphology_map = {item["gid"]: item for item in json.load(f)}

    with open(DATA_DIR / "word-map.json", encoding="utf-8") as f:
        _word_map = json.load(f)

    _verse_normalized = {}
    _verse_word_sets = {}
    for v in _quran_data:
        norm = normalize_arabic(v["standard"])
        _verse_normalized[v["gid"]] = norm
        _verse_word_sets[v["gid"]] = set(norm.split())

    logger.info(
        "Quran data loaded: %d verses, %d morphology, %d word-map entries",
        len(_quran_data), len(_morphology_map), len(_word_map),
    )


def _data():
    _load()
    return _quran_data, _morphology_map, _word_map, _verse_normalized, _verse_word_sets


# ─── Search Layers ─────────────────────────────────────────────────────────────

def _exact_search(tokens: List[str], verse_word_sets: Dict[int, set],
                  quran_data: List[Dict]) -> List[Dict]:
    """Strict AND — every query token must appear as a whole word in the verse."""
    return [v for v in quran_data
            if all(t in verse_word_sets[v["gid"]] for t in tokens)]


def _linguistic_search(tokens: List[str], quran_data: List[Dict],
                        morphology_map: Dict[int, Dict],
                        word_map: Dict[str, Dict]) -> List[Dict]:
    """
    Lemma + root search with AND logic.
    Each token is looked up in word_map → lemma/root, then matched against
    morphology_map. Only tokens with known linguistic entries participate.
    """
    token_gid_sets: List[set] = []

    for token in tokens:
        entry = word_map.get(token, {})
        lemma = entry.get("lemma")
        root = entry.get("root")
        if not lemma and not root:
            continue  # skip tokens with no linguistic info

        gids: set = set()
        for verse in quran_data:
            m = morphology_map.get(verse["gid"], {})
            if lemma and lemma in m.get("lemmas", []):
                gids.add(verse["gid"])
            elif root and root in m.get("roots", []):
                gids.add(verse["gid"])
        if gids:
            token_gid_sets.append(gids)

    if not token_gid_sets:
        return []

    result_gids = token_gid_sets[0]
    for s in token_gid_sets[1:]:
        result_gids &= s
        if not result_gids:
            return []

    gid_to_verse = {v["gid"]: v for v in quran_data}
    return [gid_to_verse[gid] for gid in result_gids if gid in gid_to_verse]


def _relaxed_search(tokens: List[str], verse_word_sets: Dict[int, set],
                    quran_data: List[Dict], min_ratio: float = 0.60) -> List[Dict]:
    """
    Relaxed exact match — at least ⌈min_ratio × N⌉ tokens must match.
    Fallback for when the user said one wrong word (breaking strict AND)
    but the other words correctly identify the verse.
    Only used when strict AND + linguistic both return nothing.
    """
    n = len(tokens)
    min_matches = max(1, -(-int(n * min_ratio) // 1))  # ceiling
    return [v for v in quran_data
            if sum(1 for t in tokens if t in verse_word_sets[v["gid"]]) >= min_matches]


def _fuzzy_search(query: str, quran_data: List[Dict],
                  verse_normalized: Dict[int, str],
                  threshold: float = 0.45) -> List[Tuple[Dict, float]]:
    """
    Last resort when exact, linguistic and relaxed search all fail.

    Ratcliff/Obershelp similarity via the standard library's
    difflib.SequenceMatcher, scored as the best similarity between the query
    and any same-length window of the verse, so that a short degraded query
    is not penalised for the length of the verse containing it.

    Deliberately stdlib-only: an optional third-party matcher would make the
    validator's output depend on which version of that library happened to be
    installed, which defeats the point of a deterministic validator.
    """
    normalized_query = normalize_arabic(query)
    if not normalized_query:
        return []

    n = len(normalized_query)
    sm = SequenceMatcher(None, autojunk=False)
    sm.set_seq2(normalized_query)

    results: List[Tuple[Dict, float]] = []
    for v in quran_data:
        text = verse_normalized[v["gid"]]
        if not text:
            continue
        if len(text) <= n:
            sm.set_seq1(text)
            if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
                continue
            score = sm.ratio()
        else:
            # Best-matching window of the verse, stepped to keep this linear
            # enough over 6,236 verses while remaining fully deterministic.
            best = 0.0
            step = max(1, n // 4)
            for i in range(0, len(text) - n + 1, step):
                sm.set_seq1(text[i:i + n])
                if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                    continue
                r = sm.ratio()
                if r > best:
                    best = r
                    if best == 1.0:
                        break
            score = best
        if score >= threshold:
            results.append((v, score))

    results.sort(key=lambda x: (-x[1], x[0]["gid"]))
    return results[:30]


# ─── Scoring ───────────────────────────────────────────────────────────────────

_MATCH_PRIORITY = {"exact": 4, "lemma": 3, "root": 2, "fuzzy": 1, "none": 0}


def _score_verse(verse: Dict, tokens: List[str],
                 morphology_map: Dict[int, Dict],
                 word_map: Dict[str, Dict],
                 verse_word_sets: Dict[int, set],
                 fuzzy_score: float = 0.0) -> Tuple[float, str, List[str]]:
    """
    Compute weighted relevance score for a candidate verse.

      Base score per token:  exact=3  lemma=2  root=1
      Coverage bonus:        × (1 + matched_tokens / verse_word_count)
        → short verses where the query covers most words rank above long verses
          where the same words are scattered (e.g. "الرحمن الرحيم" → 1:3 not 1:1)
    """
    total = 0.0
    best_type = "none"
    matched: List[str] = []

    words = verse_word_sets.get(verse["gid"], set())
    morph = morphology_map.get(verse["gid"], {})

    for token in tokens:
        t_score = 0.0
        t_type = "none"

        if token in words:
            t_score, t_type = 3.0, "exact"
        else:
            entry = word_map.get(token, {})
            t_lemma, t_root = entry.get("lemma"), entry.get("root")
            if t_lemma and t_lemma in morph.get("lemmas", []):
                t_score, t_type = 2.0, "lemma"
            elif t_root and t_root in morph.get("roots", []):
                t_score, t_type = 1.0, "root"

        if t_type != "none":
            total += t_score
            matched.append(token)
            if _MATCH_PRIORITY[t_type] > _MATCH_PRIORITY[best_type]:
                best_type = t_type

    if best_type == "none" and fuzzy_score > 0:
        total = fuzzy_score * 0.5
        best_type = "fuzzy"

    if matched:
        # Coverage bonus: prefer verses where the query fills a larger fraction
        coverage = len(matched) / max(len(words), 1)
        total *= (1.0 + coverage)

    return total, best_type, matched


# ─── Public API ────────────────────────────────────────────────────────────────

def find_best_verse(query: str, min_score: float = 0.5) -> Optional[Dict]:
    """
    Find the best-matching Quran verse for a recited query string.

    Pipeline (each layer only runs if the previous layers found nothing):
      1. Exact AND     — all tokens must appear as whole words
      2. Linguistic    — lemma + root AND across tokens
      3. Relaxed exact — ≥60% of tokens match (handles 1 wrong/extra word)
      4. Fuzzy         — difflib.SequenceMatcher windowed fallback

    Returns the highest-scored verse dict augmented with:
      _score, _match_type, _matched_tokens
    or None if nothing scores above min_score.
    """
    quran_data, morphology_map, word_map, verse_normalized, verse_word_sets = _data()

    normalized = normalize_arabic(query)
    if not normalized:
        return None

    tokens = [t for t in normalized.split() if t]
    if not tokens:
        return None

    # ── Candidate collection ──────────────────────────────────────────────
    # Always run BOTH exact AND linguistic together so that morphological
    # variants (e.g. "واحد" vs "احد" sharing root وحد) are candidates
    # alongside exact matches — scoring then picks the best.
    # Relaxed + fuzzy are last-resort fallbacks only.
    seen: set = set()
    candidates: List[Dict] = []

    def _add(verses: List[Dict]):
        for v in verses:
            if v["gid"] not in seen:
                seen.add(v["gid"])
                candidates.append(v)

    _add(_exact_search(tokens, verse_word_sets, quran_data))
    _add(_linguistic_search(tokens, quran_data, morphology_map, word_map))

    if not candidates and len(tokens) >= 3:
        _add(_relaxed_search(tokens, verse_word_sets, quran_data))

    fuzzy_scores: Dict[int, float] = {}
    if not candidates:
        for verse, fscore in _fuzzy_search(query, quran_data, verse_normalized):
            if verse["gid"] not in seen:
                seen.add(verse["gid"])
                candidates.append(verse)
                fuzzy_scores[verse["gid"]] = fscore

    if not candidates:
        return None

    # Score all candidates and return the best
    scored: List[Dict] = []
    for verse in candidates:
        score, match_type, matched_tokens = _score_verse(
            verse, tokens, morphology_map, word_map, verse_word_sets,
            fuzzy_scores.get(verse["gid"], 0.0),
        )
        if score >= min_score:
            scored.append({
                **verse,
                "_score": score,
                "_match_type": match_type,
                "_matched_tokens": matched_tokens,
            })

    if not scored:
        return None

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[0]
