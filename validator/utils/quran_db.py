"""
Quran Database — in-memory indexed store for fast range queries.

Provides O(1) lookup by:
  - gid (global verse index 1–6236)
  - (sura_id, aya_id)
  - sura_id → list of verses
  - juz_id  → list of verses
  - page_id → list of verses

All data loaded once at import time from quran.json.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Internal store ──────────────────────────────────────────────────────────

_by_gid: Dict[int, Dict] = {}              # gid → verse
_by_sura_aya: Dict[Tuple[int, int], Dict] = {}  # (sura_id, aya_id) → verse
_by_sura: Dict[int, List[Dict]] = {}       # sura_id → [verses ordered by aya_id]
_by_juz: Dict[int, List[Dict]] = {}        # juz_id  → [verses ordered by gid]
_by_page: Dict[int, List[Dict]] = {}       # page_id → [verses ordered by gid]
_sorted_gids: List[int] = []               # all gids in order


def _load():
    global _sorted_gids
    if _by_gid:
        return

    with open(DATA_DIR / "quran.json", encoding="utf-8") as f:
        data = json.load(f)

    for v in data:
        gid = v["gid"]
        sura = v["sura_id"]
        aya = v["aya_id"]

        _by_gid[gid] = v
        _by_sura_aya[(sura, aya)] = v
        _by_sura.setdefault(sura, []).append(v)
        _by_juz.setdefault(v["juz_id"], []).append(v)
        _by_page.setdefault(v["page_id"], []).append(v)

    _sorted_gids = sorted(_by_gid.keys())
    logger.debug("QuranDB loaded: %d verses", len(_by_gid))


# Ensure load on import
_load()


# ── Public API ───────────────────────────────────────────────────────────────

def get(gid: int) -> Optional[Dict]:
    """Get a verse by global ID (1–6236). Returns None if not found."""
    return _by_gid.get(gid)


def get_by_sura_aya(sura_id: int, aya_id: int) -> Optional[Dict]:
    """Get a verse by (sura_id, aya_id)."""
    return _by_sura_aya.get((sura_id, aya_id))


def verses_in_sura(sura_id: int) -> List[Dict]:
    """All verses of a surah, in ayah order."""
    return _by_sura.get(sura_id, [])


def verses_in_juz(juz_id: int) -> List[Dict]:
    """All verses of a juz, in order."""
    return _by_juz.get(juz_id, [])


def verses_in_page(page_id: int) -> List[Dict]:
    """All verses of a Mushaf page, in order."""
    return _by_page.get(page_id, [])


def gid_range(start_gid: int, end_gid: int) -> List[Dict]:
    """Verses from start_gid to end_gid inclusive."""
    return [_by_gid[g] for g in range(start_gid, end_gid + 1) if g in _by_gid]


def next_gid(gid: int) -> Optional[int]:
    """Return the next GID after gid, or None if last verse."""
    idx = _sorted_gids.index(gid) if gid in _by_gid else -1
    if idx < 0 or idx + 1 >= len(_sorted_gids):
        return None
    return _sorted_gids[idx + 1]


def total_verses() -> int:
    return len(_by_gid)


def sura_count() -> int:
    return len(_by_sura)


def all_gids() -> List[int]:
    return list(_sorted_gids)


@lru_cache(maxsize=128)
def sura_info(sura_id: int) -> Optional[Dict]:
    """
    Return metadata for a surah: name, gid range, verse count.
    """
    verses = _by_sura.get(sura_id)
    if not verses:
        return None
    return {
        "sura_id": sura_id,
        "sura_name": verses[0]["sura_name"],
        "sura_name_en": verses[0]["sura_name_en"],
        "verse_count": len(verses),
        "start_gid": verses[0]["gid"],
        "end_gid": verses[-1]["gid"],
    }


@lru_cache(maxsize=64)
def juz_info(juz_id: int) -> Optional[Dict]:
    verses = _by_juz.get(juz_id)
    if not verses:
        return None
    return {
        "juz_id": juz_id,
        "verse_count": len(verses),
        "start_gid": verses[0]["gid"],
        "end_gid": verses[-1]["gid"],
        "start_verse": f"{verses[0]['sura_id']}:{verses[0]['aya_id']}",
        "end_verse": f"{verses[-1]['sura_id']}:{verses[-1]['aya_id']}",
    }
