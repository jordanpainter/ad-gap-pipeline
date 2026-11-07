"""
Look up a video's corpus metadata by video_id: category, subgenre, and the
hand-coded deixis fields if it was coded.

Reads all_videos_master.csv from the corpus directory, which isn't part of
this repo. Every row has a category and subgenre; only the hand-coded rows
have deixis_present, modality, levinson_type and referent_note filled in.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

# Same default and override as batch_analyze.py. With no manifest, lookup()
# returns blanks.
CORPUS_DIR = Path(os.environ.get("AD_GAP_CORPUS_DIR", Path.home() / "data" / "ad_gap_analysis"))
ALL_VIDEOS_CSV = CORPUS_DIR / "all_videos_master.csv"

_by_id: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _by_id
    if _by_id is None:
        if not ALL_VIDEOS_CSV.exists():
            _by_id = {}
        else:
            _by_id = {r["video_id"]: r for r in csv.DictReader(open(ALL_VIDEOS_CSV))}
    return _by_id


def lookup(video_id: str) -> dict:
    """
    Return category, subgenre, seed_channel and the hand-coded fields.

    Fields are blank if the video isn't in the manifest or wasn't hand-coded.
    """
    row = _load().get(video_id, {})
    return {
        "category": row.get("category", ""),
        "subgenre": row.get("subgenre", ""),
        "seed_channel": row.get("seed_channel", ""),
        "hand_coded": row.get("hand_coded", ""),
        "deixis_present": row.get("deixis_present_Y_N_unclear", ""),
        "modality": row.get("modality_verbal_gestural_both", ""),
        "levinson_type": row.get("levinson_type", ""),
        "referent_note": row.get("referent_note", ""),
        "in_corpus": bool(row),
    }
