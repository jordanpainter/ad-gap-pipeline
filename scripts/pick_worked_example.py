"""
Shortlist candidate videos for the Chapter 4 worked-example figure and
Appendix E (cut detection), from the full-pool outputs. No models run.

A video qualifies if it is in the 730 hand-coded sample, is coded
deixis-present, and has a "featured" gap: speech on both sides, 2 to 8 s,
a non-empty Qwen subject, and an audio bucket other than speech (a speech
bucket means narration leaked into the classified window). --require-cut
also demands a hard cut in the featured gap, for Appendix E.

Ranked by: featured gap contains a cut, then number of audio buckets across
the video's gaps, then featured gap duration. Read-only: writes nothing.

    .venv/bin/python scripts/pick_worked_example.py [--top 30] [--require-cut]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "data" / "analysis"

MIN_FEATURED_GAP_S = 2.0
MAX_FEATURED_GAP_S = 8.0  # longer gaps swamp the timeline with one red block
MAX_VIDEO_S = 90.0
MAX_N_GAPS = 12  # keeps the timeline panel legible


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--require-cut", action="store_true")
    args = ap.parse_args()

    videos = pd.read_csv(ANALYSIS_DIR / "videos_summary.csv")
    gaps = pd.read_csv(ANALYSIS_DIR / "gaps_detail.csv")

    videos = videos[(videos.hand_coded == "Y") & (videos.video_deixis_present == "Y")
                    & (videos.video_n_gaps <= MAX_N_GAPS)
                    & (videos.video_duration <= MAX_VIDEO_S)]
    gaps = gaps.merge(videos[["video_id", "video_duration"]], on="video_id")

    # position is set by the midpoint, so "middle" alone doesn't guarantee speech on both sides
    featured = gaps[(gaps.gap_start > 0) & (gaps.gap_end < gaps.video_duration - 0.05)
                    & gaps.gap_duration.between(MIN_FEATURED_GAP_S, MAX_FEATURED_GAP_S, inclusive="right")
                    & gaps.subject.notna()
                    & (gaps.subject.str.lower() != "none")
                    & (gaps.audio_bucket != "speech")].copy()
    featured["cut"] = featured.has_cut == True  # noqa: E712 (column has NaNs)
    if args.require_cut:
        featured = featured[featured.cut]
    featured = featured.sort_values(["cut", "gap_duration"], ascending=False).drop_duplicates("video_id")

    buckets = gaps.groupby("video_id").audio_bucket.agg(lambda b: ",".join(sorted(set(b.dropna()))))
    out = (featured.drop(columns="video_duration").merge(videos, on="video_id")
           .assign(buckets=lambda d: d.video_id.map(buckets),
                   n_buckets=lambda d: d.buckets.str.count(",") + 1))
    print(f"{len(out)} qualifying videos; top {min(args.top, len(out))}:\n")
    out = out.sort_values(["cut", "n_buckets", "gap_duration"], ascending=False).head(args.top)

    cols = {"video_id": "video", "category": "category", "video_duration": "dur_s",
            "video_n_gaps": "gaps", "gap_index": "gap#", "gap_duration": "gap_s", "cut": "cut",
            "audio_bucket": "gap_audio", "audio_top_label": "top_label", "audio_top_prob": "p",
            "buckets": "video_buckets", "subject": "subject"}
    table = out[list(cols)].rename(columns=cols)
    with pd.option_context("display.width", 220, "display.max_colwidth", 26):
        print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
