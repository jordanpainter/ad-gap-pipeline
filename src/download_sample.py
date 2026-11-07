"""
Download a small random sample of Shorts with yt-dlp, one per channel, for
trying the pipeline out.

Reads video metadata from the CSV given with --csv (not in this repo).

Usage:
    python src/download_sample.py --n 8
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from pathlib import Path

DEFAULT_CSV = Path.home() / "Desktop" / "all_long_shorts.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_videos"


def sample_rows(csv_path: Path, n: int, seed: int) -> list[dict]:
    rows = list(csv.DictReader(open(csv_path)))
    by_channel: dict[str, list[dict]] = {}
    for r in rows:
        by_channel.setdefault(r["channel_name"], []).append(r)

    rng = random.Random(seed)
    channels = list(by_channel.keys())
    rng.shuffle(channels)

    picked: list[dict] = []
    ci = 0
    while len(picked) < n and ci < len(channels):
        chan_rows = by_channel[channels[ci]]
        picked.append(rng.choice(chan_rows))
        ci += 1
    return picked


def download(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        video_id = r["video_id"]
        url = r["shorts_url"]
        out_tmpl = str(out_dir / f"{video_id}.%(ext)s")
        existing = list(out_dir.glob(f"{video_id}.*"))
        if existing:
            print(f"skip (already have) {video_id}")
            continue
        print(f"downloading {video_id}  [{r['channel_name'].strip()}]  {r['duration_seconds']}s")
        subprocess.run(
            [
                "yt-dlp",
                "-f", "mp4/best",
                "-o", out_tmpl,
                url,
            ],
            check=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of videos to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    rows = sample_rows(args.csv, args.n, args.seed)
    download(rows, args.out)

    print(f"\n{len(rows)} videos requested -> {args.out}")


if __name__ == "__main__":
    main()
