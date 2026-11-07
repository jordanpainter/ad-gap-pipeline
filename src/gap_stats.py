"""
Descriptive statistics over the extracted gaps: duration percentiles, and
how many gaps are left at a few candidate minimum lengths.

AD guidelines don't give a minimum pause length, so this reports the
distribution rather than picking a cutoff.

Usage:
    python src/gap_stats.py
    python src/gap_stats.py --candidate-thresholds 0.5 1 1.5 2 3
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAPS_DIR = ROOT / "data" / "gaps"


def load_records(gaps_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(gaps_dir.glob("*.json"))]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize(records: list[dict], candidate_thresholds: list[float]) -> None:
    all_gaps = [g for r in records for g in r["gaps"]]
    durations = [g["duration"] for g in all_gaps]
    ratios = [r["gap_to_runtime_ratio"] for r in records if r["gap_to_runtime_ratio"] is not None]

    print(f"videos: {len(records)}   total gaps: {len(all_gaps)}")
    if not durations:
        print("no gaps found")
        return

    print("\n-- gap duration (s) --")
    print(f"  min={min(durations):.2f}  p10={percentile(durations, .10):.2f}  "
          f"p25={percentile(durations, .25):.2f}  median={stats.median(durations):.2f}  "
          f"p75={percentile(durations, .75):.2f}  p90={percentile(durations, .90):.2f}  "
          f"p95={percentile(durations, .95):.2f}  max={max(durations):.2f}")
    print(f"  mean={stats.mean(durations):.2f}  stdev={stats.pstdev(durations):.2f}")

    print("\n-- gaps by position --")
    for pos in ("start", "middle", "end"):
        pos_durs = [g["duration"] for g in all_gaps if g["position"] == pos]
        if pos_durs:
            print(f"  {pos:6s} n={len(pos_durs):3d}  median={stats.median(pos_durs):.2f}s  "
                  f"mean={stats.mean(pos_durs):.2f}s")

    print("\n-- gap-to-runtime ratio per video --")
    print(f"  min={min(ratios):.1%}  median={stats.median(ratios):.1%}  max={max(ratios):.1%}  "
          f"mean={stats.mean(ratios):.1%}")

    print("\n-- gaps retained at candidate minimum-duration thresholds --")
    print("  (this is what a WCAG-1.2.7-style minimum-pause rule would keep/discard)")
    for t in candidate_thresholds:
        kept = [d for d in durations if d >= t]
        print(f"  >= {t:>4.1f}s: {len(kept):4d} / {len(durations)} gaps kept "
              f"({len(kept) / len(durations):.1%}), "
              f"{sum(kept):.1f}s / {sum(durations):.1f}s total gap time retained "
              f"({sum(kept) / sum(durations):.1%})" if durations else "")


def plot_histogram(records: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    durations = [g["duration"] for r in records for g in r["gaps"]]
    if not durations:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(durations, bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("gap duration (s)")
    ax.set_ylabel("count")
    ax.set_title(f"Gap duration distribution (n={len(durations)} gaps, {len(records)} videos)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nhistogram saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps-dir", type=Path, default=GAPS_DIR)
    ap.add_argument("--candidate-thresholds", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    records = load_records(args.gaps_dir)
    summarize(records, args.candidate_thresholds)

    if not args.no_plot:
        plot_histogram(records, args.gaps_dir / "gap_duration_histogram.png")


if __name__ == "__main__":
    main()
