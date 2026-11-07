"""
Chapter 4 worked-example figure: one video's speech-and-gap timeline, the
frames of one gap, and that gap's record from all four pipeline stages.
Values come from the full-pool outputs (data/analysis/gaps_detail.csv and
data/gaps/<video_id>.json); no models run. Candidates come from
scripts/pick_worked_example.py.

The report uses hAkkMMa0qig, gap index 0.
No title: the caption carries the claim.

    .venv/bin/python scripts/worked_example_figure.py OUT_DIR VIDEO_ID:GAP_INDEX [...]
"""

import json
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import gap_extract  # noqa: E402

OUT = Path(sys.argv[1])
PICKS = [a.split(":") for a in sys.argv[2:]]
G = pd.read_csv(ROOT / "data" / "analysis" / "gaps_detail.csv")

BLUE, ORANGE, INK, MUTED, RULE = "#2a78d6", "#eb6834", "#1a1a19", "#5c5c58", "#e3e3e0"
BAND_SPEECH, BAND_GAP = "#dde7f5", "#f7cdb8"  # calm fills for the timeline band
STAGE = {"1": ("speech", BLUE), "2": ("audio", "#1baf7a"), "3": ("visual", "#eda100"),
         "4": ("motion", "#e87ba4")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def wrap(label, value, width):
    lines = textwrap.wrap(value, width) or [""]
    return [(label, lines[0])] + [("", ln) for ln in lines[1:]]


for vid, gi in PICKS:
    gi = int(gi)
    rec = json.load(open(ROOT / "data" / "gaps" / f"{vid}.json"))
    video = Path(rec["video_path"])
    dur = rec["duration"]
    rows = G[G.video_id == vid].sort_values("gap_index")
    gap = rows[rows.gap_index == gi].iloc[0]

    audio = gap_extract.load_wav_16k(video).numpy()
    # envelope: peak amplitude per time bin, normalised to fill the band
    bins = 1400
    trim = len(audio) - len(audio) % bins
    env = np.abs(audio[:trim].reshape(bins, -1)).max(axis=1)
    env = env / env.max() * 0.92
    t = np.linspace(0, dur, bins)

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = GridSpec(3, 1, height_ratios=[0.78, 2.3, 1.5], hspace=0.30, figure=fig)

    # (a) timeline: waveform recedes behind speech/gap bands
    ax = fig.add_subplot(gs[0])
    ax.fill_between(t, -env, env, lw=0, color="#6f6f6a", zorder=2)
    for r in rec["speech_regions_vad"]:
        ax.axvspan(r["start"], r["end"], color=BAND_SPEECH, lw=0, zorder=0)
    for _, gg in rows.iterrows():
        hi = gg.gap_index == gi
        ax.axvspan(gg.gap_start, gg.gap_end, color=ORANGE if hi else BAND_GAP,
                   alpha=0.85 if hi else 1.0, lw=0, zorder=1)
        ax.text((gg.gap_start + gg.gap_end) / 2, 1.06, str(gg.gap_index + 1),
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                color=ORANGE if hi else MUTED, fontweight="bold" if hi else "normal")
    ax.set_xlim(0, dur)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("time (s)", labelpad=2, loc="right")
    ax.text(0.0, 1.30, f"{vid}   {dur:.1f} s   {len(rows)} gaps   gap ratio {rec['gap_to_runtime_ratio']:.0%}",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.text(dur * 0.01, -0.92, "speech", color=BLUE, fontsize=8, ha="left", fontweight="bold")

    # (b) frames of the featured gap
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tss = np.linspace(gap.gap_start, gap.gap_end - 1 / fps, 5)
    frames = []
    for ts in tss:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(ts * fps)))
        ok, fr = cap.read()
        fr = cv2.resize(fr, (int(fr.shape[1] * 480 / fr.shape[0]), 480))
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    sep = np.full((480, 14, 3), 255, np.uint8)
    strip = np.hstack([x for f in frames for x in (f, sep)][:-1])
    axf = fig.add_subplot(gs[1])
    axf.imshow(strip)
    axf.axis("off")
    fw = frames[0].shape[1] + 14
    for i, ts in enumerate(tss):
        axf.text(i * fw + frames[0].shape[1] / 2, 480 + 26, f"{ts:.2f} s", ha="center", va="top",
                 fontsize=8, color=MUTED)

    # zoom connectors from the featured gap down to the frame strip
    fig.canvas.draw()
    x0 = ax.transData.transform((gap.gap_start, 0))[0]
    x1 = ax.transData.transform((gap.gap_end, 0))[0]
    y_top = ax.get_window_extent().y0
    fb = axf.get_window_extent()
    inv = fig.transFigure.inverted()
    p = [inv.transform((x0, y_top)), inv.transform((x1, y_top)),
         inv.transform((fb.x1, fb.y1)), inv.transform((fb.x0, fb.y1))]
    for a, b in ((p[0], p[3]), (p[1], p[2])):
        fig.add_artist(matplotlib.lines.Line2D([a[0], b[0]], [a[1], b[1]], color=ORANGE, lw=0.8,
                                               alpha=0.55, ls=(0, (4, 3)), zorder=0))

    # (c) the gap record, as stage-tagged rows
    segs = rec["speech_segments"]
    before = [s for s in segs if s["end"] <= gap.gap_start + 0.01]
    after = [s for s in segs if s["end"] > gap.gap_end]
    q = lambda s, tail: ("“…" + s[-46:] + "”") if tail and len(s) > 46 else \
        ("“" + s[:46] + ("…”" if len(s) > 46 else "”"))
    top3 = ", ".join(x.rsplit(":", 1)[0] + " " + x.rsplit(":", 1)[1] for x in gap.audio_top5.split("; ")[:3])
    pad = f", padded {gap.audio_padded_seconds:.1f} s to 1 s" if gap.audio_padded_seconds > 0 else ""

    lines = []
    lines += [("1", f"start: {gap.gap_start:.1f} s   end: {gap.gap_end:.1f} s   duration: {gap.gap_duration:.1f} s   position: {gap.gap_position}")]
    lines += [("", f"speech before: {q(before[-1]['text'].strip(), True) if before else '(none)'}")]
    lines += [("", f"speech after: {q(after[0]['text'].strip(), False) if after else '(none)'}")]
    lines += [("2", f"audio bucket: {gap.audio_bucket}   ({top3}{pad})")]
    lines += [("3", f"subject: {gap.subject}   action: {gap.action}   scene: {gap.scene}   on-screen text: {gap.on_screen_text}")]
    lines += wrap("", "description: " + str(gap.description), 76)
    lines += [("4", f"motion: {gap.motion_magnitude:.3f}   hard cut: {'yes' if gap.has_cut else 'no'} "
                    f"(lowest frame correlation {gap.min_hist_correlation:.2f}, threshold 0.40)")]
    lines += [("", f"nearest verbal deixis event: {gap.nearest_verbal_deixis_s:.2f} s")]

    axr = fig.add_subplot(gs[2])
    axr.axis("off")
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    step = 1.0 / (len(lines) + 1)
    y = 1 - step * 0.6
    axr.text(0.0, 1.10, f"the record for gap {gi + 1}", fontsize=8.5, color=MUTED)
    for stage, txt in lines:
        if stage:
            name, col = STAGE[stage]
            axr.add_patch(FancyBboxPatch((0.0, y - step * 0.30), 0.108, step * 0.60,
                                         boxstyle="round,pad=0.002,rounding_size=0.012",
                                         facecolor=col, lw=0, transform=axr.transAxes))
            axr.text(0.054, y, f"{stage}  {name}", ha="center", va="center",
                     fontsize=7, color="white", fontweight="bold")
            axr.plot([0.0, 1.0], [y + step * 0.42, y + step * 0.42], color=RULE, lw=0.6, zorder=0)
        axr.text(0.125, y, txt, va="center", fontsize=8.5)
        y -= step

    fig.savefig(OUT / f"worked_example_{vid}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"worked_example_{vid}.png")
