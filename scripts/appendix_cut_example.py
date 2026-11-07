"""
Appendix E figure: how hard cuts are detected. Shows the consecutive frame
pair with the lowest histogram correlation in a gap (the cut) and a pair from
within one shot, with the brightness histograms each correlation is computed
from. Mirrors src/visual_signal.py exactly: consecutive decoded frames of the
extracted clip, 64-bin greyscale histograms, OpenCV HISTCMP_CORREL.

    .venv/bin/python scripts/appendix_cut_example.py OUT_DIR VIDEO_ID:GAP_INDEX
"""

import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import visual_signal  # noqa: E402

OUT = Path(sys.argv[1])
vid, gi = sys.argv[2].split(":")
gi = int(gi)
G = pd.read_csv(ROOT / "data" / "analysis" / "gaps_detail.csv")
gap = G[(G.video_id == vid) & (G.gap_index == gi)].iloc[0]

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#5c5c58", "#e3e3e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

tmp = Path(tempfile.mkdtemp(prefix="appendix_cut_"))
clip = visual_signal.extract_clip(ROOT / "data" / "raw_videos" / f"{vid}.mp4",
                                  gap.gap_start, gap.gap_end, tmp / "clip.mp4")
cap = cv2.VideoCapture(str(clip))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
colour, grey = [], []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    colour.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    grey.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
cap.release()

hists = [cv2.calcHist([f], [0], None, [64], [0, 256]) for f in grey]
corrs = [float(cv2.compareHist(hists[i], hists[i + 1], cv2.HISTCMP_CORREL)) for i in range(len(grey) - 1)]
cut_i = int(np.argmin(corrs))
within_i = int(np.argsort(corrs)[len(corrs) // 2])  # a median-correlation pair, inside one shot

fig = plt.figure(figsize=(7.2, 4.0))
gs = GridSpec(2, 3, width_ratios=[1, 1, 2.4], hspace=0.28, wspace=0.12, figure=fig)
for row, (i, label) in enumerate(((cut_i, "across a cut"), (within_i, "within one shot"))):
    for col in (0, 1):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(colour[i + col])
        ax.axis("off")
        ax.text(0.5, -0.04, f"{(i + col) / fps:.2f} s", transform=ax.transAxes, ha="center",
                va="top", fontsize=7.5, color=MUTED)
    axh = fig.add_subplot(gs[row, 2])
    for col, c in ((0, BLUE), (1, ORANGE)):
        axh.plot(hists[i + col].ravel(), color=c, lw=1.3)
    axh.set_yticks([])
    axh.set_xlim(0, 63)
    axh.set_xticks([0, 16, 32, 48, 63])
    axh.set_xticklabels(["0", "64", "128", "192", "255"], fontsize=7.5)
    axh.tick_params(length=2, pad=2)
    axh.grid(axis="x", color=GRID, lw=0.5)
    verdict = "cut" if corrs[i] < visual_signal.CUT_CORRELATION_THRESHOLD else "no cut"
    axh.set_title(f"{label}: correlation {corrs[i]:.2f}, {verdict}", fontsize=8.5, loc="left", pad=4)
    if row == 1:
        axh.set_xlabel("pixel brightness", fontsize=7.5, labelpad=1)

fig.savefig(OUT / f"appendix_cut_{vid}.png", dpi=300, bbox_inches="tight")
print(f"saved {OUT / f'appendix_cut_{vid}.png'}  cut pair {cut_i} corr {corrs[cut_i]:.3f}, "
      f"within-shot pair {within_i} corr {corrs[within_i]:.3f}, frames {len(grey)}")
