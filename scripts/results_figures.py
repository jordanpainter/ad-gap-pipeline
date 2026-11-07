"""
Chapter 4 Results figures, drawn from the full-pool outputs
(data/analysis/{videos_summary,gaps_detail}.csv). No models run, nothing is
recomputed that scripts/results_summary.py does not also report.

Figures have no titles: the caption carries the claim in the report.

    .venv/bin/python scripts/results_figures.py [OUT_DIR]

Writes results_corpus_raster.png, results_gapratio_ridgeline.png,
results_threshold_sweep.png, results_capacity_k.png, results_audio_composition.png,
results_deixis_raincloud.png.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "data" / "analysis"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Categorical slots 1-3 of the palette, colourblind-safe (deutan dE 9.2).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
# The raster fills the whole page with two categories, so it uses one hue against a
# near-neutral instead of the categorical pair, so the gaps stand out.
RASTER_SPEECH, RASTER_GAP = "#e6e8ea", "#1c5cab"
# Sequential blue ramp (light to dark) for magnitude: the ridgeline shades each
# category by its mean gap ratio, so the ordering is visible as tone.
BLUE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#184f95"]
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#e3e3e0"

v = pd.read_csv(ANALYSIS_DIR / "videos_summary.csv")
g = pd.read_csv(ANALYSIS_DIR / "gaps_detail.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
CATS = v.groupby("category").video_gap_ratio.mean().sort_values().index.tolist()


def save(fig, name, dpi=300):
    fig.savefig(OUT / f"results_{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"results_{name}.png")


# 1. corpus raster: every analysed video, one row, sorted by gap ratio
vr = v.sort_values("video_gap_ratio", ascending=False).reset_index(drop=True)
COLS = 900
M = np.zeros((len(vr), COLS), dtype=np.uint8)  # 0 = speech, 1 = gap
row_of = {vid: i for i, vid in enumerate(vr.video_id)}
dur = dict(zip(vr.video_id, vr.video_duration))
for r in g.itertuples():
    i = row_of[r.video_id]; d = dur[r.video_id]
    a = int(np.floor(r.gap_start / d * COLS)); b = int(np.ceil(r.gap_end / d * COLS))
    M[i, a:max(b, a + 1)] = 1

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.imshow(M, aspect="auto", interpolation="nearest", cmap=ListedColormap([RASTER_SPEECH, RASTER_GAP]))
n = len(vr)
n_zs = int((vr.video_gap_ratio == 1.0).sum()); n_zp = int((vr.video_n_gaps == 0).sum())
ax.set_xticks(np.linspace(0, COLS, 6)); ax.set_xticklabels([f"{x:.0%}" for x in np.linspace(0, 1, 6)])
ax.set_yticks([]); ax.spines[:].set_visible(False)
ax.set_xlabel("position in video (proportion of runtime)")
ax.set_ylabel(f"all {n:,} analysed videos, sorted by gap ratio")
def bracket(y0, y1, txt, col):
    x = COLS * 1.015; w = COLS * 0.012
    ax.plot([x, x], [y0, y1], color=col, lw=1.4, clip_on=False)
    for y in (y0, y1):
        ax.plot([x, x + w], [y, y], color=col, lw=1.4, clip_on=False)
    ax.annotate(txt, xy=(x + w * 2.2, (y0 + y1) / 2), va="center", fontsize=8.5,
                fontweight="bold", color=col, annotation_clip=False)
bracket(0, n_zs, f"no speech at all\n{n_zs:,} videos ({n_zs/n:.1%})", RASTER_GAP)
bracket(n - n_zp, n, f"no gaps at all\n{n_zp:,} videos ({n_zp/n:.1%})", MUTED)
ax.annotate("gap", xy=(COLS * 1.06, n * 0.55), va="center", fontsize=8.5, fontweight="bold",
            color=RASTER_GAP, annotation_clip=False)
ax.annotate("speech", xy=(COLS * 1.06, n * 0.62), va="center", fontsize=8.5, fontweight="bold",
            color=MUTED, annotation_clip=False)
save(fig, "corpus_raster")


# 2. ridgeline -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
bins = np.linspace(0, 1, 41); centres = (bins[:-1] + bins[1:]) / 2
for i, c in enumerate(CATS):
    h, _ = np.histogram(v.video_gap_ratio[v.category == c], bins=bins, density=True)
    h = np.sqrt(h / h.max()) * 1.6
    means = v.groupby("category").video_gap_ratio.mean()
    lo, hi = means.min(), means.max()
    step = int(round((means[c] - lo) / (hi - lo) * (len(BLUE_RAMP) - 1)))
    ax.fill_between(centres, i, i + h, color=BLUE_RAMP[step], lw=0, zorder=10 - i)
    ax.plot(centres, i + h, color="white", lw=1.2, zorder=10 - i)
    part = v[v.category == c]
    zp = (part.video_n_gaps == 0).mean() * 100; zs = (part.video_gap_ratio == 1.0).mean() * 100
    ax.text(-0.03, i + 0.30, f"{c}  (n={len(part)})", ha="right", va="bottom", fontsize=8.5)
    ax.text(-0.03, i + 0.06, f"no gaps {zp:.1f}%   no speech {zs:.1f}%", ha="right", va="bottom",
            fontsize=7.5, color=MUTED)
ax.set_xlim(0, 1); ax.set_ylim(-0.2, len(CATS) + 1.4); ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("gap ratio (proportion of runtime with no speech)")
ax.text(1.0, -0.16, "density on a square-root scale", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
save(fig, "gapratio_ridgeline")

# 3. threshold sweep -------------------------------------------------------
zero_pause = v.video_n_gaps == 0; zero_speech = v.video_gap_ratio == 1.0
mixed = ~zero_pause & ~zero_speech
longest = v.video_id.map(g.groupby("video_id").gap_duration.max())
words = np.arange(0, 25.1, 0.25); n_v = len(v)
zp = zero_pause.sum() / n_v * 100; zs = zero_speech.sum() / n_v * 100
ins = np.array([(mixed & (longest < w * 60 / 160)).sum() / n_v * 100 for w in words])
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.fill_between(words, 0, zp, color=BLUE, lw=0)
ax.fill_between(words, zp, zp + zs, color=ORANGE, lw=0)
ax.fill_between(words, zp + zs, zp + zs + ins, color=AQUA, lw=0)
ax.plot(words, zp + zs + ins, color="white", lw=1.5)
for y, lbl in ((zp / 2, f"zero-pause  {zp:.1f}%"), (zp + zs / 2, f"zero-speech  {zs:.1f}%"), (zp + zs + 6, "insufficient pause")):
    ax.text(12.5, y, lbl, ha="center", va="center", color="white", fontweight="bold", fontsize=9)
for w in (3, 6, 10, 15, 20):
    t = zp + zs + ins[np.argmin(abs(words - w))]
    ax.plot([w, w], [0, t], color="white", lw=0.8, ls=":", alpha=0.8)
    ax.plot(w, t, "o", ms=5, color=INK, zorder=5)
    ax.annotate(f"{t:.1f}%", (w, t), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)
ax.set_xlim(0, 25); ax.set_ylim(0, 100); ax.set_xlabel("minimum description length (words, at 160 wpm)")
ax.set_ylabel("videos with no placeable description (%)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(False)
save(fig, "threshold_sweep")

# 3b. capacity: k descriptions, one per pause, videos with speech -----------
# The k-th longest gap of each video decides whether it holds k descriptions of
# a given length in separate pauses. Zero-speech videos are left out (one gap
# each, so k >= 2 fails by counting); zero-gap videos stay in and hold none.
sp_ids = v.video_id[~zero_speech]; n_sp = len(sp_ids)
gs = g[g.video_id.isin(sp_ids)]
rank = gs.groupby("video_id").gap_duration.rank(ascending=False, method="first")
words = np.linspace(3, 20, 341)
K_COLOURS = {1: BLUE_RAMP[5], 2: BLUE_RAMP[3], 3: BLUE_RAMP[2], 5: BLUE_RAMP[1]}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.axvline(6, color=MUTED, lw=0.8, ls=":")
for k, c in K_COLOURS.items():
    kth = gs.gap_duration[rank == k].values
    y = np.array([(kth >= w * 60 / 160).sum() / n_sp * 100 for w in words])
    ax.plot(words, y, color=c, lw=2)
    ax.text(2.8, y[0], f"k = {k}", ha="right", va="center", color=c, fontsize=8.5, fontweight="bold")
    y6 = y[np.argmin(abs(words - 6))]
    ax.plot(6, y6, "o", ms=4.5, color=c, zorder=5)
    ax.annotate(f"{y6:.1f}%", (6, y6), textcoords="offset points", xytext=(5, 4), fontsize=8, color=c)
ax.set_xlim(1.6, 20); ax.set_ylim(0, 65); ax.set_xticks([3, 6, 10, 15, 20])
ax.set_xlabel("description length (words, at 160 wpm)")
ax.set_ylabel("videos with speech that can hold k descriptions (%)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
save(fig, "capacity_k")

# 4. audio composition, count vs seconds -----------------------------------
order = ["music", "speech", "sfx", "ambient", "silence"]
cols = dict(zip(order, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
byn = g.audio_bucket.value_counts(normalize=True).reindex(order) * 100
bys = g.groupby("audio_bucket").gap_duration.sum().reindex(order); bys = bys / bys.sum() * 100
fig, ax = plt.subplots(figsize=(7.2, 2.9))
for row, (series, lbl) in enumerate(((bys, "share of gap seconds"), (byn, "share of gaps"))):
    left = 0
    for b in order:
        w = series[b]
        ax.barh(row, w, left=left, height=0.55, color=cols[b], lw=1.6, edgecolor="white")
        if w > 8:
            ax.text(left + w / 2, row, f"{b}\n{w:.1f}%", ha="center", va="center", color="white",
                    fontweight="bold", fontsize=8.5)
        else:
            ax.annotate(f"{b} {w:.1f}%", (left + w / 2, row + (0.36 if b == "ambient" else 0.52)),
                        ha="center", fontsize=8, color=cols[b], fontweight="bold")
        left += w
    ax.text(-1.5, row, lbl, ha="right", va="center", fontsize=9)
ax.set_xlim(0, 100); ax.set_ylim(-0.6, 2.0); ax.set_yticks([]); ax.set_xlabel("%")
ax.spines["left"].set_visible(False)
save(fig, "audio_composition")

# 5. deixis raincloud ------------------------------------------------------
hc = v[(v.hand_coded == "Y") & (v.video_deixis_present == "Y")].copy()
hc["verbal"] = hc.video_modality.fillna("").str.contains("verbal")
groups = [("verbal component", hc[hc.verbal], BLUE), ("non-verbal only", hc[~hc.verbal], ORANGE)]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
xs = np.linspace(0, 1, 200)
for i, (lbl, part, col) in enumerate(groups):
    d = part.video_gap_ratio.values
    kde = stats.gaussian_kde(d, bw_method=0.25)
    dens = kde(xs) + kde(-xs) + kde(2 - xs)  # reflect at both bounds
    dens = dens / dens.max() * 0.38
    ax.fill_between(xs, i + 0.08, i + 0.08 + dens, color=col, alpha=0.6, lw=0)
    y = i - 0.12 + (np.random.default_rng(3).random(len(d)) - 0.5) * 0.14
    ax.scatter(d, y, s=5, color=col, alpha=0.35, lw=0, clip_on=False)
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    ax.plot([q1, q3], [i - 0.3, i - 0.3], color=INK, lw=3, solid_capstyle="butt", clip_on=False)
    ax.plot(med, i - 0.3, "o", ms=5, color="white", zorder=5, markeredgecolor=INK, clip_on=False)
    ax.text(-0.03, i, f"{lbl}\nn={len(d)}, median {med:.2f}", ha="right", va="center", fontsize=8.5)
ax.set_xlim(-0.015, 1.015); ax.set_ylim(-0.6, 2.0); ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("gap ratio")
save(fig, "deixis_raincloud")
