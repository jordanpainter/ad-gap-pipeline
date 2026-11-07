"""
Recompute every descriptive number in the confirmation report's gap
analysis from the full-pool outputs.

Reads data/analysis/{videos_summary,gaps_detail}.csv (the 2026-08-18
full-pool run) and prints a sectioned report. Read-only: writes nothing.

    .venv/bin/python scripts/results_summary.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "data" / "analysis"

POOL_ATTEMPTED = 4605  # videos in the discovered pool, before download
NARRATION_WPM = 160  # ACB (2009) narration pace, the only numeric pacing figure in the guidelines
WORD_FLOORS = (3, 6, 10, 15, 20)
DEFAULT_WORD_FLOOR = 6
CONFIDENT_PROB = 0.6  # PANNs top-1 probability treated as a confident label
LONG_SPEECH_GAP_S = 15.0


def pct(n: float, d: float) -> str:
    return f"{100 * n / d:.1f}%"


def normalise_labels(series: pd.Series) -> pd.Series:
    """'spatial,person' and 'person,spatial' are the same coded set."""
    return series.fillna("").map(lambda s: ",".join(sorted(p.strip() for p in s.split(",") if p.strip())))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    v = pd.read_csv(ANALYSIS_DIR / "videos_summary.csv")
    g = pd.read_csv(ANALYSIS_DIR / "gaps_detail.csv")
    n_v, n_g = len(v), len(g)

    section("Corpus acquisition and stage coverage")
    print(f"pool attempted {POOL_ATTEMPTED}, analysed {n_v} ({pct(n_v, POOL_ATTEMPTED)}), unavailable {POOL_ATTEMPTED - n_v}")
    print(f"gaps {n_g}")
    print(f"stage 1 VAD gaps        {pct(g.gap_duration.notna().sum(), n_g)}")
    print(f"stage 2 PANNs audio     {pct(g.audio_bucket.notna().sum(), n_g)}")
    print(f"stage 3 Qwen3-VL text   {pct(g.description.notna().sum(), n_g)}  (missing {g.description.isna().sum()})")
    print(f"stage 4 motion/cut      {pct(g.motion_magnitude.notna().sum(), n_g)}  (missing {g.motion_magnitude.isna().sum()})")
    per_channel = v.seed_channel.value_counts()
    print(f"seed channels {v.seed_channel.nunique()}, max videos per channel {per_channel.max()}, "
          f"top 10 channels {pct(per_channel.head(10).sum(), n_v)}, top 20 {pct(per_channel.head(20).sum(), n_v)}")

    section("Gap ratio distribution")
    r = v.video_gap_ratio
    print(f"mean {r.mean():.3f}, median {r.median():.3f}")
    print(f"ratio in [0, 0.025): {(r < 0.025).sum()} ({pct((r < 0.025).sum(), n_v)})")
    print(f"ratio in [0.975, 1]: {(r >= 0.975).sum()} ({pct((r >= 0.975).sum(), n_v)})")
    between = ((r >= 0.025) & (r < 0.975))
    print(f"ratio strictly between: {between.sum()} ({pct(between.sum(), n_v)})")
    print(f"ratio exactly 0: {(r == 0).sum()}, exactly 1.0: {(r == 1.0).sum()} ({pct((r == 1.0).sum(), n_v)})")
    print(f"ratio >= 0.5: {pct((r >= 0.5).sum(), n_v)}, >= 0.95: {pct((r >= 0.95).sum(), n_v)}")

    section("Individual gaps")
    d = g.gap_duration
    print(f"duration median {d.median():.2f}s, mean {d.mean():.2f}s, max {d.max():.1f}s")
    print(f"< 1s {pct((d < 1).sum(), n_g)}, < 2s {pct((d < 2).sum(), n_g)}, >= 10s {pct((d >= 10).sum(), n_g)}, "
          f">= 60s {(d >= 60).sum()} ({pct((d >= 60).sum(), n_g)})")
    ng = v.video_n_gaps
    print(f"gaps per video mean {ng.mean():.2f}, median {ng.median():.0f}, max {ng.max()}, "
          f"zero {(ng == 0).sum()}, exactly one {(ng == 1).sum()} ({pct((ng == 1).sum(), n_v)})")
    pos = g.gap_position.value_counts()
    print("position (gap midpoint in first 10% / last 10% / otherwise): "
          + ", ".join(f"{p} {pct(pos.get(p, 0), n_g)}" for p in ("start", "middle", "end")))

    section("By category")
    cat = v.groupby("category").agg(
        n=("video_id", "size"),
        mean_ratio=("video_gap_ratio", "mean"),
        median_ratio=("video_gap_ratio", "median"),
        zero_pause=("video_n_gaps", lambda s: (s == 0).mean() * 100),
        zero_speech=("video_gap_ratio", lambda s: (s == 1.0).mean() * 100),
    ).sort_values("mean_ratio")
    print(cat.round(3).to_string())

    section("Gap ratio vs video duration")
    rho, p_rho = stats.spearmanr(v.video_duration, r)
    pr, p_pr = stats.pearsonr(v.video_duration, r)
    print(f"Spearman rho {rho:.3f} (p={p_rho:.2g}), Pearson r {pr:.3f} (p={p_pr:.2g}), n={n_v}")
    bins = pd.cut(v.video_duration, [0, 15, 30, 60, 120, 300, 10_000], right=False)
    print(v.groupby(bins, observed=True).video_gap_ratio.agg(["size", "mean"]).round(3).to_string())

    section("Structural failure modes")
    zero_pause = v.video_n_gaps == 0
    zero_speech = v.video_gap_ratio == 1.0
    mixed = ~zero_pause & ~zero_speech
    longest = v.video_id.map(g.groupby("video_id").gap_duration.max())
    print(f"zero-pause {zero_pause.sum()} ({pct(zero_pause.sum(), n_v)}), "
          f"zero-speech {zero_speech.sum()} ({pct(zero_speech.sum(), n_v)}), overlap {(zero_pause & zero_speech).sum()}")
    print(f"two-mode combined {(zero_pause | zero_speech).sum()} ({pct((zero_pause | zero_speech).sum(), n_v)}), mixed {mixed.sum()}")
    for words in WORD_FLOORS:
        floor_s = words * 60 / NARRATION_WPM
        insufficient = mixed & (longest < floor_s)
        total = zero_pause.sum() + zero_speech.sum() + insufficient.sum()
        marker = "  <- default" if words == DEFAULT_WORD_FLOOR else ""
        print(f"{words:>2}-word floor ({floor_s:.2f}s): insufficient {insufficient.sum()}, "
              f"three-mode total {total} ({pct(total, n_v)}){marker}")
    floor_s = DEFAULT_WORD_FLOOR * 60 / NARRATION_WPM
    v = v.assign(zero_pause=zero_pause, zero_speech=zero_speech, insufficient=mixed & (longest < floor_s))
    comp = v.groupby("category")[["zero_pause", "zero_speech", "insufficient"]].mean().mul(100)
    comp["three_mode"] = comp.sum(axis=1)
    print(f"by category at {DEFAULT_WORD_FLOOR}-word floor (% of category):")
    print(comp.round(1).sort_values("three_mode").to_string())

    section("Capacity: how many descriptions videos with speech can hold")
    # Denominator is videos with speech, zero-gap ones included (they hold none). Zero-speech
    # videos are one gap each, so they would fail k >= 2 by counting, not because of speech.
    # Separate: k gaps each long enough, one description per pause (the default form).
    # Packed: several descriptions may share one long pause (the upper bound).
    sp_ids = v.video_id[~zero_speech]
    gs = g[g.video_id.isin(sp_ids)]
    print(f"videos with speech {len(sp_ids)}, of which zero-gap {zero_pause.sum()}")
    for words in WORD_FLOORS:
        floor_s = words * 60 / NARRATION_WPM
        sep = (gs.gap_duration >= floor_s).groupby(gs.video_id).sum().reindex(sp_ids, fill_value=0)
        packed = (gs.gap_duration // floor_s).groupby(gs.video_id).sum().reindex(sp_ids, fill_value=0)
        print(f"{words:>2} words  separate: " + ", ".join(f"k>={k} {(sep >= k).sum()} ({pct((sep >= k).sum(), len(sp_ids))})" for k in (1, 2, 3, 5))
              + "  |  packed: " + ", ".join(f"k>={k} {(packed >= k).sum()} ({pct((packed >= k).sum(), len(sp_ids))})" for k in (1, 2, 3, 5)))
    floor_s = DEFAULT_WORD_FLOOR * 60 / NARRATION_WPM
    sep = (gs.gap_duration >= floor_s).groupby(gs.video_id).sum().reindex(sp_ids, fill_value=0)
    dur = v.set_index("video_id").video_duration.reindex(sp_ids)
    bands = pd.cut(dur, [0, 30, 60, 120, 10_000], labels=["<=30s", "30-60s", "60-120s", ">120s"])
    print(f"room for two {DEFAULT_WORD_FLOOR}-word descriptions (separate) by duration: " + ", ".join(
        f"{b} {(s >= 2).mean() * 100:.1f}% (n={len(s)})" for b, s in sep.groupby(bands.values, observed=True)))
    print(f"spearman(duration, separate {DEFAULT_WORD_FLOOR}-word gaps) {stats.spearmanr(dur, sep).statistic:.2f}")

    section("Zero-speech videos: what fills the runtime")
    zs_ids = set(v.video_id[zero_speech])
    gz = g[g.video_id.isin(zs_ids)]
    print(f"zero-speech videos {len(zs_ids)}, each a single gap: {(v.video_n_gaps[zero_speech] == 1).all()}")
    print("bucket share of those gaps: " + ", ".join(
        f"{k} {p:.1f}%" for k, p in (gz.audio_bucket.value_counts(normalize=True) * 100).items()))
    # The speech share is a VAD/PANNs disagreement: Silero at its 0.5 default marks no
    # speech, PANNs hears some. Spot-checked 2026-09-16 by rerunning Silero at 0.3/0.15
    # and transcribing ungated: brief shouts and background speech, typically under 4 s.
    sp = gz[gz.audio_bucket == "speech"]
    print(f"speech-bucket {len(sp)} ({pct(len(sp), len(gz))}), of which confident (>= {CONFIDENT_PROB}) "
          f"{(sp.audio_top_prob >= CONFIDENT_PROB).sum()}; their total {sp.gap_duration.sum():.0f}s "
          f"= {pct(sp.gap_duration.sum(), gz.gap_duration.sum())} of zero-speech gap seconds")
    floor_s_d = DEFAULT_WORD_FLOOR * 60 / NARRATION_WPM
    insufficient_d = mixed & (longest < floor_s_d)
    base = zero_pause.sum() + zero_speech.sum() + insufficient_d.sum()
    print(f"sensitivity, if every speech-bucket zero-speech video were counted as having speech: "
          f"zero-speech {zero_speech.sum() - len(sp)} ({pct(zero_speech.sum() - len(sp), n_v)}), "
          f"three-mode total {base - len(sp)} ({pct(base - len(sp), n_v)}) vs {base} ({pct(base, n_v)}); "
          f"none would become insufficient-pause (each has one gap spanning the video)")

    section("Audio content of gaps (PANNs)")
    bucket = g.audio_bucket.value_counts()
    print("buckets: " + ", ".join(f"{b} {pct(c, n_g)}" for b, c in bucket.items()))
    g_cat = g.merge(v[["video_id", "category"]], on="video_id")
    speech_share = g_cat.groupby("category").audio_bucket.apply(lambda s: (s == "speech").mean() * 100).sort_values()
    music_share = g_cat.groupby("category").audio_bucket.apply(lambda s: (s == "music").mean() * 100)
    print("speech-bucket share by category: " + ", ".join(f"{c} {x:.1f}%" for c, x in speech_share.items()))
    print(f"music-bucket share within Music: {music_share.get('Music', float('nan')):.1f}%")
    print("duration by bucket (median / mean s):")
    print(g.groupby("audio_bucket").gap_duration.agg(["median", "mean"]).round(2).to_string())
    speech = g[g.audio_bucket == "speech"]
    total_s = d.sum()
    print(f"speech-bucket gaps {len(speech)}, {speech.gap_duration.sum():.0f}s = {pct(speech.gap_duration.sum(), total_s)} of all gap seconds")
    long_speech = speech[speech.gap_duration > LONG_SPEECH_GAP_S]
    weak = long_speech[long_speech.audio_top_prob < CONFIDENT_PROB]
    print(f"speech-bucket gaps > {LONG_SPEECH_GAP_S:.0f}s: {len(long_speech)} ({pct(long_speech.gap_duration.sum(), speech.gap_duration.sum())} of speech-bucket seconds), "
          f"of which confident (>= {CONFIDENT_PROB}) {(long_speech.audio_top_prob >= CONFIDENT_PROB).sum()}")
    print(f"long and low-confidence: {len(weak)} gaps, {weak.gap_duration.sum():.0f}s = {pct(weak.gap_duration.sum(), total_s)} of all gap seconds")
    # Appendix D check: some AudioSet classes sit under more than one root
    # category, and the bucket mapping follows a single parent chain for them.
    import json
    from collections import defaultdict
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import audio_classify
    onto = json.loads(audio_classify.ONTOLOGY_PATH.read_text())
    by_id = {e["id"]: e for e in onto}
    by_name = {e["name"]: e for e in onto}
    parents = defaultdict(set)
    for e in onto:
        for c in e["child_ids"]:
            parents[c].add(e["id"])
    def roots(i: str) -> set[str]:
        return {by_id[i]["name"]} if not parents[i] else set().union(*(roots(p) for p in parents[i]))
    ambiguous = {
        name for name, b in audio_classify.LABEL_TO_BUCKET.items()
        if b not in ("speech", "silence") and name in by_name
        and len({audio_classify.ROOT_TO_BUCKET.get(r, "sfx") for r in roots(by_name[name]["id"])}) > 1
    }
    amb = g[g.audio_top_label.isin(ambiguous)]
    print(f"top labels whose roots map to more than one bucket: {len(amb)} gaps = {pct(len(amb), n_g)} of gaps, "
          f"{pct(amb.gap_duration.sum(), total_s)} of gap seconds")

    section("Visual content of gaps (Qwen3-VL, motion)")
    # Each measure is a share of the gaps that have that field: a gap whose clip
    # extraction failed is unknown, not negative. 187 gaps have no stage-3 output at
    # all; 6 more have the structured fields but no free description (the model hit
    # the token cap looping on repetitive on-screen text), so denominators differ by
    # field. Matches how has_cut is counted below.
    print(f"stage-3 coverage: any structured field {g[['subject', 'action', 'scene', 'on_screen_text']].notna().any(axis=1).sum()}, "
          f"description {g.description.notna().sum()}, neither {g[['subject', 'action', 'scene', 'on_screen_text', 'description']].isna().all(axis=1).sum()} of {n_g}")
    text = g.on_screen_text
    n_text = text.notna().sum()
    has_text = text.notna() & (text.str.strip().str.lower() != "none")
    words = text[has_text].str.split().str.len()
    print(f"on-screen text present {has_text.sum()} ({pct(has_text.sum(), n_text)} of {n_text} gaps with the field); "
          f"of those > 3 words {pct((words > 3).sum(), has_text.sum())}, median {words.median():.0f} words")
    action = g.action.str.strip().str.lower()
    static = action == "static"
    print(f"action 'static' {pct(static.sum(), action.notna().sum())} of {action.notna().sum()} gaps with the field")
    scene = g.scene.str.strip().str.lower()
    print(f"scene 'same' {pct((scene == 'same').sum(), scene.notna().sum())} of {scene.notna().sum()} gaps with the field")
    print(f"motion magnitude median, static {g.motion_magnitude[static].median():.3f} vs non-static {g.motion_magnitude[~static & action.notna()].median():.3f}")
    print(f"gaps containing a hard cut {pct((g.has_cut == True).sum(), g.has_cut.notna().sum())}")  # noqa: E712

    section("Deixis cross-reference (hand-coded sample)")
    hc = v[v.hand_coded == "Y"].copy()
    print(f"hand-coded {len(hc)}; presence: " + ", ".join(f"{k} {c}" for k, c in hc.video_deixis_present.value_counts().items()))
    y = hc[hc.video_deixis_present == "Y"].copy()
    y["verbal"] = y.video_modality.fillna("").str.contains("verbal")
    for label, part in (("verbal component", y[y.verbal]), ("non-verbal only", y[~y.verbal])):
        print(f"{label:<17} n={len(part):>3}  gap ratio median {part.video_gap_ratio.median():.3f}, mean {part.video_gap_ratio.mean():.3f}")
    y["modality_set"] = normalise_labels(y.video_modality)
    print("modality combinations (normalised): " + ", ".join(f"{k} {c}" for k, c in y.modality_set.value_counts().items()))
    y["type_set"] = normalise_labels(y.video_levinson_type)
    types = y.groupby("type_set").video_gap_ratio.agg(["size", "median", "mean"]).sort_values("size", ascending=False)
    print("Levinson type (normalised):")
    print(types.round(3).to_string())
    # Not in the report: density is per minute of video, so this correlation
    # mostly measures how much of a video is speech. Kept for reference.
    density_rho, density_p = stats.spearmanr(hc.video_verbal_deixis_density_per_min, hc.video_gap_ratio)
    pool_rho, _ = stats.spearmanr(v.video_verbal_deixis_density_per_min, v.video_gap_ratio)
    print(f"[not reported] verbal deixis density vs gap ratio: Spearman rho {density_rho:.3f} (p={density_p:.2g}, n={len(hc)}); full pool rho {pool_rho:.3f}")
    speech_min = v.video_duration * (1 - v.video_gap_ratio) / 60
    has_speech = speech_min > 0.05
    sp = v[has_speech]
    per_video_rho, _ = stats.spearmanr(sp.video_verbal_deixis_density_per_min, sp.video_gap_ratio)
    per_speech_rho, per_speech_p = stats.spearmanr(sp.video_n_verbal_deixis_events / speech_min[has_speech], sp.video_gap_ratio)
    print(f"[not reported] speech-amount check, videos with >3s speech (n={len(sp)}): "
          f"density per video-minute rho {per_video_rho:.3f}; density per speech-minute rho {per_speech_rho:.3f} (p={per_speech_p:.2g}); "
          f"pool videos with zero lexicon matches {pct((v.video_n_verbal_deixis_events == 0).sum(), len(v))}")
    quart = pd.qcut(hc.video_verbal_deixis_density_per_min.rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    print(hc.groupby(quart, observed=True).agg(
        density_max=("video_verbal_deixis_density_per_min", "max"),
        ratio_median=("video_gap_ratio", "median"),
        ratio_mean=("video_gap_ratio", "mean"),
    ).round(3).to_string())
    hc_gaps = g[g.video_id.isin(hc.video_id)]
    near = hc_gaps.nearest_verbal_deixis_s
    print(f"hand-coded gaps {len(hc_gaps)}; no verbal deixis in video {pct(near.isna().sum(), len(hc_gaps))}; "
          f"where present n={near.notna().sum()}, median distance {near.median():.1f}s, within 2s {pct((near <= 2).sum(), near.notna().sum())}")

    section("Hand-coded sample vs rest of pool")
    rest = v[v.hand_coded != "Y"]
    ks = stats.ks_2samp(hc.video_gap_ratio, rest.video_gap_ratio)
    print(f"KS D={ks.statistic:.3f}, p={ks.pvalue:.3f}")
    print(f"gap ratio mean/median: hand-coded {hc.video_gap_ratio.mean():.3f}/{hc.video_gap_ratio.median():.3f}, "
          f"rest {rest.video_gap_ratio.mean():.3f}/{rest.video_gap_ratio.median():.3f}")
    for label, part in (("hand-coded", hc), ("rest of pool", rest)):
        combined = (part.video_n_gaps == 0).sum() + (part.video_gap_ratio == 1.0).sum()
        print(f"{label:<13} two-mode {pct(combined, len(part))}")
    table = pd.crosstab(v.category, v.hand_coded == "Y")
    chi2, chi_p, dof, _ = stats.chi2_contingency(table)
    print(f"category mix chi2={chi2:.2f}, dof={dof}, p={chi_p:.2g}  (expected: sampling was equal per subgenre, not proportional)")


if __name__ == "__main__":
    main()
