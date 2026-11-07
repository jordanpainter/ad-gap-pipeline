"""
Run the whole pipeline over a corpus of Shorts.

Runs one stage at a time across every video rather than one video at a
time through every stage. Each stage loads its model once, runs it over
every gap and releases it before the next one loads, so only one large
model is in memory at a time.

  Stage 1 (Silero VAD + Whisper): gaps + transcript, per video
  Stage 2 (PANNs):                audio bucket, per gap
  Stage 3 (Qwen3-VL):             structured description, per gap
  Stage 4 (OpenCV, no model):     motion/cut signal, per gap

The default pool is the 730 hand-coded videos in all_coded_master.csv;
--pool all uses the full 4,605-video manifest. Videos missing from
data/raw_videos/ are downloaded first.

State is saved to data/analysis/gaps_working.json after every stage, so an
interrupted run picks up where it stopped, per gap. Each run ends by
exporting:

  data/analysis/gaps_detail.csv     one row per gap
  data/analysis/videos_summary.csv  one row per video
  data/analysis/failures.csv        downloads and stages that failed

Usage:
    python src/batch_analyze.py                # all 4 stages, hand-coded pool
    python src/batch_analyze.py --stage 1       # just stage 1 across the pool
    python src/batch_analyze.py --force         # redo every stage
    python src/batch_analyze.py --limit 20      # first 20 videos only (smoke test)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import mlx.core as mx

import gap_extract
import deixis_crossref
import audio_classify
import frame_describe
import visual_signal

ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEOS = ROOT / "data" / "raw_videos"
ANALYSIS_DIR = ROOT / "data" / "analysis"
WORKING_JSON = ANALYSIS_DIR / "gaps_working.json"
GAPS_DETAIL_CSV = ANALYSIS_DIR / "gaps_detail.csv"
VIDEOS_SUMMARY_CSV = ANALYSIS_DIR / "videos_summary.csv"
FAILURES_CSV = ANALYSIS_DIR / "failures.csv"

# Corpus manifests aren't in the repo. Set AD_GAP_CORPUS_DIR to point at yours.
CORPUS_DIR = Path(os.environ.get("AD_GAP_CORPUS_DIR", Path.home() / "data" / "ad_gap_analysis"))
ALL_CODED_CSV = CORPUS_DIR / "all_coded_master.csv"     # 730 hand-coded videos
ALL_VIDEOS_CSV = CORPUS_DIR / "all_videos_master.csv"   # full 4,605-video pool, has its own hand_coded column

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
VLM_MAX_TOKENS = 350
VLM_REPETITION_PENALTY = 1.3  # stops the occasional repetition loop, see frame_describe.py

# One row per gap. Video-level fields live in videos_summary.csv; merge on
# video_id if you want them side by side.
GAP_FIELDS = [
    "video_id",
    "gap_index", "gap_start", "gap_end", "gap_duration", "gap_position",
    "nearest_verbal_deixis_s",
    "audio_bucket", "audio_top_label", "audio_top_prob", "audio_top5",
    "audio_classify_start", "audio_classify_end", "audio_padded_seconds",
    "subject", "action", "scene", "on_screen_text", "description",
    "clip_total_frames", "sampled_frame_count",
    "motion_magnitude", "has_cut", "min_hist_correlation", "motion_n_frames_analyzed",
]

# One row per video, including the hand-coded fields.
VIDEO_FIELDS = [
    "video_id", "category", "subgenre", "seed_channel", "video_title", "video_url",
    "hand_coded", "video_deixis_present", "video_modality", "video_levinson_type", "video_referent_note",
    "video_duration", "video_language", "video_n_gaps", "video_gap_ratio",
    "video_n_verbal_deixis_events", "video_verbal_deixis_density_per_min",
]


# --------------------------------------------------------------------------
# Working-state persistence
# --------------------------------------------------------------------------

def load_working() -> dict:
    if WORKING_JSON.exists():
        state = json.loads(WORKING_JSON.read_text())
        # old format: a bare list of gap rows
        if isinstance(state, list):
            return {"gap_rows": state, "stage1_done_ids": sorted({r["video_id"] for r in state}), "video_meta": {}}
        state.setdefault("video_meta", {})
        state.setdefault("failures", [])
        return state
    return {"gap_rows": [], "stage1_done_ids": [], "video_meta": {}, "failures": []}


def record_failure(state: dict, video_id: str, stage: str, reason: str, gap_index: int | None = None) -> None:
    """Log a failure to the working state.

    A video that fails to download or fails stage 1 adds no rows to either
    CSV, so without this it would look the same as one not processed yet.
    """
    state["failures"].append({
        "video_id": video_id, "stage": stage, "gap_index": gap_index, "reason": reason,
    })


def save_working(state: dict) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    WORKING_JSON.write_text(json.dumps(state, indent=2))


def load_targets(pool: str) -> list[dict]:
    """Load the video manifest for a pool.

    "handcoded": all_coded_master.csv, the 730 hand-coded videos. It has no
    hand_coded column, so one is added.
    "all": all_videos_master.csv, the full 4,605-video pool, with its own
    hand_coded Y/N column. Only the hand-coded rows have the deixis fields
    filled in.
    """
    if pool == "handcoded":
        rows = list(csv.DictReader(open(ALL_CODED_CSV)))
        for r in rows:
            r["hand_coded"] = "Y"
        return rows
    return list(csv.DictReader(open(ALL_VIDEOS_CSV)))


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def ensure_downloaded(video_id: str) -> tuple[Path | None, str | None]:
    """Return (path, error_reason); exactly one of them is None."""
    path = RAW_VIDEOS / f"{video_id}.mp4"
    if path.exists():
        return path, None
    print(f"    downloading {video_id} ...")
    try:
        subprocess.run(
            [
                "yt-dlp", "--cookies-from-browser", "safari",
                "-f", "mp4/best", "-o", str(RAW_VIDEOS / f"{video_id}.%(ext)s"),
                f"https://www.youtube.com/shorts/{video_id}",
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        reason = (e.stderr or e.stdout or str(e)).strip().splitlines()[-1] if (e.stderr or e.stdout) else str(e)
        print(f"    download FAILED: {reason}")
        return None, reason
    if not path.exists():
        return None, "yt-dlp exited cleanly but no file was written"
    return path, None


# --------------------------------------------------------------------------
# Stage 1: VAD + Whisper (gaps and transcript), per video
# --------------------------------------------------------------------------

def run_stage1(targets: list[dict], state: dict, force: bool, limit: int | None) -> dict:
    rows = state["gap_rows"]
    done_video_ids = set() if force else set(state["stage1_done_ids"])
    todo = [t for t in targets if t["video_id"] not in done_video_ids]
    if limit:
        todo = todo[:limit]
    print(f"\n=== Stage 1 (VAD + Whisper): {len(todo)}/{len(targets)} videos to do ===")

    todo_ids = {t["video_id"] for t in todo}
    rows = [r for r in rows if r["video_id"] not in todo_ids]  # drop stale rows for redo
    state["stage1_done_ids"] = [v for v in state["stage1_done_ids"] if v not in todo_ids]

    for i, target in enumerate(todo, 1):
        video_id = target["video_id"]
        print(f"  [{i}/{len(todo)}] {video_id} ...")
        video_path, dl_error = ensure_downloaded(video_id)
        if video_path is None:
            record_failure(state, video_id, "download", dl_error or "unknown download failure")
            continue

        try:
            duration = gap_extract.probe_duration(video_path)
            speech_regions = gap_extract.detect_speech_regions(video_path, duration)
            gaps = gap_extract.compute_gaps(speech_regions, duration)
            whisper_result = gap_extract.transcribe_regions(video_path, speech_regions, WHISPER_MODEL)
            speech_spans = gap_extract.clean_speech_spans(whisper_result)
        except Exception as e:
            print(f"    FAILED: {e}")
            record_failure(state, video_id, "stage1", str(e))
            continue

        gap_record = {
            "video_id": video_id, "video_path": str(video_path), "duration": duration,
            "model": WHISPER_MODEL, "vad_threshold": gap_extract.VAD_THRESHOLD,
            "language": whisper_result.get("language"),
            "speech_regions_vad": [{"start": s, "end": e} for s, e in speech_regions],
            "speech_segments": [s.__dict__ for s in speech_spans],
            "gaps": [g.__dict__ for g in gaps],
            "gap_to_runtime_ratio": sum(g.duration for g in gaps) / duration if duration else None,
        }
        (ROOT / "data" / "gaps" / f"{video_id}.json").write_text(json.dumps(gap_record, indent=2))

        xref = deixis_crossref.compute_crossref(whisper_result, [(g.start, g.end) for g in gaps], duration)

        state["video_meta"][video_id] = {
            "video_id": video_id, "category": target["category"], "subgenre": target["subgenre"],
            "seed_channel": target["seed_channel"], "video_title": target["title"],
            "video_url": target["url"], "hand_coded": target["hand_coded"],
            "video_deixis_present": target["deixis_present_Y_N_unclear"],
            "video_modality": target["modality_verbal_gestural_both"],
            "video_levinson_type": target["levinson_type"],
            "video_referent_note": target["referent_note"],
            "video_duration": round(duration, 2), "video_language": whisper_result.get("language"),
            "video_n_gaps": len(gaps),
            "video_gap_ratio": round(gap_record["gap_to_runtime_ratio"], 4) if gap_record["gap_to_runtime_ratio"] is not None else None,
            "video_n_verbal_deixis_events": xref["n_verbal_deixis_events"],
            "video_verbal_deixis_density_per_min": xref["verbal_deixis_density_per_min"],
        }

        # gap rows only carry video_id; video-level fields are in video_meta
        for idx, g in enumerate(gaps):
            dists = [max(0.0, e["start"] - g.end, g.start - e["end"]) for e in xref["events"]]
            rows.append({
                "video_id": video_id,
                "gap_index": idx, "gap_start": round(g.start, 2), "gap_end": round(g.end, 2),
                "gap_duration": round(g.duration, 2), "gap_position": g.position,
                "nearest_verbal_deixis_s": round(min(dists), 2) if dists else None,
                "audio_bucket": None, "audio_top_label": None, "audio_top_prob": None, "audio_top5": None,
                "audio_classify_start": None, "audio_classify_end": None, "audio_padded_seconds": None,
                "subject": None, "action": None, "scene": None, "on_screen_text": None, "description": None,
                "clip_total_frames": None, "sampled_frame_count": None,
                "motion_magnitude": None, "has_cut": None, "min_hist_correlation": None, "motion_n_frames_analyzed": None,
                "stage1_done": True, "stage2_done": False, "stage3_done": False, "stage4_done": False,
            })
        # Tracked separately from gap rows: a video with no gaps adds no rows,
        # so it would otherwise be redone on every rerun.
        state["stage1_done_ids"].append(video_id)
        state["gap_rows"] = rows
        save_working(state)
        print(f"    duration={duration:.1f}s  gaps={len(gaps)}  gap_ratio={gap_record['gap_to_runtime_ratio']:.1%}")
        # MLX keeps its GPU buffer cache between calls; without this, memory
        # keeps climbing over a long run
        mx.clear_cache()

    state["gap_rows"] = rows
    return state


# --------------------------------------------------------------------------
# Stage 2: PANNs audio classification, per gap
# --------------------------------------------------------------------------

def run_stage2(state: dict, force: bool) -> dict:
    rows = state["gap_rows"]
    todo = [r for r in rows if force or not r.get("stage2_done")]
    print(f"\n=== Stage 2 (PANNs audio): {len(todo)}/{len(rows)} gaps to do ===")

    for i, r in enumerate(todo, 1):
        video_path = RAW_VIDEOS / f"{r['video_id']}.mp4"
        video_duration = state["video_meta"][r["video_id"]]["video_duration"]
        try:
            result = audio_classify.classify_gap(video_path, r["gap_start"], r["gap_end"], video_duration)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {r['video_id']} gap {r['gap_index']} FAILED: {e}")
            record_failure(state, r["video_id"], "stage2", str(e), r["gap_index"])
            continue
        r.update({
            "audio_bucket": result.bucket, "audio_top_label": result.top_label,
            "audio_top_prob": result.top_prob,
            "audio_top5": "; ".join(f"{lbl}:{p:.3f}" for lbl, p in result.top5),
            "audio_classify_start": result.classify_start, "audio_classify_end": result.classify_end,
            "audio_padded_seconds": result.padded_seconds, "stage2_done": True,
        })
        if i % 10 == 0 or i == len(todo):
            save_working(state)
            print(f"  [{i}/{len(todo)}] ...")

    save_working(state)
    audio_classify._at = None  # release before the next stage's model loads
    return state


# --------------------------------------------------------------------------
# Stage 3: Qwen3-VL structured description, per gap
# --------------------------------------------------------------------------

def run_stage3(state: dict, force: bool) -> dict:
    rows = state["gap_rows"]
    todo = [r for r in rows if force or not r.get("stage3_done")]
    print(f"\n=== Stage 3 (Qwen3-VL): {len(todo)}/{len(rows)} gaps to do ===")

    for i, r in enumerate(todo, 1):
        video_path = RAW_VIDEOS / f"{r['video_id']}.mp4"
        try:
            desc = frame_describe.describe_gap(
                video_path, r["gap_start"], r["gap_end"],
                max_tokens=VLM_MAX_TOKENS, repetition_penalty=VLM_REPETITION_PENALTY,
            )
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {r['video_id']} gap {r['gap_index']} FAILED: {e}")
            record_failure(state, r["video_id"], "stage3", str(e), r["gap_index"])
            continue
        r.update({
            "subject": desc.subject, "action": desc.action, "scene": desc.scene,
            "on_screen_text": desc.on_screen_text, "description": desc.description,
            "clip_total_frames": desc.clip_total_frames, "sampled_frame_count": desc.sampled_frame_count,
            "stage3_done": True,
        })
        mx.clear_cache()
        # remove the extracted clip; thousands of these pile up in temp otherwise
        try:
            desc.clip_path.parent.rmdir() if not any(desc.clip_path.parent.iterdir()) else None
        except Exception:
            import shutil
            shutil.rmtree(desc.clip_path.parent, ignore_errors=True)
        if i % 10 == 0 or i == len(todo):
            save_working(state)
            print(f"  [{i}/{len(todo)}] ...")

    save_working(state)
    frame_describe._model = None
    frame_describe._processor = None
    frame_describe._loaded_repo = None
    return state


# --------------------------------------------------------------------------
# Stage 4: motion and cut signal (no model), per gap
# --------------------------------------------------------------------------

def run_stage4(state: dict, force: bool) -> dict:
    rows = state["gap_rows"]
    todo = [r for r in rows if force or not r.get("stage4_done")]
    print(f"\n=== Stage 4 (motion/cut signal): {len(todo)}/{len(rows)} gaps to do ===")

    for i, r in enumerate(todo, 1):
        video_path = RAW_VIDEOS / f"{r['video_id']}.mp4"
        try:
            sig = visual_signal.analyze_gap(video_path, r["gap_start"], r["gap_end"])
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {r['video_id']} gap {r['gap_index']} FAILED: {e}")
            record_failure(state, r["video_id"], "stage4", str(e), r["gap_index"])
            continue
        r.update({
            "motion_magnitude": sig.motion_magnitude, "has_cut": sig.has_cut,
            "min_hist_correlation": sig.min_hist_correlation,
            "motion_n_frames_analyzed": sig.n_frames_analyzed, "stage4_done": True,
        })
        if i % 25 == 0 or i == len(todo):
            save_working(state)
            print(f"  [{i}/{len(todo)}] ...")

    save_working(state)
    return state


# --------------------------------------------------------------------------
# CSV export
# --------------------------------------------------------------------------

def export_csvs(state: dict) -> None:
    rows = state["gap_rows"]
    with open(GAPS_DETAIL_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GAP_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # written from video_meta, not gap_rows, so videos with no gaps are kept
    video_meta = state["video_meta"]
    with open(VIDEOS_SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VIDEO_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in video_meta.values():
            w.writerow(r)

    failure_fields = ["video_id", "stage", "gap_index", "reason"]
    with open(FAILURES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=failure_fields)
        w.writeheader()
        for r in state["failures"]:
            w.writerow(r)

    print(f"\nexported {len(rows)} gap rows / {len(video_meta)} videos -> {GAPS_DETAIL_CSV}, {VIDEOS_SUMMARY_CSV}")
    print(f"exported {len(state['failures'])} failures -> {FAILURES_CSV}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["1", "2", "3", "4", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="redo the selected stage(s) even if already done")
    ap.add_argument("--limit", type=int, default=None, help="only the first N videos (stage 1), for smoke testing")
    ap.add_argument("--pool", choices=["handcoded", "all"], default="handcoded",
                     help="handcoded = 730 hand-coded videos (default); all = full 4,605-video pool")
    args = ap.parse_args()

    targets = load_targets(args.pool)
    state = load_working()
    print(f"{len(targets)} videos in the '{args.pool}' pool, "
          f"{len(state['stage1_done_ids'])} videos / {len(state['gap_rows'])} gap rows in working state so far")

    if args.stage in ("1", "all"):
        state = run_stage1(targets, state, args.force, args.limit)
    if args.stage in ("2", "all"):
        state = run_stage2(state, args.force)
    if args.stage in ("3", "all"):
        state = run_stage3(state, args.force)
    if args.stage in ("4", "all"):
        state = run_stage4(state, args.force)

    export_csvs(state)

    # every target should end up either processed or logged as a failure
    if args.stage in ("1", "all") and not args.limit:
        target_ids = {t["video_id"] for t in targets}
        video_level_failure_ids = {f["video_id"] for f in state["failures"] if f["stage"] in ("download", "stage1")}
        accounted = set(state["video_meta"]) | video_level_failure_ids
        unaccounted = target_ids - accounted
        print(f"\nreconciliation: {len(target_ids)} targets, {len(state['video_meta'])} processed, "
              f"{len(video_level_failure_ids)} failed at video level, {len(unaccounted)} unaccounted for")
        if unaccounted:
            print(f"  UNACCOUNTED (should be empty): {sorted(unaccounted)[:20]}"
                  + (" ..." if len(unaccounted) > 20 else ""))


if __name__ == "__main__":
    main()
