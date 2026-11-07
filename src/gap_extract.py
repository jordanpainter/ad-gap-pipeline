"""
Stage 1: Silero VAD finds speech, Whisper transcribes it, and everything
VAD didn't mark as speech is a gap.

Gap boundaries come from VAD, not from Whisper's segments. Whisper
hallucinates text over silence (a confident "Thank you.", or one word
repeated in a loop; see https://github.com/openai/whisper/discussions/679),
and no setting of no_speech_threshold, logprob_threshold or
compression_ratio filters all of it out. One 162 s test video with no
VAD-detected speech still came back with four "Thank you." segments.
Running Whisper only inside VAD speech windows means it never sees the
silence. This is the usual fix from that thread: split on VAD first, and
set condition_on_previous_text=False.

The transcript isn't used for the gap statistics. It's kept for checking
results and for the verbal-deixis cross-reference.

No minimum gap length is applied here. Every gap is kept with its duration,
and thresholds are applied at analysis time (see gap_stats.py).

Usage:
    python src/gap_extract.py
    python src/gap_extract.py --video data/raw_videos/abc123.mp4
    python src/gap_extract.py --model mlx-community/whisper-small-mlx
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx_whisper
import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEOS = ROOT / "data" / "raw_videos"
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
GAPS_DIR = ROOT / "data" / "gaps"

VAD_SAMPLE_RATE = 16_000
# Silero's default. 0.3 catches a few more sub-second blips on ASMR-style
# videos, which isn't enough reason to move off the library default.
VAD_THRESHOLD = 0.5
# Pad speech regions so Whisper doesn't get words with clipped starts or
# ends (VAD's frame size can cut off a soft consonant).
VAD_PAD_S = 0.2
# Merge speech regions closer together than this, so a breath between
# words doesn't count as a gap.
MERGE_TOLERANCE_S = 0.15

# Transcript filtering only: which Whisper text to keep inside a speech
# window. Gap boundaries come from VAD.
NO_SPEECH_PROB_MAX = 0.6
AVG_LOGPROB_MIN = -1.0
COMPRESSION_RATIO_MAX = 2.4


@dataclass
class SpeechSpan:
    start: float
    end: float
    text: str


@dataclass
class Gap:
    start: float
    end: float
    duration: float
    position: str  # "start", "middle" or "end" of the video


def probe_duration(video_path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def load_wav_16k(video_path: Path) -> torch.Tensor:
    out = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-f", "s16le", "-ac", "1", "-ar", str(VAD_SAMPLE_RATE),
            "-v", "error", "-",
        ],
        capture_output=True, check=True,
    )
    audio = np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


_vad_model = None


def get_vad_model():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


def detect_speech_regions(video_path: Path, duration: float) -> list[tuple[float, float]]:
    wav = load_wav_16k(video_path)
    raw = get_speech_timestamps(
        wav, get_vad_model(),
        threshold=VAD_THRESHOLD,
        sampling_rate=VAD_SAMPLE_RATE,
        return_seconds=True,
    )
    padded = [
        (max(0.0, r["start"] - VAD_PAD_S), min(duration, r["end"] + VAD_PAD_S))
        for r in raw
    ]
    return merge_regions(padded, MERGE_TOLERANCE_S)


def merge_regions(regions: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if not regions:
        return []
    regions = sorted(regions)
    merged = [regions[0]]
    for start, end in regions[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= tolerance:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def transcribe_regions(video_path: Path, regions: list[tuple[float, float]], model: str) -> dict:
    if not regions:
        return {"segments": [], "language": None}
    clip_timestamps = [t for region in regions for t in region]
    return mlx_whisper.transcribe(
        str(video_path),
        path_or_hf_repo=model,
        word_timestamps=True,
        clip_timestamps=clip_timestamps,
        condition_on_previous_text=False,
        no_speech_threshold=NO_SPEECH_PROB_MAX,
        logprob_threshold=AVG_LOGPROB_MIN,
        compression_ratio_threshold=COMPRESSION_RATIO_MAX,
        verbose=False,
    )


def clean_speech_spans(result: dict) -> list[SpeechSpan]:
    spans: list[SpeechSpan] = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        if seg.get("no_speech_prob", 0.0) > NO_SPEECH_PROB_MAX:
            continue
        if seg.get("avg_logprob", 0.0) < AVG_LOGPROB_MIN:
            continue
        if seg.get("compression_ratio", 0.0) > COMPRESSION_RATIO_MAX:
            continue
        spans.append(SpeechSpan(start=seg["start"], end=seg["end"], text=text))
    spans.sort(key=lambda s: s.start)
    return spans


def compute_gaps(regions: list[tuple[float, float]], duration: float) -> list[Gap]:
    gaps: list[Gap] = []
    cursor = 0.0
    for start, end in regions:
        if start - cursor > 0:
            gaps.append(_make_gap(cursor, start, duration))
        cursor = max(cursor, end)
    if duration - cursor > 0:
        gaps.append(_make_gap(cursor, duration, duration))
    return gaps


def _make_gap(start: float, end: float, duration: float) -> Gap:
    mid = (start + end) / 2
    if mid < duration * 0.1:
        position = "start"
    elif mid > duration * 0.9:
        position = "end"
    else:
        position = "middle"
    return Gap(start=start, end=end, duration=end - start, position=position)


def process_video(video_path: Path, model: str) -> dict:
    video_id = video_path.stem
    duration = probe_duration(video_path)

    speech_regions = detect_speech_regions(video_path, duration)
    gaps = compute_gaps(speech_regions, duration)

    result = transcribe_regions(video_path, speech_regions, model)
    speech_spans = clean_speech_spans(result)

    record = {
        "video_id": video_id,
        "video_path": str(video_path),
        "duration": duration,
        "model": model,
        "vad_threshold": VAD_THRESHOLD,
        "language": result.get("language"),
        "speech_regions_vad": [{"start": s, "end": e} for s, e in speech_regions],
        "speech_segments": [asdict(s) for s in speech_spans],
        "gaps": [asdict(g) for g in gaps],
        "gap_to_runtime_ratio": sum(g.duration for g in gaps) / duration if duration else None,
    }
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=None, help="single video; default: all in data/raw_videos")
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo", help="mlx-community HF repo or local path")
    args = ap.parse_args()

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    GAPS_DIR.mkdir(parents=True, exist_ok=True)

    videos = [args.video] if args.video else sorted(RAW_VIDEOS.glob("*.mp4"))
    if not videos:
        print(f"no videos found in {RAW_VIDEOS}")
        return

    for video_path in videos:
        print(f"processing {video_path.name} ...")
        record = process_video(video_path, args.model)

        out_path = GAPS_DIR / f"{record['video_id']}.json"
        out_path.write_text(json.dumps(record, indent=2))

        n_gaps = len(record["gaps"])
        ratio = record["gap_to_runtime_ratio"]
        print(f"  duration={record['duration']:.1f}s  vad_speech_regions={len(record['speech_regions_vad'])}"
              f"  whisper_segments={len(record['speech_segments'])}"
              f"  gaps={n_gaps}  gap_ratio={ratio:.2%}  -> {out_path}")


if __name__ == "__main__":
    main()
