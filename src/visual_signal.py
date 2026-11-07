"""
Motion magnitude and hard-cut detection for a gap clip, with no model.

A cheap cross-check on frame_describe.py's `action` and `scene` fields.
Both are simple heuristics (mean frame difference, a drop in histogram
correlation) and haven't been validated against hand labels.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Consecutive frames whose histogram correlation drops below this count as
# a hard cut. A common shot-boundary heuristic; not tuned on this data.
CUT_CORRELATION_THRESHOLD = 0.4


def extract_clip(video_path: Path, start: float, end: float, out_path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ss", str(start), "-to", str(end),
            "-c:v", "libx264", "-preset", "ultrafast", "-an", "-v", "error",
            str(out_path),
        ],
        check=True,
    )
    return out_path


@dataclass
class VisualSignal:
    motion_magnitude: float  # mean normalized [0,1] frame-to-frame pixel difference
    has_cut: bool
    cut_time_in_clip: float | None
    min_hist_correlation: float
    n_frames_analyzed: int


def analyze_gap(video_path: Path, start: float, end: float) -> VisualSignal:
    tmp_dir = Path(tempfile.mkdtemp(prefix="gap_signal_"))
    clip_path = extract_clip(video_path, start, end, tmp_dir / "clip.mp4")

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(frames) < 2:
        return VisualSignal(
            motion_magnitude=0.0, has_cut=False, cut_time_in_clip=None,
            min_hist_correlation=1.0, n_frames_analyzed=len(frames),
        )

    diffs = []
    hist_corrs = []
    for a, b in zip(frames[:-1], frames[1:]):
        diffs.append(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))) / 255.0)
        ha = cv2.calcHist([a], [0], None, [64], [0, 256])
        hb = cv2.calcHist([b], [0], None, [64], [0, 256])
        hist_corrs.append(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))

    min_corr = min(hist_corrs)
    has_cut = min_corr < CUT_CORRELATION_THRESHOLD
    cut_idx = hist_corrs.index(min_corr) if has_cut else None
    cut_time = (cut_idx + 1) / fps if cut_idx is not None else None

    return VisualSignal(
        motion_magnitude=round(float(np.mean(diffs)), 4),
        has_cut=has_cut,
        cut_time_in_clip=round(cut_time, 3) if cut_time is not None else None,
        min_hist_correlation=round(float(min_corr), 3),
        n_frames_analyzed=len(frames),
    )
