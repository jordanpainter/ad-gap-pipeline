"""
Stage 2: classify the audio in each gap as silence, music, sfx or ambient,
using PANNs (Cnn14, AudioSet-pretrained, 527 classes).

Each class is bucketed by walking AudioSet's ontology
(`audioset_ontology.json`, from https://github.com/audioset/ontology) up to
one of its seven root categories:

    Music                              -> music
    Natural sounds                     -> ambient
    Channel, environment and background -> ambient
    Source-ambiguous sounds            -> ambient, except the "Silence" leaf -> silence
    Human sounds                       -> sfx, except the "Speech" subtree  -> speech (QA flag)
    Animal                             -> sfx
    Sounds of things                   -> sfx

"speech" is a fifth bucket. If the top prediction for a gap is in the Speech
subtree, PANNs heard speech where VAD didn't, so it's kept separate for
checking rather than folded into the other four.

Singing, humming and laughter sit under Human sounds in the ontology, so
they land in sfx, not music. The ontology is followed as is.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ONTOLOGY_PATH = ROOT / "audioset_ontology.json"

PANNS_SAMPLE_RATE = 32_000

ROOT_TO_BUCKET = {
    "Music": "music",
    "Natural sounds": "ambient",
    "Channel, environment and background": "ambient",
    "Source-ambiguous sounds": "ambient",
    "Human sounds": "sfx",
    "Animal": "sfx",
    "Sounds of things": "sfx",
}


def _load_ontology() -> list[dict]:
    return json.loads(ONTOLOGY_PATH.read_text())


def _build_label_to_bucket() -> dict[str, str]:
    onto = _load_ontology()
    by_id = {e["id"]: e for e in onto}
    by_name = {e["name"]: e for e in onto}
    parent_of: dict[str, str] = {}
    for e in onto:
        for c in e["child_ids"]:
            parent_of[c] = e["id"]

    def root_name(entry_id: str) -> str:
        cur = entry_id
        while cur in parent_of:
            cur = parent_of[cur]
        return by_id[cur]["name"]

    speech_ids = {by_name["Speech"]["id"]} | set(by_name["Speech"]["child_ids"])
    silence_id = by_name["Silence"]["id"]

    label_to_bucket: dict[str, str] = {}
    for e in onto:
        if e["id"] == silence_id:
            label_to_bucket[e["name"]] = "silence"
        elif e["id"] in speech_ids:
            label_to_bucket[e["name"]] = "speech"
        else:
            label_to_bucket[e["name"]] = ROOT_TO_BUCKET.get(root_name(e["id"]), "sfx")
    return label_to_bucket


LABEL_TO_BUCKET = _build_label_to_bucket()

_at = None


def get_model():
    global _at
    if _at is None:
        from panns_inference import AudioTagging
        _at = AudioTagging(checkpoint_path=None, device="cpu")
    return _at


# Cnn14 needs a minimum number of spectrogram frames (it was trained on 10 s
# clips); shorter input fails at a pooling layer instead of just scoring
# badly. Most gaps are shorter than this, so they're padded on both sides
# with neighbouring audio, clipped to the video's bounds. Padding can pull in
# nearby speech, so classify_gap reports how much it used.
MIN_CLASSIFY_DURATION = 1.0


def load_audio_window(video_path: Path, start: float, end: float) -> np.ndarray:
    out = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-ss", str(start), "-to", str(end),
            "-f", "s16le", "-ac", "1", "-ar", str(PANNS_SAMPLE_RATE),
            "-v", "error", "-",
        ],
        capture_output=True, check=True,
    )
    audio = np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


@dataclass
class GapClassification:
    top_label: str
    top_prob: float
    bucket: str
    top5: list[tuple[str, float]]
    classify_start: float
    classify_end: float
    padded_seconds: float


def classify_gap(
    video_path: Path, start: float, end: float, video_duration: float, top_k: int = 5,
) -> GapClassification:
    duration = end - start
    pad_needed = max(0.0, MIN_CLASSIFY_DURATION - duration) / 2
    classify_start = max(0.0, start - pad_needed)
    classify_end = min(video_duration, end + pad_needed)

    audio = load_audio_window(video_path, classify_start, classify_end)
    padded_seconds = (classify_end - classify_start) - duration

    if audio.size < PANNS_SAMPLE_RATE * 0.3:  # still too short even after padding (near file edge)
        return GapClassification(
            top_label="(insufficient audio)", top_prob=0.0, bucket="silence", top5=[],
            classify_start=classify_start, classify_end=classify_end, padded_seconds=padded_seconds,
        )

    at = get_model()
    from panns_inference import labels as panns_labels

    clipwise_output, _ = at.inference(audio[None, :])
    scores = clipwise_output[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    top5 = [(panns_labels[i], float(scores[i])) for i in top_idx]
    top_label, top_prob = top5[0]
    return GapClassification(
        top_label=top_label,
        top_prob=top_prob,
        bucket=LABEL_TO_BUCKET.get(top_label, "sfx"),
        top5=top5,
        classify_start=classify_start,
        classify_end=classify_end,
        padded_seconds=padded_seconds,
    )
