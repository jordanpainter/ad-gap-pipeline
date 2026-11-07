"""
Stage 3: describe what's on screen during each gap, with Qwen3-VL running
locally through MLX. Exploratory; the gap statistics don't depend on it.

The model gets the gap as a video clip, not a single frame. mlx_vlm's
`video=` input samples frames itself, at about 2 per second with a minimum
of 4, so a 0.2 s gap and a 30 s gap both get a sensible number of frames.
There's no minimum gap length here (PANNs needs one, the VLM doesn't): a
thin description of a short, static gap is a valid result.

The prompt asks for four short fields, based on what AD guidelines agree
should be described (characters, actions, location, time, objects,
on-screen text), cut down to what one gap can show: the subject in focus
(worded close to the corpus's `referent_note` coding field, so the two
can be compared), the action, whether the scene continues, and any
on-screen text. A fifth field, `description`, is one or two sentences of
plain prose for reading, not parsing.

`save_descriptions()` writes each video's results to
`data/descriptions/<video_id>.json`, alongside gap_extract.py's
`data/gaps/<video_id>.json`.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import mlx_vlm
from mlx_vlm.prompt_utils import apply_chat_template

DEFAULT_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
DESCRIPTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "descriptions"

STRUCTURED_PROMPT = """\
This is a short video clip from a moment in a short-form video with no speech. \
Answer in exactly this five-line format, one field per line, no extra commentary:

Subject: <the person/animal/object that is the focus of attention in this clip, or "none" if nothing is being shown/pointed at>
Action: <what physically happens or changes across the clip, in a few words, or "static" if nothing changes>
Scene: <"same" if this looks like a continuation of one ongoing shot/setting, "changed" if there's a cut to a new shot/setting, or "unclear">
Text: <any on-screen text/captions visible, transcribed, or "none">
Description: <a succinct 1-2 sentence plain-language description of what's happening across the whole clip, on this one line>
"""

FIELD_PATTERN = re.compile(r"^(Subject|Action|Scene|Text):\s*(.*)$", re.IGNORECASE | re.MULTILINE)
DESCRIPTION_LABEL = re.compile(r"^Description:\s*", re.IGNORECASE | re.MULTILINE)
BLANK_LINE = re.compile(r"\n\s*\n")

_model = None
_processor = None
_config = None
_loaded_repo = None


def get_model(model_repo: str = DEFAULT_MODEL):
    global _model, _processor, _config, _loaded_repo
    if _model is None or _loaded_repo != model_repo:
        _model, _processor = mlx_vlm.load(model_repo)
        _config = _model.config
        _loaded_repo = model_repo
    return _model, _processor, _config


def extract_clip(video_path: Path, start: float, end: float, out_path: Path) -> Path:
    # Re-encode instead of stream-copying: a stream copy cut away from a
    # keyframe can get broken frame-count metadata, which OpenCV reads as
    # zero frames. The clips are short, so it's cheap.
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


def _probe_frame_count(clip_path: Path) -> tuple[int, float]:
    """Total frames in the clip file.

    Not the number the model sees, which is a smaller sample (see
    `_estimate_sampled_frame_count`). A 60 s clip has about 1,500 frames
    here but the model gets 120.
    """
    cap = cv2.VideoCapture(str(clip_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return n, fps


def _estimate_sampled_frame_count(
    total_frames: int, source_fps: float,
    fps: float = 2.0, min_frames: int = 4, max_frames: int = 768, frame_factor: int = 2,
) -> int:
    """Number of frames the model receives.

    Same formula and defaults as mlx_vlm.utils.load_video, without decoding
    the clip again. Checked against a real call (60 s clip, 120 frames).
    """
    if total_frames <= 0 or source_fps <= 0:
        return 0
    lo = math.ceil(min_frames / frame_factor) * frame_factor
    hi = math.floor(min(max_frames, total_frames) / frame_factor) * frame_factor
    n = total_frames / source_fps * fps
    n = min(max(n, lo), hi, total_frames)
    return math.floor(n / frame_factor) * frame_factor


def _parse_fields(text: str) -> dict[str, str]:
    # Keeps the last match per field. "Thinking" models often redraft their
    # answer two or three times, and the last draft is the one they settle on.
    fields = {m.group(1).lower(): m.group(2).strip() for m in FIELD_PATTERN.finditer(text)}

    # Description is the last field, so taking everything after the first
    # "Description:" would also grab any redrafting that follows. Take the
    # last label and stop at the next blank line: a Thinking model can write
    # a clean sentence and then carry on ("Wait, but need to be precise...")
    # with no label to stop at.
    label_matches = list(DESCRIPTION_LABEL.finditer(text))
    if label_matches:
        tail = text[label_matches[-1].end():]
        cut = BLANK_LINE.search(tail)
        description = tail[:cut.start()] if cut else tail
    else:
        description = ""

    return {
        "subject": fields.get("subject", ""),
        "action": fields.get("action", ""),
        "scene": fields.get("scene", ""),
        "text": fields.get("text", ""),
        "description": description.strip(),
    }


@dataclass
class GapDescription:
    clip_start: float
    clip_end: float
    clip_path: Path
    clip_total_frames: int  # frames in the clip file, not what the model saw
    clip_source_fps: float
    sampled_frame_count: int  # what the model actually received
    raw_response: str
    subject: str
    action: str
    scene: str
    on_screen_text: str
    description: str


def describe_gap(
    video_path: Path, start: float, end: float,
    model_repo: str = DEFAULT_MODEL, prompt: str = STRUCTURED_PROMPT,
    max_tokens: int = 200, repetition_penalty: float | None = None,
) -> GapDescription:
    """Describe one gap.

    Set `repetition_penalty` (about 1.3) for the "Thinking" variants. Without
    it they can loop on a phrase while reasoning and never produce an answer:
    Qwen3-VL-8B-Thinking once repeated "then a bat, then a bird, then a
    fish..." about 80 times until max_tokens cut it off. The Instruct
    variants haven't needed it, so the default is None (no penalty, as in
    mlx_vlm).
    """
    model, processor, config = get_model(model_repo)

    tmp_dir = Path(tempfile.mkdtemp(prefix="gap_clip_"))
    clip_path = extract_clip(video_path, start, end, tmp_dir / "clip.mp4")
    n_frames, source_fps = _probe_frame_count(clip_path)
    sampled_frames = _estimate_sampled_frame_count(n_frames, source_fps)

    formatted = apply_chat_template(processor, config, prompt, video=str(clip_path))
    generate_kwargs = {}
    if repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = repetition_penalty
    result = mlx_vlm.generate(
        model, processor, formatted, video=[str(clip_path)],
        max_tokens=max_tokens, verbose=False, **generate_kwargs,
    )
    text = (result.text if hasattr(result, "text") else str(result)).strip()
    fields = _parse_fields(text)

    return GapDescription(
        clip_start=start, clip_end=end, clip_path=clip_path,
        clip_total_frames=n_frames, clip_source_fps=source_fps,
        sampled_frame_count=sampled_frames,
        raw_response=text,
        subject=fields["subject"], action=fields["action"],
        scene=fields["scene"], on_screen_text=fields["text"],
        description=fields["description"],
    )


def save_descriptions(video_id: str, gaps: list, descriptions: list[GapDescription]) -> Path:
    """Write this video's gap descriptions to data/descriptions/<video_id>.json."""
    DESCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for g, d in zip(gaps, descriptions):
        rows.append({
            "start": g.start, "end": g.end, "duration": g.duration, "position": g.position,
            "subject": d.subject, "action": d.action, "scene": d.scene,
            "on_screen_text": d.on_screen_text, "description": d.description,
            "sampled_frame_count": d.sampled_frame_count,
        })
    out_path = DESCRIPTIONS_DIR / f"{video_id}.json"
    out_path.write_text(json.dumps({"video_id": video_id, "gaps": rows}, indent=2))
    return out_path
