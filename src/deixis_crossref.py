"""
Cross-reference gaps with verbal deixis in the transcript.

deixis_density.score_transcript finds deictic words with timestamps, and
sweep_thresholds measures how many of them have a usable gap nearby.

Only spoken deixis ("this", "here") is covered, because it's the only kind
with timestamps. Gestural deixis (holding something up to the camera, a
reveal pan) counts too, but the hand-coded labels are per video, not per
moment, so there's nothing to line up with gap times. A "no nearby gap"
result says nothing about deixis in general.

deixis_density.py also sorts events into entity, spatial, procedural and
comparative types. That typology isn't supported by the literature, so it
isn't used here.
"""

from __future__ import annotations

import deixis_density as dd


def compute_crossref(whisper_result: dict, gaps: list[tuple[float, float]], duration: float) -> dict:
    """
    whisper_result: output of gap_extract.transcribe_regions(); needs word
        timestamps, which it already requests.
    gaps: this video's gaps as (start, end) tuples.
    """
    report = dd.score_transcript(whisper_result, duration_s=duration)
    sweep = dd.sweep_thresholds(report, gaps)
    return {
        "n_verbal_deixis_events": report.n_events,
        "verbal_deixis_density_per_min": report.density_per_min,
        "events": [
            {"start": e.start, "end": e.end, "text": e.text, "supported": e.supported}
            for e in report.events
        ],
        "coverage_by_threshold": sweep,
    }
