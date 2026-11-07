"""
Deixis density scorer for short-form UGC transcripts.

Takes Whisper output (with word-level timestamps) and returns:
  - overall deixis density (expressions per minute)
  - a per-type breakdown (entity / spatial / procedural / comparative)
  - TIMESTAMPED deixis events

The timestamps matter most. Density alone can rank videos, but timestamps
let you line deictic moments up against the VAD gaps and ask what share of
them have no usable gap within N seconds.

Usage:
    # Whisper with word timestamps
    import whisper
    model = whisper.load_model("small")
    result = model.transcribe("clip.mp4", word_timestamps=True)

    from deixis_density import score_transcript
    report = score_transcript(result)

Lexical detection over-fires ("this is crazy" is not pointing at anything).
Treat the output as a screening filter and check by hand whatever you keep.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict, field

# --------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------
# Multi-word entries are matched first so "over here" doesn't also fire "here".

DEIXIS_LEXICONS: dict[str, list[str]] = {
    "entity": [
        "this one", "that one", "these ones", "those ones",
        "this", "that", "these", "those",
        "the one", "that bit", "this bit", "this part", "that part",
    ],
    "spatial": [
        "over here", "over there", "right here", "right there",
        "back here", "back there", "up here", "up there",
        "down here", "down there", "in here", "in there",
        "behind me", "next to me", "beside me", "in front of me",
        "on my left", "on my right", "this side", "that side",
        "here", "there",
    ],
    "procedural": [
        "and then", "after that", "next up", "first of all",
        "now", "then", "next", "first", "second", "finally", "lastly",
    ],
    "comparative": [
        "compared to", "versus", "vs", "the other one", "this versus",
        "both of these", "either one", "than this", "than that",
    ],
}

# Verbs that introduce a demonstration. A demonstrative near one of these is
# more likely to be pointing at something on screen.
FRAMING_VERBS: list[str] = [
    "look at", "look how", "check out", "check this", "see how", "see this",
    "watch this", "notice", "spot the", "have a look", "take a look",
    "you can see", "you see", "here we have", "here's",
]

# Weak demonstratives that most often appear in non-deictic constructions
# ("this is crazy", "that was wild"). Down-weighted unless supported.
WEAK_PATTERNS = [
    re.compile(r"\bthis is\b"), re.compile(r"\bthat is\b"),
    re.compile(r"\bthat's\b"), re.compile(r"\bthis was\b"),
    re.compile(r"\bthat was\b"),
]

SUPPORT_WINDOW_S = 2.5  # seconds either side, for framing-verb support


@dataclass
class DeixisEvent:
    start: float
    end: float
    text: str
    type: str
    supported: bool = False   # framing verb nearby
    weak: bool = False        # matched a weak construction


@dataclass
class DeixisReport:
    duration_s: float
    n_events: int
    density_per_min: float
    weighted_density_per_min: float
    type_counts: dict[str, int]
    type_proportions: dict[str, float]
    dominant_type: str | None
    events: list[DeixisEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["events"] = [asdict(e) for e in self.events]
        return d


def _clean_lexicons() -> dict[str, list[str]]:
    """Sort each lexicon longest-first so multi-word phrases win."""
    out = {}
    for k, terms in DEIXIS_LEXICONS.items():
        terms = [t for t in terms if isinstance(t, str)]
        out[k] = sorted(set(terms), key=len, reverse=True)
    return out


def _words_from_whisper(result: dict) -> list[dict]:
    """Flatten Whisper segments into a word list with timings."""
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            words.append({
                "word": w["word"].strip().lower().strip(".,!?;:\"'"),
                "start": float(w["start"]),
                "end": float(w["end"]),
            })
    return words


def _find_events(words: list[dict], lexicons: dict[str, list[str]]) -> list[DeixisEvent]:
    events: list[DeixisEvent] = []
    consumed: set[int] = set()

    max_span = max(len(t.split()) for terms in lexicons.values() for t in terms)

    for i in range(len(words)):
        if i in consumed:
            continue
        for span in range(min(max_span, len(words) - i), 0, -1):
            idxs = range(i, i + span)
            if any(j in consumed for j in idxs):
                continue
            phrase = " ".join(words[j]["word"] for j in idxs)
            matched_type = None
            for dtype, terms in lexicons.items():
                if phrase in terms:
                    matched_type = dtype
                    break
            if matched_type:
                events.append(DeixisEvent(
                    start=words[i]["start"],
                    end=words[i + span - 1]["end"],
                    text=phrase,
                    type=matched_type,
                ))
                consumed.update(idxs)
                break
    return events


def _annotate_support(events: list[DeixisEvent], full_text: str,
                      words: list[dict]) -> None:
    """Mark events that sit near a framing verb, and those in weak constructions."""
    verb_times: list[float] = []
    joined = [(w["start"], w["word"]) for w in words]
    text_stream = " ".join(w for _, w in joined)

    for verb in FRAMING_VERBS:
        for m in re.finditer(re.escape(verb), text_stream):
            # approximate the time by word offset
            prefix_words = text_stream[:m.start()].split()
            if prefix_words and len(prefix_words) < len(joined):
                verb_times.append(joined[len(prefix_words)][0])

    for ev in events:
        ev.supported = any(abs(vt - ev.start) <= SUPPORT_WINDOW_S for vt in verb_times)
        window = full_text[max(0, int(ev.start)):]
        ev.weak = any(p.search(f"{ev.text}") for p in WEAK_PATTERNS)


def score_transcript(whisper_result: dict,
                     duration_s: float | None = None) -> DeixisReport:
    """Score a Whisper transcript for deictic density and type distribution."""
    lexicons = _clean_lexicons()
    words = _words_from_whisper(whisper_result)

    if not words:
        return DeixisReport(0.0, 0, 0.0, 0.0, {}, {}, None, [])

    if duration_s is None:
        duration_s = max(w["end"] for w in words)
    minutes = max(duration_s / 60.0, 1e-6)

    full_text = " ".join(w["word"] for w in words)
    events = _find_events(words, lexicons)
    _annotate_support(events, full_text, words)

    # Weighted density: framing-verb support counts double, weak forms count half.
    weight = 0.0
    for ev in events:
        w = 1.0
        if ev.supported:
            w *= 2.0
        if ev.weak and not ev.supported:
            w *= 0.5
        weight += w

    counts = Counter(ev.type for ev in events)
    total = sum(counts.values()) or 1
    proportions = {k: round(v / total, 3) for k, v in counts.items()}

    return DeixisReport(
        duration_s=round(duration_s, 2),
        n_events=len(events),
        density_per_min=round(len(events) / minutes, 2),
        weighted_density_per_min=round(weight / minutes, 2),
        type_counts=dict(counts),
        type_proportions=proportions,
        dominant_type=counts.most_common(1)[0][0] if counts else None,
        events=events,
    )


# --------------------------------------------------------------------------
# Gap coverage
# --------------------------------------------------------------------------

def deixis_gap_coverage(report: DeixisReport,
                        gaps: list[tuple[float, float]],
                        min_gap_s: float = 1.0,
                        search_window_s: float = 2.0) -> dict:
    """
    Cross-reference deictic moments against available speech gaps.

    `gaps` is a list of (start, end) intervals without speech, from Silero
    VAD or from the spaces between Whisper's word timings.

    Returns the proportion of deictic moments with a usable gap close enough
    to describe them in.
    """
    usable = [(s, e) for s, e in gaps if (e - s) >= min_gap_s]

    covered = 0
    for ev in report.events:
        if any(s <= ev.end + search_window_s and e >= ev.start - search_window_s
               for s, e in usable):
            covered += 1

    n = len(report.events)
    return {
        "min_gap_s": min_gap_s,
        "search_window_s": search_window_s,
        "n_deixis_events": n,
        "n_covered": covered,
        "n_uncovered": n - covered,
        "pct_uncovered": round(100.0 * (n - covered) / n, 1) if n else 0.0,
        "n_usable_gaps": len(usable),
        "total_usable_gap_s": round(sum(e - s for s, e in usable), 2),
    }


def sweep_thresholds(report: DeixisReport,
                     gaps: list[tuple[float, float]],
                     thresholds=(0.5, 1.0, 1.5, 2.0, 3.0)) -> list[dict]:
    """Coverage across several gap thresholds, instead of picking one."""
    return [deixis_gap_coverage(report, gaps, min_gap_s=t) for t in thresholds]


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as f:
        result = json.load(f)
    rep = score_transcript(result)
    print(json.dumps(rep.to_dict(), indent=2))
