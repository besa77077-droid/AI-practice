"""Link an LLM-extracted quote back to the exact transcript segment(s) it
came from.

This is the technical backbone of the demo's signature interaction: clicking
a quote in an insight card (or in a hypothesis's basis) smooth-scrolls the
transcript panel to the right reply and flashes it. The LLM is asked to copy
quotes verbatim, but real models still occasionally paraphrase a word or
merge two adjacent turns, so this does fuzzy containment/ratio matching
across sliding windows of segments rather than requiring an exact match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from insight_engine.models import SpeakerSegment

MIN_CONFIDENCE = 0.35
MAX_WINDOW = 3


@dataclass
class QuoteMatch:
    segment_id: int | None
    start: float | None
    end: float | None
    confidence: float


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def match_quote(quote: str, segments: list[SpeakerSegment]) -> QuoteMatch:
    if not quote.strip() or not segments:
        return QuoteMatch(segment_id=None, start=None, end=None, confidence=0.0)

    norm_quote = _normalize(quote)
    norm_segments = [_normalize(seg.text) for seg in segments]

    best = QuoteMatch(segment_id=None, start=None, end=None, confidence=0.0)

    for window in range(1, min(MAX_WINDOW, len(segments)) + 1):
        for start_idx in range(len(segments) - window + 1):
            window_segments = segments[start_idx : start_idx + window]
            window_norm = " ".join(norm_segments[start_idx : start_idx + window])
            if not window_norm:
                continue

            if norm_quote in window_norm or window_norm in norm_quote:
                confidence = 1.0
            else:
                confidence = SequenceMatcher(None, norm_quote, window_norm).ratio()

            if confidence > best.confidence:
                best = QuoteMatch(
                    segment_id=window_segments[0].id,
                    start=window_segments[0].start,
                    end=window_segments[-1].end,
                    confidence=confidence,
                )

    if best.confidence < MIN_CONFIDENCE:
        return QuoteMatch(segment_id=None, start=None, end=None, confidence=best.confidence)
    return best
