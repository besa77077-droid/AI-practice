"""Data model for the interview-synthesis pipeline.

Schema follows the fixed extraction scheme: pain, context, workaround, JTBD,
verbatim quote, signal strength. Unlike a flat quote string, every Insight is
linked to the exact SpeakerSegment it was extracted from (`segment_id`) — this
link is what lets the UI jump from an insight card (or a hypothesis's basis
quote) straight to the highlighted line in the transcript panel and back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SignalStrength(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def label_ru(self) -> str:
        return {"low": "Слабый", "medium": "Средний", "high": "Сильный"}[self.value]

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]


@dataclass
class SpeakerSegment:
    """One diarized/transcribed turn. `id` is stable within an Interview and
    is what Insight.segment_id and the transcript-highlight UI key off of."""

    id: int
    speaker: str
    text: str
    start: float | None = None
    end: float | None = None


@dataclass
class Interview:
    id: str
    title: str
    respondent: str
    segment_label: str
    transcript: str
    segments: list[SpeakerSegment] = field(default_factory=list)
    created_at: str = ""


@dataclass
class Insight:
    id: str
    interview_id: str
    pain: str
    context: str
    workaround: str
    jtbd: str
    quote: str
    signal_strength: SignalStrength
    speaker: str | None = None
    segment_id: int | None = None
    quote_start: float | None = None
    quote_end: float | None = None
    match_confidence: float = 0.0
    status: str = "new"  # new | accepted | rejected


@dataclass
class ClusterMember:
    insight_id: str
    interview_id: str
    interview_title: str
    respondent: str
    quote: str
    segment_id: int | None


@dataclass
class Cluster:
    id: str
    representative_pain: str
    signal_strength: SignalStrength
    member_ids: list[str] = field(default_factory=list)

    @property
    def frequency(self) -> int:
        return len(self.member_ids)


@dataclass
class Hypothesis:
    id: str
    cluster_id: str
    solution: str
    metric: str
    metric_direction: str
    insight_summary: str
    priority: SignalStrength
    quote_refs: list[str] = field(default_factory=list)  # Insight ids
    status: str = "draft"  # draft | confirmed | in_backlog

    @property
    def text(self) -> str:
        return (
            f"Если {self.solution}, то {self.metric} {self.metric_direction}, "
            f"потому что {self.insight_summary}"
        )
