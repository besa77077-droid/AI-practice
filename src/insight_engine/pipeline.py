"""Orchestrates one interview through the full pipeline:

input (audio or text) -> segments -> LLM structural extraction -> quote-to-
segment matching -> storage -> corpus-wide clustering -> hypotheses.

`on_step(name, detail)` is called at each stage boundary so a caller (the
background job runner) can surface real, non-fake progress — this replaces
the old prototype's blocking, silent, all-or-nothing request.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from insight_engine.matching import match_quote
from insight_engine.models import Cluster, Hypothesis, Insight, Interview, SpeakerSegment
from insight_engine.providers.llm import LLMProvider
from insight_engine.providers.transcription import (
    parse_speaker_transcript,
    segments_to_transcript,
    transcribe_audio,
)
from insight_engine.storage.sqlite_store import InsightStore

StepCallback = Callable[[str, str], None]


def _noop(_name: str, _detail: str) -> None:
    return None


@dataclass
class PipelineResult:
    interview: Interview
    insights: list[Insight]
    clusters: list[Cluster]
    hypotheses: list[Hypothesis]


def ingest_audio(
    audio_path: str,
    whisper_model: str = "small",
    diarize: bool = False,
    hf_token: str | None = None,
    on_step: StepCallback = _noop,
) -> list[SpeakerSegment]:
    on_step("transcribe", "Распознавание речи…")
    segments = transcribe_audio(
        audio_path, whisper_model=whisper_model, diarize=diarize, hf_token=hf_token
    )
    words = sum(len(s.text.split()) for s in segments)
    on_step("transcribe", f"Распознано {words} слов, {len(segments)} реплик")
    return segments


def ingest_text(transcript_text: str, on_step: StepCallback = _noop) -> list[SpeakerSegment]:
    on_step("transcribe", "Разбор текстового транскрипта…")
    segments = parse_speaker_transcript(transcript_text)
    speakers = {s.speaker for s in segments}
    on_step("transcribe", f"Разобрано {len(segments)} реплик, {len(speakers)} спикеров")
    return segments


def process_interview(
    *,
    segments: list[SpeakerSegment],
    title: str,
    respondent: str,
    segment_label: str,
    llm: LLMProvider,
    store: InsightStore,
    on_step: StepCallback = _noop,
) -> PipelineResult:
    interview = Interview(
        id=str(uuid.uuid4()),
        title=title,
        respondent=respondent,
        segment_label=segment_label,
        transcript=segments_to_transcript(segments),
        segments=segments,
    )
    store.add_interview(interview)

    on_step("extract", "Структурный разбор LLM по схеме…")
    raw_insights = llm.extract_insights(segments)
    on_step("extract", f"Извлечено {len(raw_insights)} сигналов")

    on_step("match", "Сопоставление цитат с репликами транскрипта…")
    insights: list[Insight] = []
    matched = 0
    for raw in raw_insights:
        match = match_quote(raw.quote, segments)
        if match.segment_id is not None:
            matched += 1
        insights.append(
            Insight(
                id=str(uuid.uuid4()),
                interview_id=interview.id,
                pain=raw.pain,
                context=raw.context,
                workaround=raw.workaround,
                jtbd=raw.jtbd,
                quote=raw.quote,
                signal_strength=raw.signal_strength,
                speaker=raw.speaker,
                segment_id=match.segment_id,
                quote_start=match.start,
                quote_end=match.end,
                match_confidence=match.confidence,
            )
        )
    store.add_insights(insights)
    on_step("match", f"Сопоставлено {matched} из {len(insights)} цитат с транскриптом")

    on_step("cluster", "Дедупликация и кластеризация с корпусом…")
    all_insights_count = len(store.all_accepted_insights())
    clusters, hypotheses = store.recompute(llm)
    on_step(
        "cluster",
        f"Сопоставлено с корпусом из {all_insights_count} инсайтов, "
        f"{len(clusters)} кластеров боли",
    )

    on_step("hypotheses", f"Сформировано {len(hypotheses)} гипотез")

    return PipelineResult(
        interview=interview, insights=insights, clusters=clusters, hypotheses=hypotheses
    )
