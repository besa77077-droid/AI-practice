"""Cluster -> Hypothesis generation.

Each hypothesis is drafted by the LLM from a handful of representative quotes
in its cluster, but priority is computed deterministically from signal
strength + frequency — never left to the LLM's judgement — so re-running
generation doesn't silently reshuffle backlog priority.
"""

from __future__ import annotations

import hashlib

from insight_engine.models import Cluster, Hypothesis, Insight, SignalStrength
from insight_engine.providers.llm import LLMProvider

MAX_QUOTE_REFS = 4
MIN_FREQUENCY_FOR_HYPOTHESIS = 1


def _hypothesis_id(cluster_id: str) -> str:
    return "hyp_" + hashlib.sha1(cluster_id.encode("utf-8")).hexdigest()[:12]


def _priority(cluster: Cluster) -> SignalStrength:
    if cluster.signal_strength == SignalStrength.HIGH and cluster.frequency >= 2:
        return SignalStrength.HIGH
    if cluster.signal_strength == SignalStrength.HIGH or cluster.frequency >= 3:
        return SignalStrength.HIGH
    if cluster.frequency >= 2 or cluster.signal_strength == SignalStrength.MEDIUM:
        return SignalStrength.MEDIUM
    return SignalStrength.LOW


def generate_hypotheses(
    clusters: list[Cluster], insights_by_id: dict[str, Insight], llm: LLMProvider
) -> list[Hypothesis]:
    hypotheses = []
    for cluster in clusters:
        if cluster.frequency < MIN_FREQUENCY_FOR_HYPOTHESIS:
            continue

        quote_refs = cluster.member_ids[:MAX_QUOTE_REFS]
        quotes = [insights_by_id[mid].quote for mid in quote_refs if mid in insights_by_id]
        raw = llm.draft_hypothesis(cluster.representative_pain, quotes)

        hypotheses.append(
            Hypothesis(
                id=_hypothesis_id(cluster.id),
                cluster_id=cluster.id,
                solution=raw.solution,
                metric=raw.metric,
                metric_direction=raw.metric_direction,
                insight_summary=raw.insight_summary,
                priority=_priority(cluster),
                quote_refs=quote_refs,
            )
        )

    hypotheses.sort(key=lambda h: h.priority.rank, reverse=True)
    return hypotheses
