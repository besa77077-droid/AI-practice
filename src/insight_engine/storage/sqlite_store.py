"""Accumulative SQLite store for interviews, insights, clusters and hypotheses.

Clusters and hypotheses are recomputed from scratch across the *whole* corpus
every time `recompute()` is called (cheap at MVP-1 scale — TF-IDF over a few
hundred insights), but hypothesis `status` (draft/confirmed/in_backlog) is a
human decision and must survive that recomputation, so it's carried over by
matching on `cluster_id` before the new rows are written.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from insight_engine.clustering import cluster_insights
from insight_engine.hypotheses import generate_hypotheses
from insight_engine.models import (
    Cluster,
    ClusterMember,
    Hypothesis,
    Insight,
    Interview,
    SignalStrength,
    SpeakerSegment,
)
from insight_engine.providers.llm import LLMProvider

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interviews (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    respondent TEXT NOT NULL,
    segment_label TEXT NOT NULL,
    transcript TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    interview_id TEXT NOT NULL,
    seg_id INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    start REAL,
    end REAL,
    PRIMARY KEY (interview_id, seg_id)
);

CREATE TABLE IF NOT EXISTS insights (
    id TEXT PRIMARY KEY,
    interview_id TEXT NOT NULL,
    pain TEXT NOT NULL,
    context TEXT NOT NULL,
    workaround TEXT NOT NULL,
    jtbd TEXT NOT NULL,
    quote TEXT NOT NULL,
    signal_strength TEXT NOT NULL,
    speaker TEXT,
    segment_id INTEGER,
    quote_start REAL,
    quote_end REAL,
    match_confidence REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'new'
);

CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    representative_pain TEXT NOT NULL,
    signal_strength TEXT NOT NULL,
    member_ids TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    solution TEXT NOT NULL,
    metric TEXT NOT NULL,
    metric_direction TEXT NOT NULL,
    insight_summary TEXT NOT NULL,
    priority TEXT NOT NULL,
    quote_refs TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft'
);
"""


class InsightStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- interviews ---------------------------------------------------

    def add_interview(self, interview: Interview) -> None:
        created_at = interview.created_at or datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT INTO interviews (id, title, respondent, segment_label, "
                "transcript, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (interview.id, interview.title, interview.respondent, interview.segment_label,
                 interview.transcript, created_at),
            )
            self._conn.executemany(
                "INSERT INTO segments (interview_id, seg_id, speaker, text, start, end) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (interview.id, seg.id, seg.speaker, seg.text, seg.start, seg.end)
                    for seg in interview.segments
                ],
            )

    def list_interviews(self) -> list[Interview]:
        rows = self._conn.execute(
            "SELECT id, title, respondent, segment_label, transcript, created_at "
            "FROM interviews ORDER BY created_at DESC"
        ).fetchall()
        return [
            Interview(
                id=r["id"], title=r["title"], respondent=r["respondent"],
                segment_label=r["segment_label"], transcript=r["transcript"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_interview(self, interview_id: str) -> Interview | None:
        row = self._conn.execute(
            "SELECT id, title, respondent, segment_label, transcript, created_at "
            "FROM interviews WHERE id = ?",
            (interview_id,),
        ).fetchone()
        if row is None:
            return None
        seg_rows = self._conn.execute(
            "SELECT seg_id, speaker, text, start, end FROM segments "
            "WHERE interview_id = ? ORDER BY seg_id",
            (interview_id,),
        ).fetchall()
        segments = [
            SpeakerSegment(id=s["seg_id"], speaker=s["speaker"], text=s["text"],
                            start=s["start"], end=s["end"])
            for s in seg_rows
        ]
        return Interview(
            id=row["id"], title=row["title"], respondent=row["respondent"],
            segment_label=row["segment_label"], transcript=row["transcript"],
            created_at=row["created_at"], segments=segments,
        )

    # -- insights -------------------------------------------------------

    def add_insights(self, insights: list[Insight]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT INTO insights (id, interview_id, pain, context, workaround, "
                "jtbd, quote, signal_strength, speaker, segment_id, quote_start, "
                "quote_end, match_confidence, status) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (i.id, i.interview_id, i.pain, i.context, i.workaround, i.jtbd,
                     i.quote, i.signal_strength.value, i.speaker, i.segment_id,
                     i.quote_start, i.quote_end, i.match_confidence, i.status)
                    for i in insights
                ],
            )

    def list_insights_for_interview(self, interview_id: str) -> list[Insight]:
        rows = self._conn.execute(
            "SELECT * FROM insights WHERE interview_id = ?", (interview_id,)
        ).fetchall()
        return [_row_to_insight(r) for r in rows]

    def all_accepted_insights(self) -> list[Insight]:
        rows = self._conn.execute(
            "SELECT * FROM insights WHERE status != 'rejected'"
        ).fetchall()
        return [_row_to_insight(r) for r in rows]

    def get_insight(self, insight_id: str) -> Insight | None:
        row = self._conn.execute(
            "SELECT * FROM insights WHERE id = ?", (insight_id,)
        ).fetchone()
        return _row_to_insight(row) if row else None

    def update_insight_status(self, insight_id: str, status: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE insights SET status = ? WHERE id = ?", (status, insight_id)
            )

    # -- clusters / hypotheses ------------------------------------------

    def recompute(self, llm: LLMProvider) -> tuple[list[Cluster], list[Hypothesis]]:
        insights = self.all_accepted_insights()
        insights_by_id = {i.id: i for i in insights}
        clusters = cluster_insights(insights)

        previous_status = {
            r["cluster_id"]: r["status"]
            for r in self._conn.execute("SELECT cluster_id, status FROM hypotheses")
        }

        hypotheses = generate_hypotheses(clusters, insights_by_id, llm)
        for h in hypotheses:
            if h.cluster_id in previous_status:
                h.status = previous_status[h.cluster_id]

        with self._conn:
            self._conn.execute("DELETE FROM clusters")
            self._conn.execute("DELETE FROM hypotheses")
            self._conn.executemany(
                "INSERT INTO clusters (id, representative_pain, signal_strength, "
                "member_ids) VALUES (?, ?, ?, ?)",
                [
                    (c.id, c.representative_pain, c.signal_strength.value,
                     json.dumps(c.member_ids))
                    for c in clusters
                ],
            )
            self._conn.executemany(
                "INSERT INTO hypotheses (id, cluster_id, solution, metric, "
                "metric_direction, insight_summary, priority, quote_refs, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (h.id, h.cluster_id, h.solution, h.metric, h.metric_direction,
                     h.insight_summary, h.priority.value, json.dumps(h.quote_refs),
                     h.status)
                    for h in hypotheses
                ],
            )

        return clusters, hypotheses

    def list_hypotheses_ranked(self) -> list[Hypothesis]:
        rows = self._conn.execute("SELECT * FROM hypotheses").fetchall()
        hyps = [_row_to_hypothesis(r) for r in rows]
        hyps.sort(key=lambda h: h.priority.rank, reverse=True)
        return hyps

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        row = self._conn.execute(
            "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
        ).fetchone()
        return _row_to_hypothesis(row) if row else None

    def update_hypothesis_status(self, hypothesis_id: str, status: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE hypotheses SET status = ? WHERE id = ?", (status, hypothesis_id)
            )

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        row = self._conn.execute(
            "SELECT * FROM clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
        if row is None:
            return None
        return Cluster(
            id=row["id"], representative_pain=row["representative_pain"],
            signal_strength=SignalStrength(row["signal_strength"]),
            member_ids=json.loads(row["member_ids"]),
        )

    def get_cluster_for_insight(self, insight_id: str) -> Cluster | None:
        rows = self._conn.execute("SELECT * FROM clusters").fetchall()
        for row in rows:
            member_ids = json.loads(row["member_ids"])
            if insight_id in member_ids:
                return Cluster(
                    id=row["id"], representative_pain=row["representative_pain"],
                    signal_strength=SignalStrength(row["signal_strength"]),
                    member_ids=member_ids,
                )
        return None

    def get_cluster_members_detail(self, cluster_id: str) -> list[ClusterMember]:
        cluster = self.get_cluster(cluster_id)
        if cluster is None:
            return []
        members = []
        for insight_id in cluster.member_ids:
            insight = self.get_insight(insight_id)
            if insight is None:
                continue
            interview = self.get_interview(insight.interview_id)
            members.append(
                ClusterMember(
                    insight_id=insight.id,
                    interview_id=insight.interview_id,
                    interview_title=interview.title if interview else "",
                    respondent=interview.respondent if interview else "",
                    quote=insight.quote,
                    segment_id=insight.segment_id,
                )
            )
        return members


def _row_to_insight(row: sqlite3.Row) -> Insight:
    return Insight(
        id=row["id"], interview_id=row["interview_id"], pain=row["pain"],
        context=row["context"], workaround=row["workaround"], jtbd=row["jtbd"],
        quote=row["quote"], signal_strength=SignalStrength(row["signal_strength"]),
        speaker=row["speaker"], segment_id=row["segment_id"],
        quote_start=row["quote_start"], quote_end=row["quote_end"],
        match_confidence=row["match_confidence"], status=row["status"],
    )


def _row_to_hypothesis(row: sqlite3.Row) -> Hypothesis:
    return Hypothesis(
        id=row["id"], cluster_id=row["cluster_id"], solution=row["solution"],
        metric=row["metric"], metric_direction=row["metric_direction"],
        insight_summary=row["insight_summary"], priority=SignalStrength(row["priority"]),
        quote_refs=json.loads(row["quote_refs"]), status=row["status"],
    )
