"""Dedup/clustering of insights across the whole accumulated corpus.

TF-IDF + cosine similarity, no embedding downloads — works fully offline and
fast enough to re-run on every new interview for a corpus of the size an
MVP-1 pilot will actually see (dozens of interviews, hundreds of insights).
"""

from __future__ import annotations

import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from insight_engine.models import Cluster, Insight, SignalStrength

SIMILARITY_THRESHOLD = 0.3


def _cluster_id(representative_pain: str) -> str:
    return "cl_" + hashlib.sha1(representative_pain.encode("utf-8")).hexdigest()[:12]


def cluster_insights(insights: list[Insight]) -> list[Cluster]:
    """Greedy single-pass clustering: each insight joins the most similar
    existing cluster if above SIMILARITY_THRESHOLD, else starts a new one.

    Greedy (not full agglomerative) is deliberate: clusters must stay stable
    as new interviews are added one at a time — re-running full clustering
    on every upload would silently reshuffle cluster identity/history.
    """
    accepted = [i for i in insights if i.status != "rejected"]
    if not accepted:
        return []

    texts = [f"{i.pain} {i.jtbd}" for i in accepted]
    if len(texts) == 1:
        vectors = None
    else:
        # Character n-grams, not word-level: Russian's rich inflection means
        # "диапазона"/"диапазон"/"диапазоны" share no whole-word token, so a
        # word-level vectorizer would systematically fail to cluster the same
        # pain phrased with different case endings across interviews.
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        vectors = vectorizer.fit_transform(texts)

    cluster_rep_idx: list[int] = []
    cluster_members: list[list[str]] = []

    for idx, insight in enumerate(accepted):
        best_idx, best_score = -1, 0.0
        if vectors is not None and cluster_rep_idx:
            sims = cosine_similarity(vectors[idx], vectors[cluster_rep_idx])[0]
            for c, score in enumerate(sims):
                if score > best_score:
                    best_score, best_idx = score, c

        if best_idx >= 0 and best_score >= SIMILARITY_THRESHOLD:
            cluster_members[best_idx].append(insight.id)
        else:
            cluster_rep_idx.append(idx)
            cluster_members.append([insight.id])

    by_id = {i.id: i for i in accepted}
    clusters = []
    for member_ids in cluster_members:
        rep_insight = by_id[member_ids[0]]
        strongest = max(
            (by_id[mid].signal_strength for mid in member_ids), key=lambda s: s.rank
        )
        clusters.append(
            Cluster(
                id=_cluster_id(rep_insight.pain),
                representative_pain=rep_insight.pain,
                signal_strength=strongest,
                member_ids=member_ids,
            )
        )

    clusters.sort(key=lambda c: (c.signal_strength.rank, c.frequency), reverse=True)
    return clusters
