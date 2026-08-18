from insight_engine.clustering import cluster_insights
from insight_engine.hypotheses import generate_hypotheses
from insight_engine.models import Insight, SignalStrength
from insight_engine.providers.llm import MockLLMProvider


def _insight(id_, interview_id, pain, jtbd, strength=SignalStrength.MEDIUM):
    return Insight(
        id=id_, interview_id=interview_id, pain=pain, context="", workaround="",
        jtbd=jtbd, quote=f"цитата {id_}", signal_strength=strength,
    )


def test_similar_pains_across_interviews_cluster_together():
    insights = [
        _insight("i1", "iv1", "Нет произвольного диапазона дат в выписке",
                 "Когда сдаю отчётность, я хочу выбрать даты, чтобы не резать вручную",
                 SignalStrength.HIGH),
        _insight("i2", "iv2", "Нельзя задать произвольный период выписки",
                 "Когда готовлю отчёт, я хочу произвольный диапазон, чтобы сдать вовремя",
                 SignalStrength.HIGH),
        _insight("i3", "iv3", "Массовые зарплатные переводы отсутствуют",
                 "Когда плачу зарплату, я хочу один клик на всех, чтобы не тратить час",
                 SignalStrength.LOW),
    ]
    clusters = cluster_insights(insights)
    assert len(clusters) == 2
    biggest = max(clusters, key=lambda c: c.frequency)
    assert biggest.frequency == 2
    assert set(biggest.member_ids) == {"i1", "i2"}
    assert biggest.signal_strength == SignalStrength.HIGH


def test_rejected_insights_are_excluded_from_clustering():
    insights = [
        _insight("i1", "iv1", "Боль А", "jtbd a"),
        _insight("i2", "iv1", "Боль Б", "jtbd b"),
    ]
    insights[1].status = "rejected"
    clusters = cluster_insights(insights)
    assert len(clusters) == 1
    assert clusters[0].member_ids == ["i1"]


def test_hypotheses_generated_with_deterministic_priority():
    insights = [
        _insight("i1", "iv1", "Боль", "jtbd", SignalStrength.HIGH),
        _insight("i2", "iv2", "Боль", "jtbd", SignalStrength.HIGH),
        _insight("i3", "iv3", "Боль", "jtbd", SignalStrength.HIGH),
    ]
    clusters = cluster_insights(insights)
    by_id = {i.id: i for i in insights}
    hyps = generate_hypotheses(clusters, by_id, MockLLMProvider())
    assert len(hyps) == 1
    assert hyps[0].priority == SignalStrength.HIGH
    assert set(hyps[0].quote_refs) == {"i1", "i2", "i3"}
