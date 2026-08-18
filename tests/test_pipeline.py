from pathlib import Path

from insight_engine.pipeline import ingest_text, process_interview
from insight_engine.providers.llm import MockLLMProvider
from insight_engine.storage.sqlite_store import InsightStore

FIXTURE = Path(__file__).parent / "fixtures" / "interview_farida.txt"


def test_end_to_end_pipeline_with_mock_llm(tmp_path):
    store = InsightStore(str(tmp_path / "insights.db"))
    llm = MockLLMProvider()

    segments = ingest_text(FIXTURE.read_text(encoding="utf-8"))
    result = process_interview(
        segments=segments,
        title="Интервью 08: Фарида Р.",
        respondent="Фарида Р.",
        segment_label="Малый бизнес",
        llm=llm,
        store=store,
    )

    assert result.interview.id
    assert len(result.insights) >= 2
    for insight in result.insights:
        assert insight.segment_id is not None, "quote should be traceable to a segment"
        assert insight.match_confidence > 0.5

    assert len(result.clusters) >= 1
    assert len(result.hypotheses) >= 1

    # storage round-trip
    stored_interview = store.get_interview(result.interview.id)
    assert stored_interview is not None
    assert len(stored_interview.segments) == len(segments)

    stored_insights = store.list_insights_for_interview(result.interview.id)
    assert len(stored_insights) == len(result.insights)

    hyps = store.list_hypotheses_ranked()
    assert len(hyps) == len(result.hypotheses)
    priorities = [h.priority.rank for h in hyps]
    assert priorities == sorted(priorities, reverse=True)
