"""Regression tests for parsing real Ollama output shapes.

Found by actually running llama3.1 through Ollama (not the mock provider):
its JSON-mode output is valid JSON but not reliably a bare top-level array —
it wraps insights in an object, or returns `{}` when it finds nothing. The
parser must handle those shapes instead of erroring out.
"""

import pytest

from insight_engine.providers.llm import LLMError, _parse_hypothesis, _parse_insights


def test_parses_bare_array_as_before():
    raw = '[{"pain": "p", "context": "c", "workaround": "w", "jtbd": "j", "quote": "q", "signal_strength": "high"}]'
    insights = _parse_insights(raw)
    assert len(insights) == 1
    assert insights[0].pain == "p"


def test_empty_object_means_no_insights_found():
    assert _parse_insights("{}") == []


def test_wrapper_object_with_known_key_is_unwrapped():
    raw = '{"insights": [{"pain": "p", "context": "c", "workaround": "w", "jtbd": "j", "quote": "q", "signal_strength": "low"}]}'
    insights = _parse_insights(raw)
    assert len(insights) == 1


def test_wrapper_object_with_unknown_single_key_is_unwrapped():
    raw = '{"result": [{"pain": "p", "context": "c", "workaround": "w", "jtbd": "j", "quote": "q", "signal_strength": "low"}]}'
    insights = _parse_insights(raw)
    assert len(insights) == 1


def test_single_insight_object_not_wrapped_in_list_is_still_accepted():
    raw = '{"pain": "p", "context": "c", "workaround": "w", "jtbd": "j", "quote": "q", "signal_strength": "medium"}'
    insights = _parse_insights(raw)
    assert len(insights) == 1


def test_truly_unusable_shape_raises_with_snippet_for_debugging():
    with pytest.raises(LLMError, match="unexpected shape marker"):
        _parse_insights('{"unexpected shape marker": 42}')


def test_markdown_fenced_json_still_parses():
    raw = '```json\n[{"pain": "p", "context": "c", "workaround": "w", "jtbd": "j", "quote": "q", "signal_strength": "high"}]\n```'
    insights = _parse_insights(raw)
    assert len(insights) == 1


def test_hypothesis_wrapper_object_is_unwrapped():
    raw = '{"hypothesis": {"solution": "s", "metric": "m", "metric_direction": "d", "insight_summary": "i"}}'
    hyp = _parse_hypothesis(raw, pain_title="fallback")
    assert hyp.solution == "s"


def test_hypothesis_bare_object_as_before():
    raw = '{"solution": "s", "metric": "m", "metric_direction": "d", "insight_summary": "i"}'
    hyp = _parse_hypothesis(raw, pain_title="fallback")
    assert hyp.solution == "s"
