from insight_engine.matching import match_quote
from insight_engine.models import SpeakerSegment


def _segments():
    return [
        SpeakerSegment(id=0, speaker="Интервьюер", text="Расскажите про выписки.", start=5, end=8),
        SpeakerSegment(
            id=1, speaker="Фарида Р.",
            text="Приложение отдаёт выписку только за месяц, а мне нужен точный диапазон дат.",
            start=12, end=20,
        ),
        SpeakerSegment(id=2, speaker="Интервьюер", text="Понятно.", start=21, end=22),
    ]


def test_exact_verbatim_quote_matches_segment():
    match = match_quote(
        "Приложение отдаёт выписку только за месяц, а мне нужен точный диапазон дат.",
        _segments(),
    )
    assert match.segment_id == 1
    assert match.confidence == 1.0
    assert match.start == 12 and match.end == 20


def test_slightly_paraphrased_quote_still_matches_with_lower_confidence():
    match = match_quote(
        "приложение выписку только за месяц нужен точный диапазон дат",
        _segments(),
    )
    assert match.segment_id == 1
    assert 0.35 < match.confidence <= 1.0


def test_unrelated_quote_does_not_match():
    match = match_quote("что-то совершенно не относящееся к транскрипту вообще никак", _segments())
    assert match.segment_id is None


def test_empty_quote_or_segments_does_not_crash():
    assert match_quote("", _segments()).segment_id is None
    assert match_quote("текст", []).segment_id is None
