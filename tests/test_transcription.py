import pytest

from insight_engine.providers.transcription import (
    TranscriptionError,
    parse_speaker_transcript,
    segments_to_transcript,
)


def test_parse_speaker_transcript_with_timecodes():
    text = (
        "[00:05] Интервьюер: Вопрос?\n"
        "[00:12] Фарида Р.: Ответ подробный.\n"
    )
    segments = parse_speaker_transcript(text)
    assert len(segments) == 2
    assert segments[0].speaker == "Интервьюер"
    assert segments[1].start == 12
    assert segments[1].text == "Ответ подробный."


def test_parse_speaker_transcript_without_labels_degrades_gracefully():
    segments = parse_speaker_transcript("просто текст без разметки спикеров")
    assert len(segments) == 1
    assert segments[0].speaker == "Респондент"


def test_parse_empty_transcript_raises_actionable_error():
    with pytest.raises(TranscriptionError):
        parse_speaker_transcript("   \n  ")


def test_segments_to_transcript_roundtrip():
    segments = parse_speaker_transcript("[01:02] Спикер: Текст реплики")
    rendered = segments_to_transcript(segments)
    assert "01:02" in rendered and "Спикер" in rendered and "Текст реплики" in rendered
