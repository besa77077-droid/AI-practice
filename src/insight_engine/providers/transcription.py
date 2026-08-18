"""Audio transcription + speaker diarization, and text-transcript parsing.

Real transcription/diarization are optional, heavy dependencies
(faster-whisper, pyannote.audio) and are imported lazily so the rest of the
package works without them installed — e.g. when a user only ever uploads
text transcripts.

Design note (why this file exists in this shape): the previous prototype's
audio path silently hung or failed with no actionable message — usually
because `ffmpeg` was missing, or because the first run needed to download a
~500 MB Whisper model over a slow/blocked network and the HTTP request just
timed out. Every failure mode here is checked explicitly and raises a
`TranscriptionError` with a message a non-engineer can act on.
"""

from __future__ import annotations

import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from insight_engine.models import SpeakerSegment

DEFAULT_SPEAKER = "Респондент"
INTERVIEWER_SPEAKER = "Интервьюер"

_TIMECODE = r"(?:\[(?P<mm>\d{1,2}):(?P<ss>\d{2})\]\s*)?"
_SPEAKER_LINE_RE = re.compile(rf"^\s*{_TIMECODE}(?P<speaker>[^:\n]{{1,40}}):\s*(?P<text>.+)$")


class TranscriptionError(RuntimeError):
    """Raised with a message that is safe to show directly to the user."""


def check_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise TranscriptionError(
            "ffmpeg не найден в системе. Обработка аудио невозможна без него — "
            "установите ffmpeg (в Docker-образе он есть по умолчанию) или "
            "загрузите готовый текстовый транскрипт вместо аудио."
        )


class DiarizationProvider(ABC):
    """Splits an audio file into (start, end, speaker) turns."""

    @abstractmethod
    def diarize(self, audio_path: str) -> list[tuple[float, float, str]]:
        raise NotImplementedError


class PyannoteDiarizationProvider(DiarizationProvider):
    """Diarization via pyannote.audio (needs a free HuggingFace token + the
    speaker-diarization model license accepted on huggingface.co)."""

    def __init__(self, hf_token: str) -> None:
        if not hf_token:
            raise TranscriptionError(
                "Для диаризации нужен HuggingFace-токен (HUGGINGFACE_TOKEN). "
                "Без него аудио всё равно обработается, но все реплики будут "
                "помечены одним спикером."
            )
        self.hf_token = hf_token

    def diarize(self, audio_path: str) -> list[tuple[float, float, str]]:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as exc:
            raise TranscriptionError(
                "pyannote.audio не установлен. Установите пакет с extras "
                "'diarization' или отключите диаризацию."
            ) from exc

        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token
            )
            diarization = pipeline(audio_path)
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim to the UI
            raise TranscriptionError(
                "Диаризация не удалась (неверный токен, не принята лицензия "
                f"модели на HuggingFace, либо нет сети): {exc}"
            ) from exc

        return [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]


class TranscriptionProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> list[SpeakerSegment]:
        raise NotImplementedError


def _assign_speaker(
    start: float, end: float, turns: list[tuple[float, float, str]]
) -> str:
    """Pick the diarization turn with the largest time overlap for a segment."""
    best_speaker, best_overlap = DEFAULT_SPEAKER, 0.0
    for turn_start, turn_end, speaker in turns:
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, speaker
    return best_speaker


@dataclass
class WhisperProgress:
    words_recognized: int
    duration_s: float


class FasterWhisperProvider(TranscriptionProvider):
    """Local transcription via faster-whisper (CPU, int8 by default).

    Diarization is optional: without a DiarizationProvider every segment is
    labelled with a single fallback speaker.
    """

    def __init__(
        self,
        model_size: str = "small",
        diarizer: DiarizationProvider | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        on_progress: Callable[[WhisperProgress], None] | None = None,
    ) -> None:
        self.model_size = model_size
        self.diarizer = diarizer
        self.device = device
        self.compute_type = compute_type
        self.on_progress = on_progress

    def transcribe(self, audio_path: str) -> list[SpeakerSegment]:
        check_ffmpeg_available()

        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise TranscriptionError(
                "faster-whisper не установлен. Установите пакет с extras "
                "'audio' (pip install -e '.[audio]') или используйте текстовый "
                "транскрипт вместо аудио."
            ) from exc

        try:
            model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            raw_segments, info = model.transcribe(audio_path)
            raw_segments = list(raw_segments)
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(
                "Не удалось запустить распознавание речи. Часто это значит, что "
                "модель Whisper не смогла скачаться (нет сети при первом запуске) "
                f"или файл повреждён/в неподдерживаемом формате. Детали: {exc}"
            ) from exc

        if not raw_segments:
            raise TranscriptionError(
                "Распознавание вернуло пустой результат — проверьте, что в "
                "аудиофайле действительно есть речь и звук не слишком тихий."
            )

        turns: list[tuple[float, float, str]] = []
        if self.diarizer is not None:
            turns = self.diarizer.diarize(audio_path)

        segments: list[SpeakerSegment] = []
        total_words = 0
        for idx, seg in enumerate(raw_segments):
            speaker = _assign_speaker(seg.start, seg.end, turns) if turns else DEFAULT_SPEAKER
            text = seg.text.strip()
            total_words += len(text.split())
            segments.append(
                SpeakerSegment(id=idx, speaker=speaker, text=text, start=seg.start, end=seg.end)
            )
            if self.on_progress:
                self.on_progress(WhisperProgress(words_recognized=total_words, duration_s=seg.end))

        return segments


def segments_to_transcript(segments: list[SpeakerSegment]) -> str:
    """Render speaker segments as a plain-text transcript for LLM input."""
    lines = []
    for seg in segments:
        prefix = f"[{_fmt_timecode(seg.start)}] " if seg.start is not None else ""
        lines.append(f"{prefix}{seg.speaker}: {seg.text}")
    return "\n".join(lines)


def _fmt_timecode(seconds: float | None) -> str:
    if seconds is None:
        return "00:00"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def parse_speaker_transcript(transcript: str) -> list[SpeakerSegment]:
    """Parse a "[MM:SS] Speaker: text" (timecode optional) transcript into
    segments. Lines that don't match are attributed to DEFAULT_SPEAKER, so the
    parser degrades gracefully on plain, unlabelled transcripts."""
    segments: list[SpeakerSegment] = []
    for line in transcript.splitlines():
        if not line.strip():
            continue
        match = _SPEAKER_LINE_RE.match(line)
        idx = len(segments)
        if match:
            speaker = match.group("speaker").strip()
            text = match.group("text").strip()
            start = None
            if match.group("mm") is not None:
                start = int(match.group("mm")) * 60 + int(match.group("ss"))
        else:
            speaker, text, start = DEFAULT_SPEAKER, line.strip(), None
        segments.append(SpeakerSegment(id=idx, speaker=speaker, text=text, start=start))
    if not segments:
        raise TranscriptionError("Транскрипт пуст — нечего обрабатывать.")
    return segments


def transcribe_audio(
    audio_path: str,
    whisper_model: str = "small",
    diarize: bool = False,
    hf_token: str | None = None,
    on_progress: Callable[[WhisperProgress], None] | None = None,
) -> list[SpeakerSegment]:
    """Transcribe an audio file into speaker-labelled segments.

    Shared entry point so the CLI, web app, and background job all go through
    the exact same provider wiring.
    """
    diarizer: DiarizationProvider | None = None
    if diarize:
        diarizer = PyannoteDiarizationProvider(hf_token=hf_token or "")

    provider = FasterWhisperProvider(
        model_size=whisper_model, diarizer=diarizer, on_progress=on_progress
    )
    return provider.transcribe(audio_path)
