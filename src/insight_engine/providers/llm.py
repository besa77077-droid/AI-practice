"""LLM providers for structured insight extraction and hypothesis drafting.

Kept behind a small interface (`LLMProvider`) so the rest of the pipeline
never depends on Ollama specifically — tests run entirely against
`MockLLMProvider`, no network or local model required.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

from insight_engine.models import SignalStrength, SpeakerSegment
from insight_engine.providers.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    HYPOTHESIS_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_hypothesis_prompt,
)


class LLMError(RuntimeError):
    """Raised with a message that is safe to show directly to the user."""


@dataclass
class RawInsight:
    pain: str
    context: str
    workaround: str
    jtbd: str
    quote: str
    signal_strength: SignalStrength
    speaker: str | None = None


@dataclass
class RawHypothesis:
    solution: str
    metric: str
    metric_direction: str
    insight_summary: str


_SIGNAL_MAP = {"high": SignalStrength.HIGH, "medium": SignalStrength.MEDIUM, "low": SignalStrength.LOW}


def _coerce_signal(value: str | None) -> SignalStrength:
    return _SIGNAL_MAP.get((value or "").strip().lower(), SignalStrength.MEDIUM)


def _extract_json_payload(raw: str) -> str:
    """Strip markdown code fences some models add despite instructions."""
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    return match.group(1).strip() if match else raw.strip()


class LLMProvider(ABC):
    @abstractmethod
    def extract_insights(self, segments: list[SpeakerSegment]) -> list[RawInsight]:
        raise NotImplementedError

    @abstractmethod
    def draft_hypothesis(self, pain_title: str, quotes: list[str]) -> RawHypothesis:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _generate(self, system: str, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": system,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"num_ctx": 8192},
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(
                f"Не удалось подключиться к Ollama по адресу {self.base_url}. "
                "Убедитесь, что `ollama serve` запущена и модель скачана "
                f"(`ollama pull {self.model}`)."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError(
                f"Ollama не ответила за {self.timeout_s:.0f} сек. Для длинных "
                "интервью увеличьте таймаут или используйте более лёгкую модель."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Ошибка запроса к Ollama: {exc}") from exc

        try:
            return resp.json()["response"]
        except (KeyError, ValueError) as exc:
            raise LLMError(f"Неожиданный ответ от Ollama: {resp.text[:500]}") from exc

    def extract_insights(self, segments: list[SpeakerSegment]) -> list[RawInsight]:
        raw = self._generate(EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt(segments))
        return _parse_insights(raw)

    def draft_hypothesis(self, pain_title: str, quotes: list[str]) -> RawHypothesis:
        raw = self._generate(HYPOTHESIS_SYSTEM_PROMPT, build_hypothesis_prompt(pain_title, quotes))
        return _parse_hypothesis(raw, pain_title)


def _parse_insights(raw: str) -> list[RawInsight]:
    payload = _extract_json_payload(raw)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"LLM вернула не-JSON ответ, не удалось разобрать инсайты: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise LLMError("LLM вернула не список инсайтов — ожидался JSON-массив.")

    insights = []
    for item in data:
        if not isinstance(item, dict) or not item.get("quote"):
            continue
        insights.append(
            RawInsight(
                pain=str(item.get("pain", "")).strip(),
                context=str(item.get("context", "")).strip(),
                workaround=str(item.get("workaround", "Не упомянуто")).strip(),
                jtbd=str(item.get("jtbd", "")).strip(),
                quote=str(item.get("quote", "")).strip(),
                signal_strength=_coerce_signal(item.get("signal_strength")),
                speaker=item.get("speaker"),
            )
        )
    return insights


def _parse_hypothesis(raw: str, pain_title: str) -> RawHypothesis:
    payload = _extract_json_payload(raw)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"LLM вернула не-JSON ответ при формировании гипотезы: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise LLMError("LLM вернула не объект гипотезы — ожидался JSON-объект.")
    return RawHypothesis(
        solution=str(data.get("solution", "")).strip() or "Уточнить решение",
        metric=str(data.get("metric", "")).strip() or "ключевая метрика",
        metric_direction=str(data.get("metric_direction", "")).strip() or "изменится",
        insight_summary=str(data.get("insight_summary", "")).strip() or pain_title,
    )


class MockLLMProvider(LLMProvider):
    """Deterministic provider for tests and offline demos — no network.

    Heuristically treats any respondent line longer than a few words as a
    pain signal, cycling signal strength so multi-insight scenarios are
    exercised without needing a real model.
    """

    _STRENGTHS = [SignalStrength.HIGH, SignalStrength.MEDIUM, SignalStrength.LOW]

    def extract_insights(self, segments: list[SpeakerSegment]) -> list[RawInsight]:
        insights = []
        i = 0
        for seg in segments:
            if seg.speaker.lower().startswith("интервьюер") or seg.speaker.lower().startswith("interview"):
                continue
            if len(seg.text.split()) < 4:
                continue
            insights.append(
                RawInsight(
                    pain=f"Боль: {seg.text[:40].rstrip('. ')}",
                    context="Контекст не уточнён в моке",
                    workaround="Не упомянуто",
                    jtbd=f"Когда возникает ситуация, я хочу решить проблему, чтобы {seg.text[:30]}",
                    quote=seg.text,
                    signal_strength=self._STRENGTHS[i % len(self._STRENGTHS)],
                    speaker=seg.speaker,
                )
            )
            i += 1
        return insights

    def draft_hypothesis(self, pain_title: str, quotes: list[str]) -> RawHypothesis:
        return RawHypothesis(
            solution=f"доработать продукт под «{pain_title}»",
            metric="удовлетворённость клиентов",
            metric_direction="вырастет",
            insight_summary=pain_title,
        )


def build_llm_provider(name: str, *, model: str | None = None, base_url: str = "http://localhost:11434") -> LLMProvider:
    if name == "ollama":
        return OllamaProvider(model=model or "llama3.1", base_url=base_url)
    if name == "mock":
        return MockLLMProvider()
    raise ValueError(f"Неизвестный LLM-провайдер: {name}")
