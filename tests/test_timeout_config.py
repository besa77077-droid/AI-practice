"""Regression test for the Ollama timeout that was too short for real
interviews: a 300s default was fine for short test transcripts but real
20+ minute calls on CPU inference regularly exceeded it (found on an actual
Ollama run, not a mock). The timeout must be generous by default and
still overridable via OLLAMA_TIMEOUT_S without touching code."""

from insight_engine.providers.llm import OllamaProvider, build_llm_provider


def test_ollama_provider_default_timeout_is_generous():
    provider = OllamaProvider()
    assert provider.timeout_s >= 1200


def test_build_llm_provider_forwards_custom_timeout():
    provider = build_llm_provider("ollama", timeout_s=42.0)
    assert isinstance(provider, OllamaProvider)
    assert provider.timeout_s == 42.0


def test_build_llm_provider_without_timeout_uses_default():
    provider = build_llm_provider("ollama")
    assert isinstance(provider, OllamaProvider)
    assert provider.timeout_s >= 1200
