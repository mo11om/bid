"""LLMCache key-sensitivity tests, notably for the think on/off setting."""

from src.harness.llm_cache import LLMCache


def test_cache_roundtrip(tmp_path):
    cache = LLMCache(str(tmp_path), "test-model")
    assert cache.get("prompt-a", 0.0) is None
    cache.set("prompt-a", 0.0, "response-a")
    assert cache.get("prompt-a", 0.0) == "response-a"


def test_cache_distinguishes_think_setting(tmp_path):
    """Same prompt+temperature but different `think` must be separate entries."""
    cache = LLMCache(str(tmp_path), "test-model")
    cache.set("prompt-a", 0.0, "response-no-think", think=False)

    assert cache.get("prompt-a", 0.0, think=False) == "response-no-think"
    assert cache.get("prompt-a", 0.0, think=True) is None  # different setting
    assert cache.get("prompt-a", 0.0, think=None) is None  # 'auto' is distinct too

    cache.set("prompt-a", 0.0, "response-thinking", think=True)
    assert cache.get("prompt-a", 0.0, think=True) == "response-thinking"
    assert cache.get("prompt-a", 0.0, think=False) == "response-no-think"  # unaffected


def test_cache_persists_across_instances(tmp_path):
    cache1 = LLMCache(str(tmp_path), "test-model")
    cache1.set("prompt-a", 0.5, "saved", think=True)

    cache2 = LLMCache(str(tmp_path), "test-model")
    assert cache2.get("prompt-a", 0.5, think=True) == "saved"
