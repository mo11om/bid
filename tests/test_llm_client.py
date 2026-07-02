"""LocalLLMClient tests — dual-backend (ollama / vllm).

vLLM path: monkeypatched OpenAI client (no network).
Ollama path: unittest.mock.patch on requests.post / requests.get (no network).
"""

import json
import types
import unittest.mock as mock

import pytest

from src.config import Config
from src.harness.llm_client import LocalLLMClient


# ------------------------------------------------------------------ #
# vLLM fake helpers (OpenAI SDK mock)
# ------------------------------------------------------------------ #

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.content)


def _vllm_client(content: str, think, cache_dir: str) -> tuple[LocalLLMClient, _FakeCompletions]:
    config = Config(backend="vllm", base_url="http://localhost:8000/v1",
                    cache_dir=cache_dir, think=think)
    client = LocalLLMClient(config)
    client.cache.get = lambda *a, **k: None
    client.cache.set = lambda *a, **k: None
    fake_completions = _FakeCompletions(content)
    fake_chat = types.SimpleNamespace(completions=fake_completions)
    client._client = types.SimpleNamespace(chat=fake_chat)
    return client, fake_completions


# ------------------------------------------------------------------ #
# Ollama fake helper (requests mock)
# ------------------------------------------------------------------ #

def _ollama_client(content: str, think, cache_dir: str,
                   server_thinking: str = "") -> LocalLLMClient:
    config = Config(backend="ollama", cache_dir=cache_dir, think=think)
    client = LocalLLMClient(config)
    client.cache.get = lambda *a, **k: None
    client.cache.set = lambda *a, **k: None
    return client


def _fake_ollama_post(content: str, server_thinking: str = ""):
    """Return a mock for requests.post that yields a canned /api/chat response."""
    msg = {"role": "assistant", "content": content}
    if server_thinking:
        msg["thinking"] = server_thinking
    fake_resp = mock.MagicMock()
    fake_resp.raise_for_status = mock.MagicMock()
    fake_resp.json.return_value = {"message": msg}
    return mock.patch("src.harness.llm_client._requests.post", return_value=fake_resp)


# ------------------------------------------------------------------ #
# vLLM: think field in extra_body
# ------------------------------------------------------------------ #

def test_vllm_think_none_omits_extra_body(tmp_path):
    client, backend = _vllm_client(
        json.dumps({"thinking": "ok", "bid": "Pass"}), think=None, cache_dir=str(tmp_path)
    )
    client.get_bid("some prompt")
    assert "extra_body" not in backend.calls[0]


def test_vllm_think_true_sets_extra_body(tmp_path):
    client, backend = _vllm_client(
        json.dumps({"thinking": "ok", "bid": "Pass"}), think=True, cache_dir=str(tmp_path)
    )
    client.get_bid("some prompt")
    assert backend.calls[0]["extra_body"] == {"think": True}


def test_vllm_think_false_sets_extra_body(tmp_path):
    client, backend = _vllm_client(
        json.dumps({"thinking": "ok", "bid": "Pass"}), think=False, cache_dir=str(tmp_path)
    )
    client.get_bid("some prompt")
    assert backend.calls[0]["extra_body"] == {"think": False}


# ------------------------------------------------------------------ #
# Ollama: think field in request payload
# ------------------------------------------------------------------ #

def test_ollama_think_none_omits_think_field(tmp_path):
    client = _ollama_client("", think=None, cache_dir=str(tmp_path))
    with _fake_ollama_post(json.dumps({"thinking": "ok", "bid": "Pass"})) as m:
        client.get_bid("some prompt")
    payload = m.call_args[1]["json"]
    assert "think" not in payload


def test_ollama_think_true_sends_think_field(tmp_path):
    client = _ollama_client("", think=True, cache_dir=str(tmp_path))
    with _fake_ollama_post(json.dumps({"thinking": "ok", "bid": "Pass"})) as m:
        client.get_bid("some prompt")
    payload = m.call_args[1]["json"]
    assert payload["think"] is True


def test_ollama_think_false_sends_think_field(tmp_path):
    client = _ollama_client("", think=False, cache_dir=str(tmp_path))
    with _fake_ollama_post(json.dumps({"thinking": "ok", "bid": "Pass"})) as m:
        client.get_bid("some prompt")
    payload = m.call_args[1]["json"]
    assert payload["think"] is False


def test_ollama_server_thinking_surfaced_when_model_json_empty(tmp_path):
    """Server-side reasoning is merged into BridgeBid.thinking when the model's
    own JSON has an empty thinking field."""
    client = _ollama_client("", think=True, cache_dir=str(tmp_path))
    model_json = json.dumps({"thinking": "", "bid": "1NT"})
    with _fake_ollama_post(model_json, server_thinking="deep chain of thought"):
        bid = client.get_bid("some prompt")
    assert bid.bid == "1NT"
    assert "deep chain of thought" in bid.thinking


# ------------------------------------------------------------------ #
# Parsing helpers (backend-agnostic — test _parse_bid directly)
# ------------------------------------------------------------------ #

def test_parse_bid_strips_inline_think_tags(tmp_path):
    client, _ = _vllm_client("ignored", think=None, cache_dir=str(tmp_path))
    raw = (
        "<think>Let me consider the auction... 13 HCP, balanced.</think>"
        '{"thinking": "13 HCP balanced", "bid": "1NT"}'
    )
    bid = client._parse_bid(raw)
    assert bid.bid == "1NT"


def test_parse_bid_without_think_tags_unaffected(tmp_path):
    client, _ = _vllm_client("ignored", think=None, cache_dir=str(tmp_path))
    raw = '{"thinking": "pass with nothing", "bid": "Pass"}'
    bid = client._parse_bid(raw)
    assert bid.bid == "Pass"


def test_parse_bid_ignores_trailing_data(tmp_path):
    """A valid object followed by prose or a second object still parses (the
    'Extra data' failure mode seen live)."""
    client, _ = _vllm_client("ignored", think=None, cache_dir=str(tmp_path))
    raw = (
        '{"thinking": "16 HCP balanced", "bid": "1NT"}\n'
        "Note: this is my final answer.\n"
        '{"thinking": "second object", "bid": "Pass"}'
    )
    bid = client._parse_bid(raw)
    assert bid.bid == "1NT"


def test_parse_bid_leading_prose_then_object(tmp_path):
    client, _ = _vllm_client("ignored", think=None, cache_dir=str(tmp_path))
    raw = 'Here is my call: {"thinking": "weak", "bid": "Pass"}'
    bid = client._parse_bid(raw)
    assert bid.bid == "Pass"


def test_parse_bid_no_object_raises(tmp_path):
    import pytest

    client, _ = _vllm_client("ignored", think=None, cache_dir=str(tmp_path))
    with pytest.raises(ValueError):
        client._parse_bid("no json here at all")


# ------------------------------------------------------------------ #
# Fallback-reason differentiation
# ------------------------------------------------------------------ #

def test_transport_failure_tagged_distinctly(tmp_path):
    client = _ollama_client("", think=None, cache_dir=str(tmp_path))
    with mock.patch("src.harness.llm_client._requests.post",
                    side_effect=ConnectionError("refused")):
        bid = client.get_bid("prompt")
    assert bid.bid == "Pass"
    assert bid.thinking.startswith("transport error:")


def test_parse_failure_tagged_distinctly(tmp_path):
    client = _ollama_client("", think=None, cache_dir=str(tmp_path))
    with _fake_ollama_post("not json at all"):
        bid = client.get_bid("prompt")
    assert bid.bid == "Pass"
    assert bid.thinking.startswith("parse error:")


def test_illegal_call_tagged_distinctly(tmp_path):
    client = _ollama_client("", think=None, cache_dir=str(tmp_path))
    with _fake_ollama_post(json.dumps({"thinking": "x", "bid": "1C"})):
        # 1NT already on the table; 1C does not outrank it -> illegal.
        bid = client.get_bid("prompt", history=["1NT"])
    assert bid.bid == "Pass"
    assert bid.thinking.startswith("illegal call")


# ------------------------------------------------------------------ #
# verify_connection — vLLM path
# ------------------------------------------------------------------ #

class _FakeModel:
    def __init__(self, id_):
        self.id = id_


class _FakeModelsList:
    def __init__(self, ids):
        self.data = [_FakeModel(i) for i in ids]


def _vllm_client_with_fake_models(ids, model: str, cache_dir: str) -> LocalLLMClient:
    config = Config(backend="vllm", base_url="http://localhost:8000/v1",
                    model=model, cache_dir=cache_dir)
    client = LocalLLMClient(config)
    fake_models = types.SimpleNamespace(list=lambda: _FakeModelsList(ids))
    client._client = types.SimpleNamespace(models=fake_models)
    return client


def test_vllm_verify_connection_passes_when_model_present(tmp_path):
    client = _vllm_client_with_fake_models(
        ["llama3:8b", "qwen3.6:27b"], "llama3:8b", str(tmp_path)
    )
    client.verify_connection()  # must not raise


def test_vllm_verify_connection_raises_on_missing_model(tmp_path):
    client = _vllm_client_with_fake_models(["llama3:8b"], "llama3", str(tmp_path))
    with pytest.raises(RuntimeError, match="not found"):
        client.verify_connection()


def test_vllm_verify_connection_raises_on_dead_endpoint(tmp_path):
    config = Config(backend="vllm", base_url="http://localhost:8000/v1", cache_dir=str(tmp_path))
    client = LocalLLMClient(config)
    broken_models = types.SimpleNamespace(
        list=lambda: (_ for _ in ()).throw(ConnectionError("refused"))
    )
    client._client = types.SimpleNamespace(models=broken_models)
    with pytest.raises(RuntimeError, match="Could not reach"):
        client.verify_connection()


# ------------------------------------------------------------------ #
# verify_connection — Ollama path
# ------------------------------------------------------------------ #

def _fake_tags_response(model_names):
    fake_resp = mock.MagicMock()
    fake_resp.raise_for_status = mock.MagicMock()
    fake_resp.json.return_value = {"models": [{"name": n} for n in model_names]}
    return fake_resp


def test_ollama_verify_connection_passes_when_model_present(tmp_path):
    config = Config(backend="ollama", model="qwen3.6:27b", cache_dir=str(tmp_path))
    client = LocalLLMClient(config)
    fake_resp = _fake_tags_response(["qwen3.6:27b", "llama3:8b"])
    with mock.patch("src.harness.llm_client._requests.get", return_value=fake_resp):
        client.verify_connection()  # must not raise


def test_ollama_verify_connection_raises_on_missing_model(tmp_path):
    config = Config(backend="ollama", model="llama3", cache_dir=str(tmp_path))
    client = LocalLLMClient(config)
    fake_resp = _fake_tags_response(["llama3:8b"])
    with mock.patch("src.harness.llm_client._requests.get", return_value=fake_resp):
        with pytest.raises(RuntimeError, match="not found"):
            client.verify_connection()


def test_ollama_verify_connection_raises_on_dead_endpoint(tmp_path):
    config = Config(backend="ollama", cache_dir=str(tmp_path))
    client = LocalLLMClient(config)
    with mock.patch("src.harness.llm_client._requests.get",
                    side_effect=ConnectionError("refused")):
        with pytest.raises(RuntimeError, match="Could not reach"):
            client.verify_connection()
