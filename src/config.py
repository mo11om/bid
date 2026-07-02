"""Central configuration for the Bridge AI Evaluation System.

All tunables live here so switching between a vLLM endpoint (:8000) and an
Ollama endpoint (:11434), or changing the acceptance threshold, is a one-line
change. Values can be overridden at the CLI (see run_eval.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Common endpoints.
VLLM_BASE_URL = "http://localhost:8000/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"       # OpenAI-compat (kept for reference)
OLLAMA_NATIVE_BASE_URL = "http://localhost:11434"   # native /api/* endpoint


@dataclass
class Config:
    """Runtime configuration.

    Attributes
    ----------
    base_url:
        OpenAI-compatible endpoint for the local model server.
    api_key:
        Placeholder key; local servers ignore it but the openai client requires
        a non-empty string.
    model:
        Model name as registered with the local server. For Ollama this MUST
        include the tag (e.g. ``"llama3:8b"``, not ``"llama3"``) — Ollama does
        not alias an untagged name to an installed tag, so a mismatch here
        causes every single request to 404 and silently fall back to ``Pass``.
        Run ``ollama list`` to see exact installed tags.
    temperature:
        Sampling temperature for bid generation.
    think:
        Controls "thinking"/reasoning mode on servers that support it (e.g.
        Ollama with qwen3, deepseek-r1, gpt-oss). Tri-state:

        * ``None`` (default) — omit the parameter entirely; the server/model
          decides. Safest choice for vLLM or any endpoint that doesn't know
          about ``think`` and might reject an unrecognized field.
        * ``True`` — force thinking on.
        * ``False`` — force thinking off (faster, cheaper; recommended for
          bidding since the structured ``thinking`` field in ``BridgeBid``
          already captures the model's rationale).
    request_timeout:
        Per-request timeout in seconds.
    threshold_mode:
        ``"imp"`` (default) compares contracts by IMP delta via the WBF table;
        ``"score"`` compares raw duplicate score points.
    threshold_n:
        A differing bid is "acceptable" when the delta is ``<= threshold_n``
        (IMPs in ``imp`` mode, score points in ``score`` mode).
    max_rollout_calls:
        Safety cap on LLM calls during a single auction rollout.
    cache_dir:
        Directory for the on-disk LLM response cache.
    data_dir:
        Directory for datasets and report outputs.
    """

    backend: str = "ollama"  # "ollama" | "vllm"
    base_url: str = OLLAMA_NATIVE_BASE_URL
    api_key: str = "local-no-key"
    model: str = "gemma4:26b"
    temperature: float = 0.0
    think: Optional[bool] = None  # None = server default; True/False = force
    request_timeout: float = 60.0

    threshold_mode: str = "imp"  # "imp" | "score"
    threshold_n: int = 1

    max_rollout_calls: int = 40

    cache_dir: str = "data/llm_cache"
    data_dir: str = "data"

    seats: tuple[str, ...] = field(default_factory=lambda: ("N", "E", "S", "W"))

    def __post_init__(self) -> None:
        if self.backend not in ("ollama", "vllm"):
            raise ValueError(
                f"backend must be 'ollama' or 'vllm', got {self.backend!r}"
            )
        if self.threshold_mode not in ("imp", "score"):
            raise ValueError(
                f"threshold_mode must be 'imp' or 'score', got {self.threshold_mode!r}"
            )


# A module-level default instance for convenience.
DEFAULT_CONFIG = Config()
