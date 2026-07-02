"""Local LLM client and self-play auction rollout (Phase 2).

``LocalLLMClient`` talks to any OpenAI-compatible endpoint (vLLM ``:8000`` or
Ollama ``:11434``) and returns a validated :class:`BridgeBid`. It also drives
the self-play auction rollout used by the double-dummy evaluator: starting from
a position plus a first call, it asks the model to bid all remaining seats until
the auction closes.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

import requests as _requests

from src.bridge import (
    auction_is_closed,
    count_hcp,
    normalize_call,
    seat_to_act,
)
from src.config import Config, DEFAULT_CONFIG
from src.harness.fsm_guardrail import BiddingFSM
from src.harness.llm_cache import LLMCache
from src.harness.prompt_builder import ContextBuilder
from src.schema.dataset import BridgeBid, MockDealRecord

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LocalLLMClient:
    """OpenAI-compatible client returning structured, legality-checked bids."""

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        builder: Optional[ContextBuilder] = None,
        fsm: Optional[BiddingFSM] = None,
        cache: Optional[LLMCache] = None,
    ) -> None:
        self.config = config
        self.builder = builder or ContextBuilder()
        self.fsm = fsm or BiddingFSM()
        self.cache = cache or LLMCache(config.cache_dir, config.model)
        self._client = None  # lazily constructed openai client

    # ------------------------------------------------------------------ #
    # Low-level call
    # ------------------------------------------------------------------ #
    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.request_timeout,
            )
        return self._client

    def verify_connection(self) -> None:
        """Fail fast and loudly if the endpoint or model is unreachable."""
        if self.config.backend == "ollama":
            self._verify_connection_ollama()
        else:
            self._verify_connection_vllm()

    def _verify_connection_vllm(self) -> None:
        client = self._ensure_client()
        try:
            available = [m.id for m in client.models.list().data]
        except Exception as e:
            raise RuntimeError(
                f"Could not reach vLLM endpoint at {self.config.base_url!r}: {e}\n"
                f"Is the server running? (vLLM: check --port.)"
            ) from e
        if self.config.model not in available:
            raise RuntimeError(
                f"Model {self.config.model!r} not found at {self.config.base_url!r}.\n"
                f"Available models: {available or '(none)'}"
            )

    def _verify_connection_ollama(self) -> None:
        tags_url = f"{self.config.base_url}/api/tags"
        try:
            resp = _requests.get(tags_url, timeout=self.config.request_timeout)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.config.base_url!r}: {e}\n"
                f"Is the server running? ('ollama serve')"
            ) from e
        if self.config.model not in available:
            raise RuntimeError(
                f"Model {self.config.model!r} not found in Ollama.\n"
                f"Available models: {available or '(none)'}\n"
                f"Hint: model names include the tag, e.g. 'llama3:8b'. "
                f"Run 'ollama list' to see exact installed tags."
            )

    def _raw_call(self, prompt: str) -> str:
        """Return raw model text, using the cache when available."""
        if self.config.backend == "vllm":
            return self._raw_call_vllm(prompt)
        elif self.config.backend == "ollama":
            return self._raw_call_ollama(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend!r}")

    def _raw_call_vllm(self, prompt: str) -> str:
        """OpenAI-SDK call against a vLLM /v1 endpoint."""
        think = self.config.think
        cached = self.cache.get(prompt, self.config.temperature, think)
        if cached is not None:
            return cached

        client = self._ensure_client()
        kwargs = dict(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        if think is not None:
            kwargs["extra_body"] = {"think": think}

        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        self.cache.set(prompt, self.config.temperature, text, think)
        return text

    def _raw_call_ollama(self, prompt: str) -> str:
        """Native Ollama /api/chat call — correctly honours think:false."""
        think = self.config.think
        cached = self.cache.get(prompt, self.config.temperature, think)
        if cached is not None:
            return cached

        payload: dict = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
        }
        if think is not None:
            payload["think"] = think

        try:
            resp = _requests.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ollama /api/chat request failed: {e}") from e

        msg = resp.json()["message"]
        text = msg.get("content") or ""
        # Surface server-side reasoning into the response when the model's own
        # JSON doesn't include a non-empty thinking field.
        server_thinking = msg.get("thinking") or ""
        if server_thinking:
            try:
                data = json.loads(text) if text else {}
                if not data.get("thinking"):
                    data["thinking"] = server_thinking
                    text = json.dumps(data)
            except (json.JSONDecodeError, TypeError):
                pass  # leave text as-is; _parse_bid will handle or fallback

        self.cache.set(prompt, self.config.temperature, text, think)
        return text

    # ------------------------------------------------------------------ #
    # Bid generation
    # ------------------------------------------------------------------ #
    def get_bid(self, prompt: str, history: Optional[List[str]] = None) -> BridgeBid:
        """Return a validated bid. Falls back to ``Pass`` on any failure.

        If ``history`` is given, the parsed bid is also checked for legality via
        :class:`BiddingFSM`; an illegal call falls back to ``Pass``.

        The three failure modes are distinguished in ``thinking`` (prefixes
        ``"transport error:"``, ``"parse error:"``, ``"illegal call"``) so a
        dead endpoint or a model that can't follow the JSON schema doesn't
        look identical to an occasional FSM rejection. See
        :func:`src.evaluation.metrics.evaluate_dataset`, which tallies these
        into a ``fallback_counts`` summary.
        """
        try:
            raw = self._raw_call(prompt)
        except Exception as e:
            return BridgeBid(thinking=f"transport error: {e}", bid="Pass")

        try:
            bid = self._parse_bid(raw)
        except Exception as e:
            return BridgeBid(thinking=f"parse error: {e}", bid="Pass")

        if history is not None and not self.fsm.is_valid_bid(history, bid.bid):
            return BridgeBid(thinking=f"illegal call {bid.bid!r} -> Pass", bid="Pass")
        return bid

    def _parse_bid(self, raw: str) -> BridgeBid:
        """Parse model text into a normalized :class:`BridgeBid`.

        Strips any inline ``<think>...</think>`` block first: some backends
        put reasoning in a separate response field, but others (depending on
        model/template) inline it into ``content`` regardless of the ``think``
        setting, which would otherwise break JSON extraction.

        Extraction uses ``raw_decode`` from the first ``{``, which parses the
        first complete JSON object and ignores any trailing content — some
        models emit a valid object followed by extra prose or a second object,
        which ``json.loads`` would reject with "Extra data".
        """
        cleaned = _THINK_TAG_RE.sub("", raw)
        start = cleaned.find("{")
        if start == -1:
            raise ValueError(f"no JSON object found in model output: {raw!r}")
        data, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        bid = BridgeBid(**data)
        bid.bid = normalize_call(bid.bid)  # raises ValueError on garbage
        return bid

    # ------------------------------------------------------------------ #
    # Self-play auction rollout
    # ------------------------------------------------------------------ #
    def rollout_auction(self, record: MockDealRecord, first_bid: str) -> List[str]:
        """Complete the auction from ``record`` after the active seat's ``first_bid``.

        The active seat plays ``first_bid``; remaining seats are bid by the model
        (using their own hands from ``record.all_hands``) until the auction closes
        or the safety cap is reached.
        """
        auction: List[str] = list(record.current_bidding) + [normalize_call(first_bid)]
        calls_made = 0
        while not auction_is_closed(auction) and calls_made < self.config.max_rollout_calls:
            seat = seat_to_act(record.dealer, len(auction))
            hand = record.all_hands.get(seat)
            if hand is None:
                # No hand to bid with; pass to avoid an infinite loop.
                auction.append("Pass")
                calls_made += 1
                continue
            hcp = count_hcp(hand)
            prompt = self.builder.build_prompt_parts(seat, hand, hcp, auction)
            bid = self.get_bid(prompt, history=auction)
            auction.append(bid.bid)
            calls_made += 1

        # Guarantee a closed auction for downstream settling.
        if not auction_is_closed(auction):
            auction.extend(["Pass", "Pass", "Pass"])
        return auction
