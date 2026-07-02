"""On-disk LLM response cache (Phase 2).

Self-play rollout multiplies LLM calls (up to four seats x two lines per board),
so caching keeps eval runs cheap and reproducible. Keyed by a hash of
(model, temperature, think, prompt) — ``think`` is included so toggling the
Ollama thinking-mode setting doesn't silently serve a response generated under
the other setting. One JSON file per model under ``cache_dir``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Dict, Optional


class LLMCache:
    """Simple thread-safe JSON-file cache mapping prompt-hash -> raw response."""

    def __init__(self, cache_dir: str, model: str) -> None:
        self.cache_dir = cache_dir
        self.model = model
        os.makedirs(cache_dir, exist_ok=True)
        safe_model = model.replace("/", "_").replace(":", "_")
        self.path = os.path.join(cache_dir, f"{safe_model}.json")
        self._lock = threading.Lock()
        self._store: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _key(self, prompt: str, temperature: float, think: Optional[bool]) -> str:
        raw = f"{self.model}|{temperature}|{think}|{prompt}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(
        self, prompt: str, temperature: float, think: Optional[bool] = None
    ) -> Optional[str]:
        return self._store.get(self._key(prompt, temperature, think))

    def set(
        self,
        prompt: str,
        temperature: float,
        response: str,
        think: Optional[bool] = None,
    ) -> None:
        with self._lock:
            self._store[self._key(prompt, temperature, think)] = response
            self._flush()

    def _flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._store, fh)
        os.replace(tmp, self.path)
