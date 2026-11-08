"""Minimal client for the Maritaca Sabiá API (OpenAI-compatible chat completions)."""

from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

ENV_KEY = "MARITACA_API_KEY"

_RETRYABLE = (
    "connecterror", "connecttimeout", "readtimeout", "writetimeout", "timeoutexception",
    "remoteprotocolerror", "poolTimeout".lower(), "name resolution", "connection reset",
    " 429", " 500", " 502", " 503", " 504",
)


def is_retryable(error: str) -> bool:
    """Whether an error from :meth:`LLMClient.complete` could plausibly succeed on a repeat."""
    low = (error or "").lower()
    return any(hint in low for hint in _RETRYABLE)


def retry_delay(attempt: int, base: float = 2.0, cap: float = 30.0) -> float:
    """Exponential backoff with jitter."""
    import random

    return min(cap, base * (2 ** attempt)) * (0.5 + random.random())


def load_llm_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path or os.environ.get("CONFIG_PATH", "ml/configs/default.yaml"))
    with open(cfg_path, encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("llm", {})


def is_enabled() -> bool:
    return bool(os.environ.get(ENV_KEY, "").strip())


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def estimate_cost(cfg: dict, model: str, n_calls: int, in_tokens: int, out_tokens: int) -> float:
    """Estimated spend in BRL."""
    price = (cfg.get("pricing") or {}).get(model)
    if not price:
        return 0.0
    return (n_calls * in_tokens * price["input"] + n_calls * out_tokens * price["output"]) / 1e6


def approx_tokens(text: str) -> int:
    """Rough token count for Portuguese, ~3.8 characters per token."""
    return max(1, int(len(text) / 3.8))


def expected_digit(top_logprobs: list | None) -> float | None:
    """Expected value of a single-digit answer, from the token distribution, scaled to [0, 1]."""
    if not top_logprobs:
        return None
    weights: dict[int, float] = {}
    for entry in top_logprobs:
        token = (entry.get("token") or "").strip()
        if token.isdigit() and len(token) == 1:
            weights[int(token)] = weights.get(int(token), 0.0) + math.exp(entry["logprob"])
    total = sum(weights.values())
    if total <= 0:
        return None
    return sum(d * w for d, w in weights.items()) / total / 9.0


@dataclass
class Usage:
    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost(self, cfg: dict, model: str) -> float:
        price = (cfg.get("pricing") or {}).get(model)
        if not price:
            return 0.0
        return (self.input_tokens * price["input"] + self.output_tokens * price["output"]) / 1e6


@dataclass
class LLMClient:
    cfg: dict
    model: str | None = None
    run_name: str = "run"
    usage: Usage = field(default_factory=Usage)
    _raw: list[dict] = field(default_factory=list)
    _system_once: str = ""

    def __post_init__(self) -> None:
        self.model = self.model or self.cfg.get("model", "sabiazinho-4")
        self.base_url = self.cfg.get("base_url", "https://chat.maritaca.ai/api").rstrip("/")
        self.max_calls = int(self.cfg.get("max_calls", 20000))
        self._client = httpx.Client(timeout=float(self.cfg.get("timeout_seconds", 60)))

    def complete(self, system: str, user: str, logprobs: bool = False,
                 meta: dict | None = None,
                 max_tokens: int | None = None) -> tuple[str | None, str | None, list | None]:
        """Return (completion_text, error, first_token_logprobs)."""
        if self.usage.calls >= self.max_calls:
            return None, f"call cap reached ({self.max_calls}) - refusing to spend further", None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(self.cfg.get("temperature", 0.0)),
            "max_tokens": int(max_tokens or self.cfg.get("max_tokens", 16)),
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = int(self.cfg.get("top_logprobs", 10))
        headers = {
            "Authorization": f"Bearer {os.environ.get(ENV_KEY, '').strip()}",
            "Content-Type": "application/json",
        }

        try:
            resp = self._client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - counted and surfaced, never hidden
            self.usage.calls += 1
            self.usage.errors += 1
            detail = ""
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f" | {exc.response.text[:300]}"
            return None, f"{type(exc).__name__}: {exc}{detail}", None

        self.usage.calls += 1
        used = data.get("usage") or {}
        self.usage.input_tokens += int(used.get("prompt_tokens", 0))
        self.usage.output_tokens += int(used.get("completion_tokens", 0))

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            self.usage.errors += 1
            return None, f"unexpected response shape: {json.dumps(data)[:300]}", None

        top = None
        if logprobs:
            content = ((choice.get("logprobs") or {}).get("content") or [])
            if content:
                top = content[0].get("top_logprobs")

        self._system_once = self._system_once or system
        self._raw.append({
            **(meta or {}),
            "completion": text, "top_logprobs": top,
        })
        return text, None, top

    def save_raw(self, suffix: str = "") -> Path | None:
        """Persist every completion."""
        if not self._raw:
            return None
        out_dir = _results_dir() / "llm"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.run_name}{suffix}.jsonl.gz"
        header = {"_header": True, "model": self.model, "system": self._system_once,
                  "rows": len(self._raw)}
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            for row in self._raw:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def report(self) -> str:
        return (f"{self.usage.calls} calls ({self.usage.errors} errors), "
                f"{self.usage.input_tokens} in / {self.usage.output_tokens} out tokens, "
                f"R${self.usage.cost(self.cfg, self.model):.4f}")

    def close(self) -> None:
        self._client.close()


def preflight(cfg: dict, model: str | None = None) -> int:
    """One cheap call to prove the key, the base URL and the model id are all correct."""
    if not is_enabled():
        print(f"[llm] {ENV_KEY} is not set. Put it in the gitignored .env (not .env.example, which "
              "is committed). No call was made.")
        return 1
    client = LLMClient(cfg=cfg, model=model, run_name="preflight")
    text, err, _ = client.complete("Responda apenas com o número.", "Quanto é 2 + 2?")
    if err:
        print(f"[llm] preflight FAILED using model '{client.model}': {err}")
        client.close()
        return 1
    print(f"[llm] preflight OK: model '{client.model}' returned {text!r} | {client.report()}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(preflight(load_llm_config()))
