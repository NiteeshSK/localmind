"""Thin, fully-instrumented client for the Ollama HTTP API.

Why not the `ollama` python package? Because Phase 1 is about *measurement*.
Streaming the raw NDJSON ourselves lets us stamp the wall clock at the exact
moment the first token lands (time-to-first-token), and then reconcile that
against the nanosecond counters the server reports in its final chunk.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import requests

from .config import DEFAULT_HOST, DEFAULT_TIMEOUT, RunConfig
from .sysinfo import MemorySampler

NS = 1_000_000_000.0


class OllamaError(RuntimeError):
    """Raised when the Ollama daemon is unreachable or returns an error."""


@dataclass
class GenMetrics:
    """Every number Phase 1 asks for, plus the ones you wish you had logged."""

    model: str
    temperature: float = 0.0
    seed: int | None = None

    # --- latency, measured client-side (what the user actually feels) ---
    ttft_ms: float | None = None          # time to first token
    wall_ms: float = 0.0                  # full request -> last token

    # --- server-reported breakdown (nanosecond counters from Ollama) ---
    load_ms: float = 0.0                  # weights load / cold-start cost
    prompt_tokens: int = 0
    prompt_eval_ms: float = 0.0           # prefill
    completion_tokens: int = 0
    eval_ms: float = 0.0                  # decode
    server_total_ms: float = 0.0

    # --- derived throughput ---
    decode_tps: float = 0.0               # completion tokens / decode time
    prompt_tps: float = 0.0               # prefill throughput
    end_to_end_tps: float = 0.0           # completion tokens / wall clock

    # --- memory ---
    peak_rss_mb: float | None = None
    delta_rss_mb: float | None = None
    vram_mb: float | None = None

    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def row(self) -> dict:
        """Compact view used by the CLI tables."""
        return {
            "model": self.model,
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms is not None else None,
            "decode_tps": round(self.decode_tps, 2),
            "total_ms": round(self.wall_ms, 1),
            "out_tokens": self.completion_tokens,
        }


@dataclass
class GenResult:
    text: str
    metrics: GenMetrics
    raw_final: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.metrics.ok


class OllamaClient:
    """All network access lives here so the rest of the codebase stays testable."""

    def __init__(self, host: str = DEFAULT_HOST, timeout: float = DEFAULT_TIMEOUT):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def is_up(self) -> bool:
        try:
            self._session.get(f"{self.host}/api/version", timeout=5).raise_for_status()
            return True
        except Exception:
            return False

    def version(self) -> str:
        try:
            r = self._session.get(f"{self.host}/api/version", timeout=5)
            return r.json().get("version", "unknown")
        except Exception:
            return "unreachable"

    def list_models(self) -> list[dict]:
        try:
            r = self._session.get(f"{self.host}/api/tags", timeout=15)
            r.raise_for_status()
            return r.json().get("models", [])
        except Exception as e:
            raise OllamaError(f"could not list models: {e}") from e

    def has_model(self, model: str) -> bool:
        names = {m.get("name", "") for m in self.list_models()}
        # `mistral` should match `mistral:latest`
        return model in names or f"{model}:latest" in names

    def running(self) -> list[dict]:
        """`/api/ps` — includes `size_vram`, the honest number on GPU boxes."""
        try:
            r = self._session.get(f"{self.host}/api/ps", timeout=10)
            r.raise_for_status()
            return r.json().get("models", [])
        except Exception:
            return []

    def vram_mb_for(self, model: str) -> float | None:
        for m in self.running():
            if m.get("name", "").startswith(model.split(":")[0]):
                v = m.get("size_vram") or 0
                return round(v / 1024**2, 1) if v else None
        return None

    def pull(self, model: str, on_progress: Callable[[str], None] | None = None) -> None:
        """Stream a model download so long pulls show progress instead of hanging."""
        with self._session.post(
            f"{self.host}/api/pull", json={"model": model, "stream": True},
            stream=True, timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if on_progress:
                    on_progress(chunk.get("status", ""))
                if chunk.get("error"):
                    raise OllamaError(chunk["error"])

    def unload(self, model: str) -> None:
        """Evict weights from RAM (`keep_alive: 0`) to force an honest cold start."""
        try:
            self._session.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=60,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        cfg: RunConfig,
        on_token: Callable[[str], None] | None = None,
        fmt: str | dict | None = None,
        sample_memory: bool = False,
    ) -> GenResult:
        payload: dict = {
            "model": cfg.model,
            "prompt": prompt,
            "stream": True,
            "options": cfg.options(),
            "keep_alive": cfg.keep_alive,
        }
        if cfg.system:
            payload["system"] = cfg.system
        if fmt is not None:
            payload["format"] = fmt
        return self._stream(f"{self.host}/api/generate", payload, cfg,
                            on_token, sample_memory, key="response")

    def chat(
        self,
        messages: list[dict],
        cfg: RunConfig,
        on_token: Callable[[str], None] | None = None,
        fmt: str | dict | None = None,
        sample_memory: bool = False,
    ) -> GenResult:
        msgs = list(messages)
        if cfg.system and not any(m["role"] == "system" for m in msgs):
            msgs = [{"role": "system", "content": cfg.system}] + msgs
        payload: dict = {
            "model": cfg.model,
            "messages": msgs,
            "stream": True,
            "options": cfg.options(),
            "keep_alive": cfg.keep_alive,
        }
        if fmt is not None:
            payload["format"] = fmt
        return self._stream(f"{self.host}/api/chat", payload, cfg,
                            on_token, sample_memory, key="message")

    # ------------------------------------------------------------------ #
    def _stream(
        self, url: str, payload: dict, cfg: RunConfig,
        on_token: Callable[[str], None] | None,
        sample_memory: bool, key: str,
    ) -> GenResult:
        metrics = GenMetrics(model=cfg.model, temperature=cfg.temperature, seed=cfg.seed)
        parts: list[str] = []
        final: dict = {}
        sampler = MemorySampler() if sample_memory else None

        t0 = time.perf_counter()
        try:
            if sampler:
                sampler.__enter__()
            with self._session.post(url, json=payload, stream=True, timeout=cfg.timeout) as r:
                if r.status_code >= 400:
                    raise OllamaError(f"HTTP {r.status_code}: {r.text[:300]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise OllamaError(chunk["error"])
                    piece = (
                        chunk.get("message", {}).get("content", "")
                        if key == "message" else chunk.get("response", "")
                    )
                    if piece:
                        if metrics.ttft_ms is None:
                            # First *visible* token — this is the perceived latency.
                            metrics.ttft_ms = (time.perf_counter() - t0) * 1000
                        parts.append(piece)
                        if on_token:
                            on_token(piece)
                    if chunk.get("done"):
                        final = chunk
            metrics.wall_ms = (time.perf_counter() - t0) * 1000
            self._absorb_server_counters(metrics, final)
        except Exception as e:
            metrics.wall_ms = (time.perf_counter() - t0) * 1000
            metrics.ok = False
            metrics.error = f"{type(e).__name__}: {e}"
        finally:
            if sampler:
                sampler.__exit__(None, None, None)
                metrics.peak_rss_mb = sampler.peak_mb
                metrics.delta_rss_mb = sampler.delta_mb
                metrics.vram_mb = self.vram_mb_for(cfg.model)

        return GenResult(text="".join(parts), metrics=metrics, raw_final=final)

    @staticmethod
    def _absorb_server_counters(m: GenMetrics, final: dict) -> None:
        """Convert Ollama's nanosecond counters into ms + tokens/sec."""
        m.load_ms = final.get("load_duration", 0) / 1e6
        m.prompt_tokens = final.get("prompt_eval_count", 0) or 0
        m.prompt_eval_ms = final.get("prompt_eval_duration", 0) / 1e6
        m.completion_tokens = final.get("eval_count", 0) or 0
        m.eval_ms = final.get("eval_duration", 0) / 1e6
        m.server_total_ms = final.get("total_duration", 0) / 1e6

        if m.eval_ms > 0:
            m.decode_tps = m.completion_tokens / (m.eval_ms / 1000)
        if m.prompt_eval_ms > 0:
            m.prompt_tps = m.prompt_tokens / (m.prompt_eval_ms / 1000)
        if m.wall_ms > 0:
            m.end_to_end_tps = m.completion_tokens / (m.wall_ms / 1000)


def token_printer(stream=None) -> Callable[[str], None]:
    """Callback that echoes tokens as they arrive, without buffering."""
    import sys
    out = stream or sys.stdout

    def _p(tok: str) -> None:
        out.write(tok)
        out.flush()

    return _p
