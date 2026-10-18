"""Central configuration. Everything is overridable by env var or CLI flag."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


RESULTS_DIR = _path_env("LOCALMIND_RESULTS", PROJECT_ROOT / "results")
REPORTS_DIR = _path_env("LOCALMIND_REPORTS", PROJECT_ROOT / "reports")
TESTSET_PATH = _path_env("LOCALMIND_TESTSET", PACKAGE_ROOT / "data" / "testset.json")

# Sensible defaults for a 16 GB CPU-only laptop.
DEFAULT_HOST = os.environ.get("LOCALMIND_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LOCALMIND_MODEL", "llama3.2:3b")
DEFAULT_TIMEOUT = float(os.environ.get("LOCALMIND_TIMEOUT", "600"))

DEFAULT_SYSTEM_PROMPT = (
    "You are LocalMind, a concise offline assistant running on the user's own machine. "
    "Answer accurately. If you do not know something, say so plainly instead of guessing."
)

# The comparison roster used by `localmind compare` when no --model flags are given.
DEFAULT_COMPARE_MODELS = ["llama3.2:3b", "phi4-mini:3.8b", "mistral:7b"]

# The quantization ladder used by `localmind compare --preset quant`.
DEFAULT_QUANT_MODELS = [
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q5_K_M",
    "llama3.2:3b-instruct-q8_0",
]


@dataclass
class RunConfig:
    """Options that control a single generation call."""

    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    temperature: float = 0.0
    top_p: float = 0.9
    seed: int | None = None
    num_predict: int = 512
    num_ctx: int | None = None
    system: str | None = DEFAULT_SYSTEM_PROMPT
    keep_alive: str = "5m"
    timeout: float = DEFAULT_TIMEOUT
    stop: list[str] = field(default_factory=list)

    def options(self) -> dict:
        """Translate into the Ollama `options` payload (only non-None keys)."""
        opts: dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
        }
        if self.seed is not None:
            opts["seed"] = self.seed
        if self.num_ctx is not None:
            opts["num_ctx"] = self.num_ctx
        if self.stop:
            opts["stop"] = self.stop
        return opts


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
