"""Hardware fingerprinting and live memory sampling.

Benchmark numbers are meaningless without the machine they were produced on,
so every result file LocalMind writes carries a hardware snapshot.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

try:  # psutil is optional-but-recommended; degrade gracefully without it
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

# Process names Ollama uses across platforms/versions.
_OLLAMA_PROC_HINTS = ("ollama", "llama-server", "llama_server", "ollama_llama_server")


def _gpu_info() -> list[str]:
    gpus: list[str] = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            gpus += [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        except Exception:
            pass
    return gpus


def hardware_snapshot() -> dict:
    """A JSON-serialisable description of the machine running the benchmark."""
    snap = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or platform.uname().processor,
        "python": platform.python_version(),
        "gpus": _gpu_info(),
    }
    if psutil:
        snap["physical_cores"] = psutil.cpu_count(logical=False)
        snap["logical_cores"] = psutil.cpu_count(logical=True)
        snap["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
        snap["ram_available_gb"] = round(psutil.virtual_memory().available / 1024**3, 2)
        try:
            freq = psutil.cpu_freq()
            if freq:
                snap["cpu_max_mhz"] = round(freq.max, 0)
        except Exception:
            pass
    return snap


def ollama_processes() -> list:
    """Every live process that belongs to the Ollama runtime."""
    if not psutil:
        return []
    procs = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if any(h in name for h in _OLLAMA_PROC_HINTS) or any(h in cmd for h in _OLLAMA_PROC_HINTS):
                procs.append(p)
        except Exception:
            continue
    return procs


def ollama_rss_mb() -> float | None:
    """Summed resident-set size of the Ollama runtime, in MB.

    On CPU-only machines this *is* the model's memory footprint. On a discrete
    GPU most weights live in VRAM instead, so cross-check with `/api/ps`.
    """
    procs = ollama_processes()
    if not procs:
        return None
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except Exception:
            continue
    return round(total / 1024**2, 1) if total else None


@dataclass
class MemorySampler:
    """Background RSS sampler. Use as a context manager around a generation."""

    interval: float = 0.25
    samples: list[float] = field(default_factory=list)
    baseline_mb: float | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> "MemorySampler":
        self.baseline_mb = ollama_rss_mb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = ollama_rss_mb()
            if v is not None:
                self.samples.append(v)
            time.sleep(self.interval)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def peak_mb(self) -> float | None:
        return max(self.samples) if self.samples else None

    @property
    def mean_mb(self) -> float | None:
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None

    @property
    def delta_mb(self) -> float | None:
        if self.peak_mb is None or self.baseline_mb is None:
            return None
        return round(self.peak_mb - self.baseline_mb, 1)

    def summary(self) -> dict:
        return {
            "peak_rss_mb": self.peak_mb,
            "mean_rss_mb": self.mean_mb,
            "baseline_rss_mb": self.baseline_mb,
            "delta_rss_mb": self.delta_mb,
            "n_samples": len(self.samples),
        }
