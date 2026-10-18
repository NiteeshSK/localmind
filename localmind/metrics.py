"""Statistics helpers + result persistence.

Rule of the project: never report a single run. Report median, p95 and spread,
because local inference latency is bursty (thermal throttling, page cache,
other processes) and a lone number will lie to you.
"""
from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from .config import RESULTS_DIR, ensure_dirs
from .sysinfo import hardware_snapshot


def pct(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile. Works on tiny samples where numpy would over-promise."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, int(round((p / 100) * len(vals) + 0.5)) - 1))
    return vals[k]


def describe(values: Iterable[float | None], nd: int = 2) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), nd),
        "median": round(statistics.median(vals), nd),
        "min": round(min(vals), nd),
        "max": round(max(vals), nd),
        "stdev": round(statistics.pstdev(vals), nd) if len(vals) > 1 else 0.0,
        "p95": round(pct(vals, 95), nd) if pct(vals, 95) is not None else None,
        "cv": round(statistics.pstdev(vals) / statistics.fmean(vals), 4)
        if len(vals) > 1 and statistics.fmean(vals) else 0.0,
    }


def aggregate(metric_dicts: Sequence[dict]) -> dict:
    """Roll a list of GenMetrics dicts into one summary block."""
    ok = [m for m in metric_dicts if m.get("ok")]
    failed = len(metric_dicts) - len(ok)
    return {
        "runs": len(metric_dicts),
        "failed": failed,
        "success_rate": round(len(ok) / len(metric_dicts), 4) if metric_dicts else 0.0,
        "ttft_ms": describe([m.get("ttft_ms") for m in ok], 1),
        "decode_tps": describe([m.get("decode_tps") for m in ok]),
        "total_ms": describe([m.get("wall_ms") for m in ok], 1),
        "prompt_tps": describe([m.get("prompt_tps") for m in ok]),
        "load_ms": describe([m.get("load_ms") for m in ok], 1),
        "completion_tokens": describe([m.get("completion_tokens") for m in ok], 1),
        "peak_rss_mb": describe([m.get("peak_rss_mb") for m in ok], 1),
        "delta_rss_mb": describe([m.get("delta_rss_mb") for m in ok], 1),
    }


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_run(kind: str, payload: dict, name: str | None = None) -> Path:
    """Persist a result blob with a hardware fingerprint attached."""
    ensure_dirs()
    payload = {
        "kind": kind,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware_snapshot(),
        **payload,
    }
    path = RESULTS_DIR / f"{name or kind}-{timestamp()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_run(kind: str) -> Path | None:
    ensure_dirs()
    files = sorted(RESULTS_DIR.glob(f"{kind}-*.json"))
    return files[-1] if files else None


def speedup(a: float | None, b: float | None) -> str:
    """Human phrasing for 'a is N× faster than b'."""
    if not a or not b:
        return "n/a"
    return f"{a / b:.2f}x" if a >= b else f"{b / a:.2f}x slower"
