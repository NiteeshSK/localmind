"""Phase 1: the measurement harness.

Design decisions that matter more than the code:

* **Warm-up runs are discarded.** The first call after a model loads pays the
  weight-load tax and a cold page cache. Mixing that into the average hides
  the steady-state number you actually care about.
* **Cold start is measured separately**, on purpose, by evicting the model
  first (`keep_alive: 0`). On a laptop this is often the single biggest
  latency term and it deserves its own row.
* **Every prompt runs N times.** One sample of a noisy system is an anecdote.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .client import OllamaClient
from .config import RunConfig
from .metrics import aggregate, save_run
from .testset import TestSet


@dataclass
class BenchmarkResult:
    model: str
    config: dict
    runs: list[dict] = field(default_factory=list)
    cold_start: dict | None = None
    summary: dict = field(default_factory=dict)
    per_category: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "config": self.config,
            "cold_start": self.cold_start,
            "summary": self.summary,
            "per_category": self.per_category,
            "runs": self.runs,
        }


def measure_cold_start(client: OllamaClient, cfg: RunConfig) -> dict:
    """Evict the model, then time the very next request end-to-end."""
    client.unload(cfg.model)
    probe = RunConfig(**{**cfg.__dict__, "num_predict": 16})
    res = client.generate("Say OK.", probe, sample_memory=True)
    m = res.metrics
    return {
        "ok": m.ok,
        "error": m.error,
        "load_ms": round(m.load_ms, 1),
        "ttft_ms": round(m.ttft_ms, 1) if m.ttft_ms else None,
        "total_ms": round(m.wall_ms, 1),
        "peak_rss_mb": m.peak_rss_mb,
        "delta_rss_mb": m.delta_rss_mb,
        "vram_mb": m.vram_mb,
    }


def run_benchmark(
    client: OllamaClient,
    cfg: RunConfig,
    testset: TestSet,
    repeats: int = 3,
    warmup: int = 1,
    measure_cold: bool = True,
    sample_memory: bool = True,
    on_progress=None,
) -> BenchmarkResult:
    result = BenchmarkResult(
        model=cfg.model,
        config={
            "temperature": cfg.temperature, "seed": cfg.seed, "top_p": cfg.top_p,
            "repeats": repeats, "warmup": warmup,
            "testset": f"{testset.name} v{testset.version}",
            "n_prompts": len(testset),
        },
    )

    if measure_cold:
        if on_progress:
            on_progress("cold-start", 0, 1)
        result.cold_start = measure_cold_start(client, cfg)

    # Warm-up: run and throw away. Removes load + cache effects from the sample.
    for _ in range(warmup):
        client.generate("Warm up. Reply with the word ready.",
                        RunConfig(**{**cfg.__dict__, "num_predict": 8}))

    total = len(testset) * repeats
    done = 0
    for spec in testset.prompts:
        run_cfg = RunConfig(**{**cfg.__dict__, "num_predict": spec.num_predict})
        for rep in range(repeats):
            res = client.generate(spec.prompt, run_cfg, sample_memory=sample_memory)
            row = res.metrics.to_dict()
            row.update({
                "prompt_id": spec.id, "category": spec.category, "repeat": rep,
                "output_chars": len(res.text),
                "output_preview": res.text[:200],
            })
            result.runs.append(row)
            done += 1
            if on_progress:
                on_progress(spec.id, done, total)

    result.summary = aggregate(result.runs)
    cats = sorted({r["category"] for r in result.runs})
    result.per_category = {
        c: aggregate([r for r in result.runs if r["category"] == c]) for c in cats
    }
    return result


def save_benchmark(result: BenchmarkResult) -> str:
    tag = result.model.replace(":", "_").replace("/", "_")
    return str(save_run("benchmark", result.to_dict(), name=f"benchmark-{tag}"))
