"""Phase 3: the model comparison study.

Fairness rules enforced here (they are the whole point of the exercise):

* identical prompts, identical decoding params, identical machine, same session
* every model gets its own warm-up, and its cold start is measured separately
* models are unloaded between runs so one model's residency never inflates
  another's memory reading
* quality is scored by the *same* deterministic scorer for everyone
* results carry the hardware snapshot, so the table can never be quoted
  out of context
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .benchmark import measure_cold_start
from .client import OllamaClient
from .config import RunConfig
from .metrics import aggregate, save_run
from .quality import HeuristicScorer, LLMJudge
from .testset import TestSet


@dataclass
class ModelReport:
    model: str
    available: bool = True
    error: str | None = None
    params: str | None = None
    quantization: str | None = None
    disk_size_mb: float | None = None
    cold_start: dict | None = None
    perf: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    per_category: dict = field(default_factory=dict)
    runs: list[dict] = field(default_factory=list)

    def headline(self) -> dict:
        return {
            "model": self.model,
            "params": self.params,
            "quant": self.quantization,
            "disk_mb": self.disk_size_mb,
            "ttft_ms_median": self.perf.get("ttft_ms", {}).get("median"),
            "decode_tps_median": self.perf.get("decode_tps", {}).get("median"),
            "total_ms_median": self.perf.get("total_ms", {}).get("median"),
            "peak_rss_mb": self.perf.get("peak_rss_mb", {}).get("max"),
            "quality": self.quality.get("composite_mean"),
            "success_rate": self.perf.get("success_rate"),
        }


def _model_meta(client: OllamaClient, model: str) -> dict:
    for m in client.list_models():
        if m.get("name") == model or m.get("name") == f"{model}:latest":
            det = m.get("details", {}) or {}
            return {
                "params": det.get("parameter_size"),
                "quantization": det.get("quantization_level"),
                "disk_size_mb": round((m.get("size") or 0) / 1024**2, 1) or None,
            }
    return {}


def evaluate_model(
    client: OllamaClient,
    model: str,
    testset: TestSet,
    base_cfg: RunConfig,
    repeats: int = 1,
    judge: LLMJudge | None = None,
    unload_after: bool = True,
    on_progress=None,
) -> ModelReport:
    report = ModelReport(model=model, **_model_meta(client, model))

    if not client.has_model(model):
        report.available = False
        report.error = f"model '{model}' is not pulled. Run: ollama pull {model}"
        return report

    scorer = HeuristicScorer()
    report.cold_start = measure_cold_start(client, RunConfig(**{**base_cfg.__dict__,
                                                               "model": model}))
    # warm-up (discarded)
    client.generate("Warm up. Reply with the word ready.",
                    RunConfig(**{**base_cfg.__dict__, "model": model, "num_predict": 8}))

    scores, judge_scores = [], []
    total, done = len(testset) * repeats, 0

    for spec in testset.prompts:
        cfg = RunConfig(**{**base_cfg.__dict__, "model": model,
                           "num_predict": spec.num_predict})
        for rep in range(repeats):
            res = client.generate(spec.prompt, cfg, sample_memory=True)
            q = scorer.score(spec, res.text)
            if judge and res.metrics.ok:
                q.judge = judge.score(spec, res.text)
                if q.judge:
                    judge_scores.append(q.judge["mean"])
            scores.append(q)

            row = res.metrics.to_dict()
            row.update({
                "prompt_id": spec.id, "category": spec.category, "repeat": rep,
                "quality": q.to_dict(), "output_chars": len(res.text),
                "output_preview": res.text[:300],
            })
            report.runs.append(row)
            done += 1
            if on_progress:
                on_progress(model, spec.id, done, total)

    report.perf = aggregate(report.runs)
    comps = [s.composite for s in scores]
    report.quality = {
        "composite_mean": round(statistics.fmean(comps), 2) if comps else 0.0,
        "composite_median": round(statistics.median(comps), 2) if comps else 0.0,
        "composite_stdev": round(statistics.pstdev(comps), 2) if len(comps) > 1 else 0.0,
        "n_scored": len(comps),
        "n_empty": sum(1 for s in scores if s.empty),
        "n_compliance_failures": sum(1 for s in scores if s.failures),
        "judge_mean_1to5": round(statistics.fmean(judge_scores), 3) if judge_scores else None,
        "judge_n": len(judge_scores),
    }

    cats = sorted({s.category for s in scores})
    report.per_category = {
        c: {
            "quality_mean": round(
                statistics.fmean([s.composite for s in scores if s.category == c]), 2),
            "perf": aggregate([r for r in report.runs if r["category"] == c]),
        }
        for c in cats
    }

    if unload_after:
        client.unload(model)
    return report


def run_comparison(
    client: OllamaClient,
    models: list[str],
    testset: TestSet,
    base_cfg: RunConfig,
    repeats: int = 1,
    judge_model: str | None = None,
    on_progress=None,
) -> dict:
    judge = LLMJudge(client, judge_model) if judge_model else None
    reports = [
        evaluate_model(client, m, testset, base_cfg, repeats, judge,
                       on_progress=on_progress)
        for m in models
    ]
    ok = [r for r in reports if r.available]
    return {
        "models": models,
        "testset": {"name": testset.name, "version": testset.version,
                    "n_prompts": len(testset), "categories": testset.categories()},
        "config": {"temperature": base_cfg.temperature, "seed": base_cfg.seed,
                   "top_p": base_cfg.top_p, "repeats": repeats,
                   "judge_model": judge_model},
        "headlines": [r.headline() for r in reports],
        "rankings": _rank(ok),
        "findings": _findings(ok),
        "reports": [r.__dict__ for r in reports],
    }


def _rank(reports: list[ModelReport]) -> dict:
    def by(key, reverse):
        vals = [(r.model, r.headline().get(key)) for r in reports
                if r.headline().get(key) is not None]
        return [m for m, _ in sorted(vals, key=lambda x: x[1], reverse=reverse)]

    return {
        "fastest_decode": by("decode_tps_median", True),
        "lowest_latency": by("ttft_ms_median", False),
        "smallest_memory": by("peak_rss_mb", False),
        "highest_quality": by("quality", True),
        "best_value": _value_ranking(reports),
    }


def _value_ranking(reports: list[ModelReport]) -> list[dict]:
    """Quality per second of wait — the metric that actually drives selection."""
    out = []
    for r in reports:
        h = r.headline()
        q, t = h.get("quality"), h.get("total_ms_median")
        if q is None or not t:
            continue
        out.append({"model": r.model,
                    "quality_per_second": round(q / (t / 1000), 2),
                    "quality": q, "median_response_s": round(t / 1000, 2)})
    return sorted(out, key=lambda d: d["quality_per_second"], reverse=True)


def _findings(reports: list[ModelReport]) -> list[str]:
    """Auto-written observations, so the report states conclusions, not just tables."""
    out: list[str] = []
    if len(reports) < 2:
        return out
    hs = [r.headline() for r in reports]

    spd = [h for h in hs if h.get("decode_tps_median")]
    if len(spd) >= 2:
        f = max(spd, key=lambda h: h["decode_tps_median"])
        s = min(spd, key=lambda h: h["decode_tps_median"])
        out.append(
            f"{f['model']} decodes at {f['decode_tps_median']:.1f} tok/s versus "
            f"{s['decode_tps_median']:.1f} tok/s for {s['model']} — "
            f"{f['decode_tps_median'] / s['decode_tps_median']:.2f}x faster."
        )

    ql = [h for h in hs if h.get("quality") is not None]
    if len(ql) >= 2:
        b = max(ql, key=lambda h: h["quality"])
        w = min(ql, key=lambda h: h["quality"])
        spread = b["quality"] - w["quality"]
        if spread < 0.5:
            out.append(
                f"Quality is effectively tied at ~{b['quality']:.1f}/100 across all models "
                f"(spread {spread:.1f} points). When accuracy does not separate them, pick "
                f"on speed and memory."
            )
        else:
            out.append(
                f"Quality spread is {spread:.1f} points "
                f"({b['model']} {b['quality']:.1f} vs {w['model']} {w['quality']:.1f}) on the "
                f"deterministic scorer."
            )

    mem = [h for h in hs if h.get("peak_rss_mb")]
    if len(mem) >= 2:
        big = max(mem, key=lambda h: h["peak_rss_mb"])
        small = min(mem, key=lambda h: h["peak_rss_mb"])
        ratio = big["peak_rss_mb"] / small["peak_rss_mb"]
        if ratio < 1.05:
            out.append(
                f"Peak resident memory is within 5% across models (~{big['peak_rss_mb']:.0f} MB). "
                f"On a GPU box this is expected — check `ollama ps` for VRAM instead."
            )
        else:
            out.append(
                f"Peak resident memory ranges from {small['peak_rss_mb']:.0f} MB "
                f"({small['model']}) to {big['peak_rss_mb']:.0f} MB ({big['model']}), a "
                f"{ratio:.2f}x difference — the binding constraint when the machine is also "
                f"running an IDE, a browser and containers."
            )

    if len(ql) >= 2 and len(spd) >= 2:
        fastest = max(spd, key=lambda h: h["decode_tps_median"])
        best_q = max(ql, key=lambda h: h["quality"])
        if fastest["model"] != best_q["model"] and abs(
            max(h["quality"] for h in ql) - min(h["quality"] for h in ql)) >= 0.5:
            out.append(
                f"No single winner: {fastest['model']} is fastest, {best_q['model']} scores "
                f"highest. Selection depends on whether the workload is latency-bound or "
                f"accuracy-bound."
            )
        else:
            out.append(
                f"{fastest['model']} wins on both speed and quality here, which is the "
                f"easy case — take it unless memory is the constraint."
            )
    return out


def save_comparison(payload: dict) -> str:
    return str(save_run("comparison", payload, name="comparison"))
