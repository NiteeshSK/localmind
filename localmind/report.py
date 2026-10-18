"""Turn saved result JSON into a Markdown technical report.

The deliverable for Phase 3 is a *report*, not a pile of JSON. This module
renders one that a reviewer can read top to bottom: hardware, method,
tables, and explicit conclusions.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import REPORTS_DIR, ensure_dirs
from .metrics import load_run


def _t(v, nd=1, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _hardware_block(hw: dict) -> str:
    rows = [
        ["OS", hw.get("os", "—")],
        ["CPU", hw.get("processor") or "—"],
        ["Cores", f"{hw.get('physical_cores', '—')} physical / {hw.get('logical_cores', '—')} logical"],
        ["RAM", f"{hw.get('ram_total_gb', '—')} GB total, {hw.get('ram_available_gb', '—')} GB free at start"],
        ["GPU", ", ".join(hw.get("gpus") or []) or "none detected (CPU-only inference)"],
        ["Python", hw.get("python", "—")],
    ]
    return _table(["Property", "Value"], rows)


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #
def render_benchmark(data: dict) -> str:
    s, cs = data.get("summary", {}), data.get("cold_start") or {}
    cfg = data.get("config", {})
    L: list[str] = []
    L.append(f"# Benchmark Report — `{data.get('model')}`\n")
    L.append(f"_Generated {data.get('created_at', datetime.now().isoformat())}_\n")
    L.append("## 1. Test machine\n")
    L.append(_hardware_block(data.get("hardware", {})) + "\n")
    L.append("## 2. Method\n")
    L.append(
        f"- Test set: **{cfg.get('testset')}**, {cfg.get('n_prompts')} prompts\n"
        f"- Repeats per prompt: **{cfg.get('repeats')}** (plus {cfg.get('warmup')} discarded warm-up run(s))\n"
        f"- Decoding: temperature **{cfg.get('temperature')}**, top_p {cfg.get('top_p')}, "
        f"seed `{cfg.get('seed')}`\n"
        f"- Total sampled generations: **{s.get('runs')}** "
        f"(success rate {(s.get('success_rate') or 0) * 100:.1f}%)\n"
    )
    L.append("## 3. Cold start\n")
    L.append(
        "Model evicted with `keep_alive: 0`, then timed on the next request. "
        "This is the cost a user pays on the first question after a pause.\n"
    )
    L.append(_table(
        ["Weight load", "TTFT", "Total", "Peak RSS"],
        [[f"{_t(cs.get('load_ms'))} ms", f"{_t(cs.get('ttft_ms'))} ms",
          f"{_t(cs.get('total_ms'))} ms", f"{_t(cs.get('peak_rss_mb'))} MB"]],
    ) + "\n")
    L.append("## 4. Steady-state performance\n")
    L.append(_table(
        ["Metric", "Median", "Mean", "p95", "Min", "Max", "CV"],
        [
            ["Time to first token (ms)", _t(s.get("ttft_ms", {}).get("median")),
             _t(s.get("ttft_ms", {}).get("mean")), _t(s.get("ttft_ms", {}).get("p95")),
             _t(s.get("ttft_ms", {}).get("min")), _t(s.get("ttft_ms", {}).get("max")),
             _t(s.get("ttft_ms", {}).get("cv"), 3)],
            ["Decode throughput (tok/s)", _t(s.get("decode_tps", {}).get("median"), 2),
             _t(s.get("decode_tps", {}).get("mean"), 2), _t(s.get("decode_tps", {}).get("p95"), 2),
             _t(s.get("decode_tps", {}).get("min"), 2), _t(s.get("decode_tps", {}).get("max"), 2),
             _t(s.get("decode_tps", {}).get("cv"), 3)],
            ["Prefill throughput (tok/s)", _t(s.get("prompt_tps", {}).get("median"), 2),
             _t(s.get("prompt_tps", {}).get("mean"), 2), _t(s.get("prompt_tps", {}).get("p95"), 2),
             _t(s.get("prompt_tps", {}).get("min"), 2), _t(s.get("prompt_tps", {}).get("max"), 2),
             _t(s.get("prompt_tps", {}).get("cv"), 3)],
            ["Total response (ms)", _t(s.get("total_ms", {}).get("median")),
             _t(s.get("total_ms", {}).get("mean")), _t(s.get("total_ms", {}).get("p95")),
             _t(s.get("total_ms", {}).get("min")), _t(s.get("total_ms", {}).get("max")),
             _t(s.get("total_ms", {}).get("cv"), 3)],
            ["Peak RSS (MB)", _t(s.get("peak_rss_mb", {}).get("median")),
             _t(s.get("peak_rss_mb", {}).get("mean")), _t(s.get("peak_rss_mb", {}).get("p95")),
             _t(s.get("peak_rss_mb", {}).get("min")), _t(s.get("peak_rss_mb", {}).get("max")), "—"],
        ],
    ) + "\n")
    pc = data.get("per_category", {})
    if pc:
        L.append("## 5. Per-category breakdown\n")
        rows = [[c,
                 _t(v.get("ttft_ms", {}).get("median")),
                 _t(v.get("decode_tps", {}).get("median"), 2),
                 _t(v.get("total_ms", {}).get("median")),
                 _t(v.get("completion_tokens", {}).get("median"), 0)]
                for c, v in pc.items()]
        L.append(_table(["Category", "TTFT median (ms)", "Decode (tok/s)",
                         "Total median (ms)", "Output tokens"], rows) + "\n")
    L.append("## 6. Reading these numbers\n")
    ttft = s.get("ttft_ms", {}).get("median")
    tps = s.get("decode_tps", {}).get("median")
    if ttft and tps:
        L.append(
            f"- A user waits about **{ttft/1000:.2f} s** before seeing anything, then text "
            f"arrives at **{tps:.1f} tok/s** (~{tps*0.75:.0f} words/s).\n"
            f"- A 300-token answer therefore takes roughly "
            f"**{(ttft/1000) + 300/tps:.1f} s** end to end.\n"
            f"- Human reading speed is ~4 words/s, so streaming feels acceptable above "
            f"roughly 6 tok/s. This configuration is "
            f"{'comfortably above' if tps > 8 else 'close to' if tps > 5 else 'below'} that line.\n"
        )
    cv = s.get("decode_tps", {}).get("cv")
    if cv is not None:
        L.append(
            f"- Throughput CV is **{cv:.3f}**. "
            f"{'Stable enough to quote a single number.' if cv < 0.15 else 'High variance — quote the median and p95, never the mean alone.'}\n"
        )
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Phase 2
# --------------------------------------------------------------------------- #
def render_tempstudy(data: dict) -> str:
    L: list[str] = []
    L.append("# Temperature Determinism Study\n")
    L.append(f"_Generated {data.get('created_at', '')}_\n")
    L.append(
        f"**Model:** `{data.get('model')}`  \n"
        f"**Samples per temperature:** {data.get('samples_per_temperature')}  \n"
        f"**Schema enforced:** `{data.get('schema') or 'none'}`  \n"
        f"**Seed pinned:** {data.get('fixed_seed')}\n"
    )
    L.append("## Prompt\n")
    L.append("```text\n" + str(data.get("prompt", ""))[:800] + "\n```\n")
    L.append("## Results\n")
    pts = data.get("points", [])
    headers = ["Temp", "Unique / N", "Dup rate", "Pairwise overlap",
               "Len mean (words)", "Len CV", "Decode tok/s"]
    has_schema = any("schema_valid_rate" in p for p in pts)
    if has_schema:
        headers += ["Schema valid", "Answer stability"]
    rows = []
    for p in pts:
        row = [p["temperature"], f"{p['unique_outputs']} / {p['n_samples']}",
               f"{p['exact_duplicate_rate']:.0%}", f"{p['mean_pairwise_jaccard']:.3f}",
               p["length_mean_words"], f"{p['length_cv']:.3f}", p["decode_tps_mean"]]
        if has_schema:
            row += [f"{p.get('schema_valid_rate', 0):.0%}",
                    f"{p.get('answer_stability', 0):.0%}" if p.get("answer_stability") else "—"]
        rows.append(row)
    L.append(_table(headers, rows) + "\n")
    L.append("## What the columns mean\n")
    L.append(
        "- **Pairwise overlap** — mean Jaccard similarity of word sets across every pair "
        "of samples. 1.000 means every sample used the same words.\n"
        "- **Len CV** — coefficient of variation of output length. Rises with temperature "
        "even when the content is stable.\n"
        "- **Schema valid** — how often the output still satisfied the Pydantic model. "
        "This is the column that decides whether a temperature is safe in production.\n"
        "- **Answer stability** — how often the *extracted value* (not the wording) "
        "matched the modal answer.\n"
    )
    L.append("## Conclusion\n")
    L.append(str(data.get("conclusion", "")) + "\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Phase 3
# --------------------------------------------------------------------------- #
def render_comparison(data: dict) -> str:
    L: list[str] = []
    ts, cfg = data.get("testset", {}), data.get("config", {})
    L.append("# Model Comparison Study\n")
    L.append(f"_Generated {data.get('created_at', '')}_\n")
    L.append("## 1. Test machine\n")
    L.append(_hardware_block(data.get("hardware", {})) + "\n")
    L.append("## 2. Protocol\n")
    L.append(
        f"- **Test set:** {ts.get('name')} v{ts.get('version')} — "
        f"{ts.get('n_prompts')} prompts across {len(ts.get('categories') or [])} categories "
        f"({', '.join(ts.get('categories') or [])})\n"
        f"- **Repeats:** {cfg.get('repeats')} per prompt per model\n"
        f"- **Decoding:** temperature {cfg.get('temperature')}, top_p {cfg.get('top_p')}, "
        f"seed `{cfg.get('seed')}` — identical for every model\n"
        f"- **Isolation:** each model is unloaded after its run so memory readings do "
        f"not overlap; each model gets its own discarded warm-up\n"
        f"- **Quality:** deterministic scorer (keyword coverage 45%, instruction "
        f"compliance 40%, non-degeneracy 15%)"
        + (f"; LLM judge: `{cfg.get('judge_model')}`\n" if cfg.get("judge_model") else "\n")
    )
    L.append("## 3. Headline results\n")
    rows = []
    for h in data.get("headlines", []):
        rows.append([
            f"`{h['model']}`", h.get("params") or "—", h.get("quant") or "—",
            _t(h.get("disk_mb"), 0), _t(h.get("ttft_ms_median"), 0),
            _t(h.get("decode_tps_median"), 2), _t(h.get("total_ms_median"), 0),
            _t(h.get("peak_rss_mb"), 0), _t(h.get("quality"), 1),
        ])
    L.append(_table(
        ["Model", "Params", "Quant", "Disk MB", "TTFT ms", "Decode tok/s",
         "Total ms", "Peak RSS MB", "Quality /100"], rows) + "\n")

    rk = data.get("rankings", {})
    if rk:
        L.append("## 4. Rankings\n")
        L.append(_table(["Criterion", "Order (best first)"], [
            ["Fastest decode", " > ".join(f"`{m}`" for m in rk.get("fastest_decode", []))],
            ["Lowest first-token latency", " > ".join(f"`{m}`" for m in rk.get("lowest_latency", []))],
            ["Smallest memory footprint", " > ".join(f"`{m}`" for m in rk.get("smallest_memory", []))],
            ["Highest quality score", " > ".join(f"`{m}`" for m in rk.get("highest_quality", []))],
        ]) + "\n")
        bv = rk.get("best_value", [])
        if bv:
            L.append("**Quality per second of wait** (composite score ÷ median response time):\n")
            L.append(_table(["Model", "Quality/s", "Quality", "Median response (s)"],
                            [[f"`{b['model']}`", b["quality_per_second"], b["quality"],
                              b["median_response_s"]] for b in bv]) + "\n")

    L.append("## 5. Per-category quality\n")
    reports = data.get("reports", [])
    cats = ts.get("categories") or []
    if cats and reports:
        head = ["Model"] + cats
        rows = []
        for r in reports:
            if not r.get("available"):
                continue
            pc = r.get("per_category", {})
            rows.append([f"`{r['model']}`"] +
                        [_t(pc.get(c, {}).get("quality_mean")) for c in cats])
        L.append(_table(head, rows) + "\n")

    L.append("## 6. Cold start\n")
    rows = []
    for r in reports:
        cs = r.get("cold_start") or {}
        rows.append([f"`{r['model']}`", _t(cs.get("load_ms"), 0),
                     _t(cs.get("ttft_ms"), 0), _t(cs.get("total_ms"), 0)])
    L.append(_table(["Model", "Weight load ms", "TTFT ms", "Total ms"], rows) + "\n")

    findings = data.get("findings", [])
    if findings:
        L.append("## 7. Findings\n")
        for f in findings:
            L.append(f"- {f}")
        L.append("")

    L.append("## 8. Selection guidance\n")
    bv = (data.get("rankings", {}) or {}).get("best_value", [])
    fast = (data.get("rankings", {}) or {}).get("fastest_decode", [])
    qual = (data.get("rankings", {}) or {}).get("highest_quality", [])
    mem = (data.get("rankings", {}) or {}).get("smallest_memory", [])
    if fast and qual and mem:
        L.append(
            f"- **Interactive chat / voice front-end** → `{fast[0]}`: first-token latency "
            f"dominates perceived quality here.\n"
            f"- **Batch extraction or offline document processing** → `{qual[0]}`: nobody "
            f"is watching the stream, so trade seconds for accuracy.\n"
            f"- **Running alongside an IDE, browser and containers on 16 GB** → `{mem[0]}`: "
            f"the memory headroom is worth more than the score.\n"
        )
    if bv:
        L.append(
            f"- **Default pick if you only deploy one** → `{bv[0]['model']}`, which returns "
            f"the most quality per second of user wait.\n"
        )
    L.append("## 9. Threats to validity\n")
    L.append(
        "- The quality scorer is heuristic. It rewards keyword coverage and instruction "
        "compliance, which correlates with usefulness but is not a substitute for human "
        "grading on open-ended prompts.\n"
        "- On CPU-only machines RSS is the model footprint; on a discrete GPU most weights "
        "sit in VRAM and RSS understates the true cost — cross-check `ollama ps`.\n"
        "- Laptop thermal throttling changes results between a cold and a hot chassis. "
        "Re-run back-to-back before drawing fine-grained conclusions.\n"
        "- A single seed at temperature 0 removes sampling noise but does not remove "
        "prompt sensitivity. Widening the test set matters more than adding repeats.\n"
    )
    return "\n".join(L)


RENDERERS = {
    "benchmark": render_benchmark,
    "tempstudy": render_tempstudy,
    "comparison": render_comparison,
}


def render(path: str | Path) -> str:
    data = load_run(path)
    kind = data.get("kind", "")
    if kind not in RENDERERS:
        raise ValueError(f"no renderer for kind '{kind}'")
    return RENDERERS[kind](data)


def write_report(path: str | Path, out_name: str | None = None) -> Path:
    ensure_dirs()
    md = render(path)
    stem = out_name or (Path(path).stem + ".md")
    out = REPORTS_DIR / (stem if stem.endswith(".md") else stem + ".md")
    out.write_text(md, encoding="utf-8")
    return out
