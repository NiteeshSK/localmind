"""LocalMind command-line interface.

    localmind doctor                     environment + hardware check
    localmind models                     what is installed locally
    localmind ask "..."                  one-shot streaming answer + live metrics
    localmind chat                       interactive REPL
    localmind structured "..." -s plan   schema-enforced JSON (Phase 2)
    localmind bench                      Phase 1 benchmark
    localmind temp-study                 Phase 2 determinism study
    localmind compare                    Phase 3 model comparison
    localmind report <results.json>      render a Markdown report
    localmind serve                      FastAPI wrapper
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from .benchmark import run_benchmark, save_benchmark
from .client import OllamaClient, OllamaError
from .compare import run_comparison, save_comparison
from .config import (
    DEFAULT_COMPARE_MODELS,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_QUANT_MODELS,
    DEFAULT_SYSTEM_PROMPT,
    RunConfig,
    ensure_dirs,
)
from .metrics import latest_run
from .report import write_report
from .schemas import SCHEMAS, get_schema
from .structured import StructuredGenerator
from .sysinfo import hardware_snapshot
from .tempstudy import run_temperature_study, save_temperature_study
from .testset import load_testset

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LocalMind — a fully offline AI assistant with a measurement habit.",
)
console = Console()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _client(host: str) -> OllamaClient:
    c = OllamaClient(host)
    if not c.is_up():
        console.print(Panel(
            f"[bold red]Cannot reach Ollama at {host}[/]\n\n"
            "Start it with [bold]ollama serve[/] (or launch the Ollama app), then retry.\n"
            "Install: https://ollama.com/download",
            title="offline", border_style="red"))
        raise typer.Exit(1)
    return c


def _cfg(model: str, host: str, temperature: float, seed: int | None,
         num_predict: int, system: str | None) -> RunConfig:
    return RunConfig(model=model, host=host, temperature=temperature, seed=seed,
                     num_predict=num_predict, system=system)


def _metrics_panel(m, title: str = "metrics") -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="bold cyan")
    ttft = f"{m.ttft_ms:.0f} ms" if m.ttft_ms is not None else "—"
    t.add_row("time to first token", ttft)
    t.add_row("decode throughput", f"{m.decode_tps:.2f} tok/s")
    t.add_row("prefill throughput", f"{m.prompt_tps:.1f} tok/s")
    t.add_row("total latency", f"{m.wall_ms:.0f} ms")
    t.add_row("tokens in / out", f"{m.prompt_tokens} / {m.completion_tokens}")
    if m.load_ms > 1:
        t.add_row("weight load", f"{m.load_ms:.0f} ms")
    if m.peak_rss_mb:
        t.add_row("peak RSS", f"{m.peak_rss_mb:.0f} MB")
    return Panel(t, title=title, border_style="cyan", expand=False)


def _progress():
    return Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    )


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command()
def version() -> None:
    """Print the LocalMind version."""
    console.print(f"LocalMind [bold cyan]v{__version__}[/]")


@app.command()
def doctor(host: str = typer.Option(DEFAULT_HOST, "--host")) -> None:
    """Check the runtime, the hardware, and whether anything is actually installed."""
    hw = hardware_snapshot()
    t = Table(title="machine", show_header=False, box=None)
    t.add_column(style="dim")
    t.add_column(style="bold")
    for k, v in hw.items():
        t.add_row(k.replace("_", " "), str(v) if not isinstance(v, list) else (", ".join(v) or "none"))
    console.print(t)

    c = OllamaClient(host)
    up = c.is_up()
    console.print(f"\nOllama at [bold]{host}[/]: "
                  + ("[bold green]reachable[/] (v" + c.version() + ")" if up
                     else "[bold red]unreachable[/] — run `ollama serve`"))
    if not up:
        raise typer.Exit(1)

    models = c.list_models()
    if not models:
        console.print("\n[yellow]No models installed.[/] Try:\n"
                      "  ollama pull llama3.2:3b\n  ollama pull phi4-mini:3.8b\n"
                      "  ollama pull mistral:7b")
        return
    mt = Table(title=f"{len(models)} local model(s)")
    for col in ("name", "params", "quant", "size"):
        mt.add_column(col)
    for m in models:
        d = m.get("details", {}) or {}
        mt.add_row(m.get("name", "?"), d.get("parameter_size", "—"),
                   d.get("quantization_level", "—"),
                   f"{(m.get('size') or 0) / 1024**3:.2f} GB")
    console.print(mt)

    ram = hw.get("ram_total_gb") or 0
    if ram and ram <= 16 and not hw.get("gpus"):
        console.print(Panel(
            "CPU-only inference on ≤16 GB RAM. Practical ceiling is a 7B model at Q4 "
            "(~4.5 GB resident). Expect roughly 10-16 tok/s for a 3B and 4-7 tok/s for a "
            "7B. Full Phi-4 (14B) will technically load but is too slow to sit in front "
            "of — use [bold]phi4-mini:3.8b[/] instead.",
            title="sizing note", border_style="yellow"))


@app.command()
def models(host: str = typer.Option(DEFAULT_HOST, "--host")) -> None:
    """List locally installed models and anything currently resident in memory."""
    c = _client(host)
    doctor(host=host)
    running = c.running()
    if running:
        rt = Table(title="resident now")
        for col in ("name", "size", "vram"):
            rt.add_column(col)
        for m in running:
            rt.add_row(m.get("name", "?"),
                       f"{(m.get('size') or 0) / 1024**3:.2f} GB",
                       f"{(m.get('size_vram') or 0) / 1024**3:.2f} GB")
        console.print(rt)


@app.command()
def pull(model: str, host: str = typer.Option(DEFAULT_HOST, "--host")) -> None:
    """Download a model through Ollama with live progress."""
    c = _client(host)
    with console.status(f"pulling {model}…") as st:
        try:
            c.pull(model, on_progress=lambda s: st.update(f"{model}: {s}"))
        except OllamaError as e:
            console.print(f"[red]pull failed:[/] {e}")
            raise typer.Exit(1)
    console.print(f"[green]✓[/] {model} ready")


@app.command()
def ask(
    prompt: str,
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    host: str = typer.Option(DEFAULT_HOST, "--host"),
    temperature: float = typer.Option(0.0, "--temperature", "-t"),
    seed: int = typer.Option(None, "--seed"),
    num_predict: int = typer.Option(512, "--max-tokens", "-n"),
    system: str = typer.Option(DEFAULT_SYSTEM_PROMPT, "--system"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="hide the metrics panel"),
    json_out: bool = typer.Option(False, "--json", help="emit {text, metrics} as JSON"),
) -> None:
    """Ask one question. Streams the answer, then shows what it cost."""
    c = _client(host)
    cfg = _cfg(model, host, temperature, seed, num_predict, system)
    if json_out:
        res = c.generate(prompt, cfg, sample_memory=True)
        console.print_json(json.dumps({"text": res.text, "metrics": res.metrics.to_dict()}))
        raise typer.Exit(0 if res.ok else 1)

    console.print(f"[dim]{model} · T={temperature}[/]\n")
    res = c.generate(prompt, cfg, on_token=lambda t: (sys.stdout.write(t), sys.stdout.flush()),
                     sample_memory=True)
    console.print()
    if not res.ok:
        console.print(f"[red]error:[/] {res.metrics.error}")
        raise typer.Exit(1)
    if not quiet:
        console.print(_metrics_panel(res.metrics))


@app.command()
def chat(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    host: str = typer.Option(DEFAULT_HOST, "--host"),
    temperature: float = typer.Option(0.7, "--temperature", "-t"),
    num_predict: int = typer.Option(512, "--max-tokens", "-n"),
    system: str = typer.Option(DEFAULT_SYSTEM_PROMPT, "--system"),
) -> None:
    """Interactive offline chat with conversation memory and per-turn metrics."""
    c = _client(host)
    cfg = _cfg(model, host, temperature, None, num_predict, system)
    history: list[dict] = []
    console.print(Panel(
        f"[bold]LocalMind[/] · {model} · T={temperature}\n"
        "[dim]/reset  /model <name>  /temp <0-1>  /stats  /save <file>  /exit[/]",
        border_style="cyan"))
    turn_metrics = []

    while True:
        try:
            user = console.input("\n[bold green]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            break
        if not user:
            continue
        if user in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]bye[/]")
            break
        if user == "/reset":
            history.clear()
            console.print("[dim]context cleared[/]")
            continue
        if user.startswith("/model "):
            cfg.model = user.split(" ", 1)[1].strip()
            console.print(f"[dim]model → {cfg.model}[/]")
            continue
        if user.startswith("/temp "):
            try:
                cfg.temperature = float(user.split(" ", 1)[1])
                console.print(f"[dim]temperature → {cfg.temperature}[/]")
            except ValueError:
                console.print("[red]usage: /temp 0.7[/]")
            continue
        if user == "/stats":
            if not turn_metrics:
                console.print("[dim]no turns yet[/]")
            else:
                avg_tps = sum(m.decode_tps for m in turn_metrics) / len(turn_metrics)
                avg_ttft = sum(m.ttft_ms or 0 for m in turn_metrics) / len(turn_metrics)
                console.print(f"[cyan]{len(turn_metrics)} turns · "
                              f"avg {avg_tps:.2f} tok/s · avg TTFT {avg_ttft:.0f} ms[/]")
            continue
        if user.startswith("/save "):
            p = Path(user.split(" ", 1)[1].strip())
            p.write_text(json.dumps(history, indent=2), encoding="utf-8")
            console.print(f"[dim]saved → {p}[/]")
            continue

        history.append({"role": "user", "content": user})
        console.print("[bold cyan]localmind ›[/] ", end="")
        res = c.chat(history, cfg,
                     on_token=lambda t: (sys.stdout.write(t), sys.stdout.flush()))
        console.print()
        if not res.ok:
            console.print(f"[red]error:[/] {res.metrics.error}")
            history.pop()
            continue
        history.append({"role": "assistant", "content": res.text})
        turn_metrics.append(res.metrics)
        m = res.metrics
        console.print(f"[dim]{m.ttft_ms:.0f} ms to first token · {m.decode_tps:.1f} tok/s · "
                      f"{m.completion_tokens} tokens · {m.wall_ms/1000:.1f} s[/]"
                      if m.ttft_ms else "")


@app.command()
def schemas() -> None:
    """List the built-in Pydantic output schemas."""
    t = Table(title="available schemas")
    t.add_column("name", style="bold cyan")
    t.add_column("fields")
    for name, model in SCHEMAS.items():
        t.add_row(name, ", ".join(model.model_fields.keys()))
    console.print(t)


@app.command()
def structured(
    prompt: str,
    schema: str = typer.Option("answer", "--schema", "-s", help="see `localmind schemas`"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    host: str = typer.Option(DEFAULT_HOST, "--host"),
    temperature: float = typer.Option(0.0, "--temperature", "-t"),
    max_attempts: int = typer.Option(3, "--attempts", "-a"),
    no_native: bool = typer.Option(False, "--no-native",
                                   help="skip Ollama's schema mode; prompt-only + repair"),
    show_attempts: bool = typer.Option(False, "--show-attempts"),
) -> None:
    """Phase 2 — force the model into a validated JSON object, with repair retries."""
    c = _client(host)
    try:
        model_cls = get_schema(schema)
    except KeyError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)

    cfg = _cfg(model, host, temperature, None, 640, DEFAULT_SYSTEM_PROMPT)
    gen = StructuredGenerator(c, max_attempts=max_attempts, use_native_format=not no_native)
    with console.status(f"generating {schema} …"):
        result = gen.generate(prompt, model_cls, cfg, schema)

    if show_attempts or not result.ok:
        for a in result.attempts:
            status = "[green]valid[/]" if a.error is None else "[red]rejected[/]"
            console.print(Panel(
                (a.raw[:800] or "[dim]empty[/]") +
                (f"\n\n[red]{a.error}[/]" if a.error else ""),
                title=f"attempt {a.index} · {a.strategy} · {status}",
                border_style="green" if a.error is None else "red"))

    if result.ok:
        console.print(RichJSON(json.dumps(result.data, indent=2)))
        console.print(f"[dim]validated against {model_cls.__name__} in "
                      f"{result.n_attempts} attempt(s) · {result.total_ms:.0f} ms[/]")
    else:
        console.print(Panel(json.dumps(result.envelope(), indent=2)[:1200],
                            title="graceful failure envelope", border_style="red"))
        raise typer.Exit(2)


@app.command()
def bench(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    host: str = typer.Option(DEFAULT_HOST, "--host"),
    repeats: int = typer.Option(3, "--repeats", "-r"),
    limit: int = typer.Option(None, "--limit", "-l", help="use only the first N prompts"),
    category: list[str] = typer.Option(None, "--category", "-c"),
    temperature: float = typer.Option(0.0, "--temperature", "-t"),
    seed: int = typer.Option(42, "--seed"),
    no_cold: bool = typer.Option(False, "--no-cold-start"),
    report: bool = typer.Option(True, "--report/--no-report"),
) -> None:
    """Phase 1 — measure TTFT, tokens/sec, total latency and memory, rigorously."""
    ensure_dirs()
    c = _client(host)
    if not c.has_model(model):
        console.print(f"[red]{model} is not installed.[/] Run: ollama pull {model}")
        raise typer.Exit(1)

    ts = load_testset().filter(category or None, limit)
    cfg = _cfg(model, host, temperature, seed, 512, DEFAULT_SYSTEM_PROMPT)
    console.print(f"[bold]benchmarking[/] {model} · {len(ts)} prompts × {repeats} repeats "
                  f"= {len(ts)*repeats} generations\n")

    with _progress() as prog:
        task = prog.add_task("running", total=len(ts) * repeats + (0 if no_cold else 1))
        def cb(label, done, total):
            prog.update(task, completed=done, description=f"running · {label}")
        result = run_benchmark(c, cfg, ts, repeats=repeats, measure_cold=not no_cold,
                               on_progress=cb)

    s = result.summary
    t = Table(title=f"{model} — steady state (n={s.get('runs')})")
    for col in ("metric", "median", "mean", "p95", "min", "max"):
        t.add_column(col)
    for label, key, nd in (("TTFT (ms)", "ttft_ms", 0), ("decode tok/s", "decode_tps", 2),
                           ("prefill tok/s", "prompt_tps", 1), ("total (ms)", "total_ms", 0),
                           ("peak RSS (MB)", "peak_rss_mb", 0)):
        d = s.get(key, {})
        if not d.get("n"):
            continue
        t.add_row(label, *[f"{d.get(k):.{nd}f}" if d.get(k) is not None else "—"
                           for k in ("median", "mean", "p95", "min", "max")])
    console.print(t)
    if result.cold_start:
        cs = result.cold_start
        console.print(f"[dim]cold start: {cs.get('load_ms', 0):.0f} ms weight load, "
                      f"{cs.get('total_ms', 0):.0f} ms total[/]")

    path = save_benchmark(result)
    console.print(f"\n[green]✓[/] results → [bold]{path}[/]")
    if report:
        out = write_report(path)
        console.print(f"[green]✓[/] report  → [bold]{out}[/]")


@app.command("temp-study")
def temp_study(
    prompt: str = typer.Option(
        "Extract the intent, device and value from: 'set the bedroom lamp to 40 percent'.",
        "--prompt", "-p"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    host: str = typer.Option(DEFAULT_HOST, "--host"),
    temps: str = typer.Option("0,0.3,0.7,1.0", "--temps",
                              help="comma-separated temperatures"),
    samples: int = typer.Option(8, "--samples", "-n"),
    schema: str = typer.Option("command", "--schema", "-s",
                               help="schema name, or 'none' for free text"),
    fixed_seed: bool = typer.Option(False, "--fixed-seed",
                                    help="pin the seed to show reproducibility"),
    report: bool = typer.Option(True, "--report/--no-report"),
) -> None:
    """Phase 2 — quantify how temperature changes output variance and schema validity."""
    ensure_dirs()
    c = _client(host)
    temperatures = [float(x) for x in temps.split(",") if x.strip()]
    schema_name = None if schema.lower() in ("none", "") else schema
    cfg = _cfg(model, host, 0.0, None, 400, DEFAULT_SYSTEM_PROMPT)

    with _progress() as prog:
        task = prog.add_task("sampling", total=len(temperatures) * samples)
        def cb(label, done, total):
            prog.update(task, completed=done, description=f"sampling · {label}")
        payload = run_temperature_study(c, cfg, prompt, temperatures, samples,
                                        schema_name, fixed_seed, on_progress=cb)

    t = Table(title=f"temperature sweep · {model}"
                    + (f" · schema={schema_name}" if schema_name else ""))
    cols = ["T", "unique/N", "dup rate", "word overlap", "len CV", "tok/s"]
    if schema_name:
        cols += ["schema valid", "answer stable"]
    for col in cols:
        t.add_column(col)
    for p in payload["points"]:
        row = [str(p["temperature"]), f"{p['unique_outputs']}/{p['n_samples']}",
               f"{p['exact_duplicate_rate']:.0%}", f"{p['mean_pairwise_jaccard']:.3f}",
               f"{p['length_cv']:.3f}", f"{p['decode_tps_mean']:.1f}"]
        if schema_name:
            row += [f"{p.get('schema_valid_rate', 0):.0%}",
                    f"{p.get('answer_stability', 0):.0%}" if p.get("answer_stability") else "—"]
        t.add_row(*row)
    console.print(t)
    console.print(Panel(payload["conclusion"], title="conclusion", border_style="cyan"))

    path = save_temperature_study(payload)
    console.print(f"\n[green]✓[/] results → [bold]{path}[/]")
    if report:
        console.print(f"[green]✓[/] report  → [bold]{write_report(path)}[/]")


@app.command()
def compare(
    model: list[str] = typer.Option(None, "--model", "-m",
                                    help="repeatable; defaults to the standard roster"),
    preset: str = typer.Option(None, "--preset",
                               help="'default' (3 architectures) or 'quant' (Q4/Q5/Q8)"),
    host: str = typer.Option(DEFAULT_HOST, "--host"),
    repeats: int = typer.Option(1, "--repeats", "-r"),
    limit: int = typer.Option(None, "--limit", "-l"),
    category: list[str] = typer.Option(None, "--category", "-c"),
    temperature: float = typer.Option(0.0, "--temperature", "-t"),
    seed: int = typer.Option(42, "--seed"),
    judge: str = typer.Option(None, "--judge", help="model to use as an LLM judge"),
    report: bool = typer.Option(True, "--report/--no-report"),
) -> None:
    """Phase 3 — run every model over the same test set on the same machine."""
    ensure_dirs()
    c = _client(host)
    models_ = list(model) if model else (
        DEFAULT_QUANT_MODELS if preset == "quant" else DEFAULT_COMPARE_MODELS)

    missing = [m for m in models_ if not c.has_model(m)]
    if missing:
        console.print(Panel(
            "Not installed:\n" + "\n".join(f"  ollama pull {m}" for m in missing),
            title="missing models", border_style="yellow"))
        if len(missing) == len(models_):
            raise typer.Exit(1)

    ts = load_testset().filter(category or None, limit)
    cfg = _cfg(models_[0], host, temperature, seed, 512, DEFAULT_SYSTEM_PROMPT)
    console.print(f"[bold]comparing[/] {', '.join(models_)}\n"
                  f"{len(ts)} prompts × {repeats} repeats × {len(models_)} models = "
                  f"{len(ts)*repeats*len(models_)} generations\n")

    with _progress() as prog:
        task = prog.add_task("evaluating", total=len(ts) * repeats * len(models_))
        state = {"done": 0}
        def cb(mdl, pid, done, total):
            state["done"] += 1
            prog.update(task, completed=state["done"], description=f"{mdl} · {pid}")
        payload = run_comparison(c, models_, ts, cfg, repeats, judge, on_progress=cb)

    t = Table(title="headline comparison")
    for col in ("model", "params", "quant", "disk MB", "TTFT ms", "tok/s",
                "total ms", "peak RSS", "quality"):
        t.add_column(col)
    for h in payload["headlines"]:
        def f(v, nd=0):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"
        t.add_row(h["model"], h.get("params") or "—", h.get("quant") or "—",
                  f(h.get("disk_mb")), f(h.get("ttft_ms_median")),
                  f(h.get("decode_tps_median"), 2), f(h.get("total_ms_median")),
                  f(h.get("peak_rss_mb")), f(h.get("quality"), 1))
    console.print(t)

    if payload.get("findings"):
        console.print(Panel("\n".join(f"• {x}" for x in payload["findings"]),
                            title="findings", border_style="cyan"))

    path = save_comparison(payload)
    console.print(f"\n[green]✓[/] results → [bold]{path}[/]")
    if report:
        console.print(f"[green]✓[/] report  → [bold]{write_report(path)}[/]")


@app.command()
def report(
    results: str = typer.Argument(None, help="path to a results JSON; omit to use the latest"),
    kind: str = typer.Option("comparison", "--kind",
                             help="benchmark | tempstudy | comparison (when omitting the path)"),
) -> None:
    """Render a saved result file into a Markdown technical report."""
    path = Path(results) if results else latest_run(kind)
    if not path or not Path(path).exists():
        console.print(f"[red]no results found[/] for kind '{kind}'. Run a study first.")
        raise typer.Exit(1)
    out = write_report(path)
    console.print(f"[green]✓[/] {out}")


@app.command()
def testset(
    show: bool = typer.Option(False, "--show", help="print every prompt"),
) -> None:
    """Inspect the standardised evaluation prompts."""
    ts = load_testset()
    console.print(f"[bold]{ts.name}[/] v{ts.version} — {len(ts)} prompts\n{ts.description}\n")
    t = Table(title="categories")
    t.add_column("category")
    t.add_column("count", justify="right")
    for c in ts.categories():
        t.add_row(c, str(sum(1 for p in ts.prompts if p.category == c)))
    console.print(t)
    if show:
        for p in ts.prompts:
            console.print(f"[cyan]{p.id}[/] [dim]({p.category})[/] {p.prompt[:160]}")


@app.command()
def serve(
    host_bind: str = typer.Option("127.0.0.1", "--bind"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Run the FastAPI wrapper (OpenAPI docs at /docs)."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]pip install 'fastapi' 'uvicorn'[/]")
        raise typer.Exit(1)
    console.print(f"[green]serving[/] http://{host_bind}:{port}/docs")
    uvicorn.run("api.server:app", host=host_bind, port=port, reload=False)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
