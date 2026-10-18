"""Phase 2: how much does temperature actually change the output?

"Temperature controls randomness" is a slogan. This module turns it into
numbers you can put in a report:

* **exact_duplicate_rate** — fraction of samples byte-identical to another sample
* **unique_outputs**       — how many distinct answers N samples produced
* **mean_pairwise_jaccard**— word-set overlap between every pair (1.0 = identical)
* **length_cv**            — coefficient of variation of output length
* **schema_valid_rate**    — with a schema, does randomness break structure?
* **semantic_drift**       — does the *answer* change, or just the wording?

The last one matters most in production: temperature 0.7 that rephrases is
harmless; temperature 0.7 that changes the extracted value is a bug.
"""
from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field

from pydantic import BaseModel

from .client import OllamaClient
from .config import RunConfig
from .metrics import save_run
from .quality import words
from .schemas import get_schema
from .structured import StructuredGenerator


def jaccard(a: str, b: str) -> float:
    wa, wb = set(words(a)), set(words(b))
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def mean_pairwise_jaccard(texts: list[str]) -> float:
    pairs = list(itertools.combinations(texts, 2))
    if not pairs:
        return 1.0
    return round(statistics.fmean(jaccard(a, b) for a, b in pairs), 4)


@dataclass
class TemperaturePoint:
    temperature: float
    samples: list[str] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    schema_valid: list[bool] = field(default_factory=list)
    parsed_values: list[str] = field(default_factory=list)

    def stats(self) -> dict:
        n = len(self.samples)
        uniq = len({s.strip() for s in self.samples})
        lengths = [len(words(s)) for s in self.samples]
        mean_len = statistics.fmean(lengths) if lengths else 0
        out = {
            "temperature": self.temperature,
            "n_samples": n,
            "unique_outputs": uniq,
            "exact_duplicate_rate": round(1 - (uniq / n), 4) if n else 0.0,
            "mean_pairwise_jaccard": mean_pairwise_jaccard(self.samples),
            "length_mean_words": round(mean_len, 1),
            "length_stdev": round(statistics.pstdev(lengths), 2) if n > 1 else 0.0,
            "length_cv": round(statistics.pstdev(lengths) / mean_len, 4)
            if n > 1 and mean_len else 0.0,
            "decode_tps_mean": round(
                statistics.fmean([m["decode_tps"] for m in self.metrics if m.get("ok")]), 2
            ) if self.metrics else 0.0,
        }
        if self.schema_valid:
            out["schema_valid_rate"] = round(sum(self.schema_valid) / len(self.schema_valid), 4)
        if self.parsed_values:
            out["distinct_answers"] = len(set(self.parsed_values))
            out["answer_stability"] = round(
                max(self.parsed_values.count(v) for v in set(self.parsed_values))
                / len(self.parsed_values), 4
            )
        return out


def run_temperature_study(
    client: OllamaClient,
    cfg: RunConfig,
    prompt: str,
    temperatures: list[float],
    samples: int = 8,
    schema_name: str | None = None,
    fixed_seed: bool = False,
    on_progress=None,
) -> dict:
    """Run the same prompt N times at each temperature and quantify the spread.

    `fixed_seed=True` pins the seed so you can demonstrate the other half of
    the story: with a fixed seed, even temperature 0.7 becomes reproducible.
    """
    schema: type[BaseModel] | None = get_schema(schema_name) if schema_name else None
    gen = StructuredGenerator(client, max_attempts=2) if schema else None

    points: list[TemperaturePoint] = []
    total = len(temperatures) * samples
    done = 0

    for temp in temperatures:
        point = TemperaturePoint(temperature=temp)
        for i in range(samples):
            run_cfg = RunConfig(**{
                **cfg.__dict__, "temperature": temp,
                "seed": (1234 if fixed_seed else None),
            })
            if schema and gen:
                sr = gen.generate(prompt, schema, run_cfg, schema_name)
                text = sr.attempts[-1].raw if sr.attempts else ""
                point.samples.append(text)
                point.metrics.append(sr.attempts[-1].metrics if sr.attempts else {"ok": False})
                point.schema_valid.append(sr.ok)
                if sr.ok and sr.data:
                    # Track the *first* scalar field: does the answer itself move?
                    first = next(
                        (str(v) for v in sr.data.values()
                         if isinstance(v, (str, int, float, bool))), ""
                    )
                    point.parsed_values.append(first)
            else:
                res = client.generate(prompt, run_cfg)
                point.samples.append(res.text)
                point.metrics.append(res.metrics.to_dict())
            done += 1
            if on_progress:
                on_progress(f"T={temp}", done, total)
        points.append(point)

    stats = [p.stats() for p in points]
    return {
        "prompt": prompt,
        "model": cfg.model,
        "schema": schema_name,
        "fixed_seed": fixed_seed,
        "samples_per_temperature": samples,
        "points": stats,
        "raw_samples": {
            str(p.temperature): [s[:600] for s in p.samples] for p in points
        },
        "conclusion": _conclude(stats),
    }


def _conclude(stats: list[dict]) -> str:
    """A one-paragraph, data-derived takeaway so the report writes itself."""
    if not stats:
        return "No data."
    lo, hi = stats[0], stats[-1]
    parts = [
        (
        f"At T={lo['temperature']}, {lo['unique_outputs']}/{lo['n_samples']} outputs were "
        f"distinct (pairwise word overlap {lo['mean_pairwise_jaccard']}). "
        f"At T={hi['temperature']}, {hi['unique_outputs']}/{hi['n_samples']} were distinct "
        f"(overlap {hi['mean_pairwise_jaccard']})."
        )
    ]
    if lo.get("schema_valid_rate") is not None and hi.get("schema_valid_rate") is not None:
        parts.append(
            f" Schema validity moved from {lo['schema_valid_rate']:.0%} to "
            f"{hi['schema_valid_rate']:.0%}."
        )
    if lo.get("answer_stability") and hi.get("answer_stability"):
        parts.append(
            f" The extracted answer itself was stable {lo['answer_stability']:.0%} of the "
            f"time at T={lo['temperature']} versus {hi['answer_stability']:.0%} at "
            f"T={hi['temperature']} — the number to watch if downstream code trusts it."
        )
    return "".join(parts)


def save_temperature_study(payload: dict) -> str:
    tag = payload["model"].replace(":", "_")
    return str(save_run("tempstudy", payload, name=f"tempstudy-{tag}"))
