"""Phase 3: output quality scoring.

Two independent scorers, deliberately:

1. `HeuristicScorer` — deterministic, free, reproducible. Checks keyword
   coverage, instruction compliance (word caps, banned words, bullet counts,
   JSON validity) and degeneracy (repetition loops, empty output).
2. `LLMJudge`      — a second local model grades relevance/accuracy/clarity.
   More nuanced, but slower and itself a small model, so it is opt-in and its
   scores are reported *separately*, never silently averaged in.

Never claim a benchmark measures "quality" without saying how. This file is
the "how".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .client import OllamaClient
from .config import RunConfig
from .structured import parse_json
from .testset import PromptSpec

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_REFUSAL_HINTS = (
    "i cannot", "i can't help", "i'm unable", "as an ai language model",
    "i do not have the ability",
)


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def repetition_score(text: str) -> float:
    """1.0 = no degenerate looping, 0.0 = the model got stuck in a loop."""
    w = words(text)
    if len(w) < 12:
        return 1.0
    trigrams = [tuple(w[i : i + 3]) for i in range(len(w) - 2)]
    return round(len(set(trigrams)) / len(trigrams), 4)


def keyword_coverage(text: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    low = text.lower()
    hits = sum(1 for k in keywords if k.lower() in low)
    return round(hits / len(keywords), 4)


def compliance_score(text: str, checks: dict) -> tuple[float | None, list[str]]:
    """Score 0-1 on explicit instruction constraints; also return failures."""
    if not checks:
        return None, []
    passed, total, failures = 0, 0, []

    if "max_words" in checks:
        total += 1
        n = len(words(text))
        if n <= checks["max_words"]:
            passed += 1
        else:
            failures.append(f"exceeded word cap ({n} > {checks['max_words']})")

    if "min_bullets" in checks:
        total += 1
        n = len(_BULLET_RE.findall(text))
        if n == 0:  # fall back to counting non-empty lines (poems, plain lists)
            n = len([ln for ln in text.splitlines() if ln.strip()])
        if n >= checks["min_bullets"]:
            passed += 1
        else:
            failures.append(f"only {n} list items (needed {checks['min_bullets']})")

    if checks.get("must_be_json"):
        total += 1
        if parse_json(text) is not None:
            passed += 1
        else:
            failures.append("output was not parsable JSON")

    if checks.get("must_contain_code"):
        total += 1
        if "def " in text or "```" in text or "function " in text:
            passed += 1
        else:
            failures.append("no code block detected")

    for banned in checks.get("must_not_contain", []):
        total += 1
        if banned.lower() not in text.lower():
            passed += 1
        else:
            failures.append(f"used banned word '{banned}'")

    return (round(passed / total, 4) if total else None), failures


@dataclass
class QualityScore:
    prompt_id: str
    category: str
    composite: float                      # 0-100
    keyword_coverage: float | None = None
    compliance: float | None = None
    repetition: float = 1.0
    empty: bool = False
    refusal: bool = False
    failures: list[str] = field(default_factory=list)
    judge: dict | None = None             # populated only when --judge is used

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


class HeuristicScorer:
    """Weighted, deterministic scoring. Weights renormalise when a signal is absent."""

    W_KEYWORD = 0.45
    W_COMPLIANCE = 0.40
    W_REPETITION = 0.15

    def score(self, spec: PromptSpec, text: str) -> QualityScore:
        text = (text or "").strip()
        empty = len(text) < 2
        refusal = any(h in text.lower() for h in _REFUSAL_HINTS)

        cov = keyword_coverage(text, spec.expected_keywords)
        comp, failures = compliance_score(text, spec.checks)
        rep = repetition_score(text)

        parts: list[tuple[float, float]] = []
        if cov is not None:
            parts.append((cov, self.W_KEYWORD))
        if comp is not None:
            parts.append((comp, self.W_COMPLIANCE))
        parts.append((rep, self.W_REPETITION))

        wsum = sum(w for _, w in parts)
        composite = sum(v * w for v, w in parts) / wsum * 100 if wsum else 0.0

        if empty:
            composite = 0.0
            failures.append("empty output")
        if refusal:
            composite *= 0.5
            failures.append("looks like a refusal / boilerplate")

        return QualityScore(
            prompt_id=spec.id, category=spec.category, composite=round(composite, 2),
            keyword_coverage=cov, compliance=comp, repetition=rep,
            empty=empty, refusal=refusal, failures=failures,
        )


JUDGE_SYSTEM = (
    "You are a strict evaluation judge. You score another model's answer. "
    "Reply with ONLY a JSON object: "
    '{"relevance": 1-5, "accuracy": 1-5, "clarity": 1-5, "verdict": "one short sentence"}'
)


class LLMJudge:
    """Optional second opinion from a local judge model (kept fully offline)."""

    def __init__(self, client: OllamaClient, judge_model: str):
        self.client = client
        self.judge_model = judge_model

    def score(self, spec: PromptSpec, text: str) -> dict | None:
        prompt = (
            f"TASK GIVEN TO THE MODEL:\n{spec.prompt}\n\n"
            f"MODEL'S ANSWER:\n{text[:2000]}\n\n"
            "Score it now as JSON only."
        )
        cfg = RunConfig(model=self.judge_model, temperature=0.0, seed=7,
                        num_predict=200, system=JUDGE_SYSTEM)
        res = self.client.generate(prompt, cfg, fmt="json")
        data = parse_json(res.text)
        if not isinstance(data, dict):
            return None
        try:
            out = {k: float(data[k]) for k in ("relevance", "accuracy", "clarity")
                   if k in data}
        except (TypeError, ValueError):
            return None
        if not out:
            return None
        out["mean"] = round(sum(out.values()) / len(out), 3)
        out["verdict"] = str(data.get("verdict", ""))[:200]
        return out
