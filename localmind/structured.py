"""Phase 2: schema-enforced generation with repair-retries and graceful failure.

The retry ladder, in order of how much we nag the model:

  attempt 1  native structured output  (Ollama `format: <json-schema>`)
  attempt 2  JSON mode                 (`format: "json"`) + schema in prompt
  attempt 3+ repair prompt             (show it the bad output + the validator errors)
  exhausted  FailureEnvelope           (never raises into the caller's face)

Temperature is forced to 0 on repair attempts: if the first sample was
malformed, more randomness is not the fix.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from .client import OllamaClient
from .config import RunConfig
from .schemas import schema_instructions

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str | None:
    """Pull the first plausible JSON object out of messy model output.

    Handles: clean JSON, fenced JSON, JSON with a chatty preamble, and
    JSON followed by "Hope this helps!". Brace-counting is string-aware so
    a `}` inside a quoted value doesn't end the scan early.
    """
    if not text:
        return None
    text = text.strip()

    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        return None

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json(text: str) -> Any | None:
    blob = extract_json(text)
    if blob is None:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # Last-ditch: strip trailing commas, a very common SLM slip.
        try:
            return json.loads(re.sub(r",(\s*[}\]])", r"\1", blob))
        except json.JSONDecodeError:
            return None


@dataclass
class Attempt:
    index: int
    strategy: str
    raw: str
    error: str | None
    metrics: dict


@dataclass
class StructuredResult:
    ok: bool
    schema: str
    data: dict | None = None
    model_obj: BaseModel | None = None
    attempts: list[Attempt] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    def envelope(self) -> dict:
        """What a caller gets even when everything failed — never an exception."""
        if self.ok and self.data is not None:
            return {"ok": True, "schema": self.schema, "data": self.data,
                    "attempts": self.n_attempts}
        return {
            "ok": False,
            "schema": self.schema,
            "error": "schema_validation_failed",
            "attempts": self.n_attempts,
            "last_error": self.attempts[-1].error if self.attempts else "no attempts",
            "last_raw": (self.attempts[-1].raw[:500] if self.attempts else ""),
        }


def _errors_to_text(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors()[:6]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"- field `{loc}`: {err['msg']}")
    return "\n".join(lines)


class StructuredGenerator:
    """Turns an unreliable text generator into a reliable object generator."""

    def __init__(self, client: OllamaClient, max_attempts: int = 3,
                 use_native_format: bool = True):
        self.client = client
        self.max_attempts = max(1, max_attempts)
        self.use_native_format = use_native_format

    def generate(self, prompt: str, schema: type[BaseModel], cfg: RunConfig,
                 schema_name: str | None = None) -> StructuredResult:
        name = schema_name or schema.__name__.lower()
        result = StructuredResult(ok=False, schema=name)
        instructions = schema_instructions(schema)
        last_raw, last_err = "", None

        for i in range(1, self.max_attempts + 1):
            strategy, fmt, user_prompt, run_cfg = self._plan(
                i, prompt, instructions, schema, cfg, last_raw, last_err
            )
            res = self.client.generate(user_prompt, run_cfg, fmt=fmt)
            result.total_ms += res.metrics.wall_ms

            if not res.metrics.ok:
                last_err = res.metrics.error or "generation failed"
                result.attempts.append(Attempt(i, strategy, res.text, last_err,
                                               res.metrics.to_dict()))
                continue

            last_raw = res.text
            payload = parse_json(res.text)
            if payload is None:
                last_err = "no parsable JSON object found in output"
                result.attempts.append(Attempt(i, strategy, res.text, last_err,
                                               res.metrics.to_dict()))
                continue

            try:
                obj = schema.model_validate(payload)
            except ValidationError as ve:
                last_err = _errors_to_text(ve)
                result.attempts.append(Attempt(i, strategy, res.text, last_err,
                                               res.metrics.to_dict()))
                continue

            result.attempts.append(Attempt(i, strategy, res.text, None,
                                           res.metrics.to_dict()))
            result.ok = True
            result.model_obj = obj
            result.data = obj.model_dump()
            return result

        return result

    def _plan(self, attempt: int, prompt: str, instructions: str,
              schema: type[BaseModel], cfg: RunConfig,
              last_raw: str, last_err: str | None):
        """Choose the strategy for this attempt."""
        base_system = (cfg.system or "").strip()

        if attempt == 1 and self.use_native_format:
            cfg2 = replace_cfg(cfg, system=f"{base_system}\n\n{instructions}".strip())
            # Ollama >= 0.5 accepts a raw JSON Schema as `format`.
            return "native_schema", schema.model_json_schema(), prompt, cfg2

        if attempt == 2 or (attempt == 1 and not self.use_native_format):
            cfg2 = replace_cfg(cfg, system=f"{base_system}\n\n{instructions}".strip())
            return "json_mode", "json", prompt, cfg2

        repair = (
            "Your previous reply was rejected by a strict JSON Schema validator.\n\n"
            f"--- YOUR PREVIOUS REPLY ---\n{last_raw[:1500]}\n--- END ---\n\n"
            f"Validation errors:\n{last_err}\n\n"
            "Rewrite it so it validates. Output ONLY the corrected JSON object.\n\n"
            f"Original request: {prompt}"
        )
        cfg2 = replace_cfg(
            cfg,
            system=f"{base_system}\n\n{instructions}".strip(),
            temperature=0.0,   # determinism when repairing, always
        )
        return "repair", "json", repair, cfg2


def replace_cfg(cfg: RunConfig, **overrides) -> RunConfig:
    """Shallow copy of a RunConfig with fields overridden."""
    from dataclasses import replace

    return replace(cfg, **overrides)
