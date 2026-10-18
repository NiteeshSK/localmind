"""Phase 2: the contracts a small model must satisfy.

A 3B model is perfectly capable of producing valid JSON — it just isn't
*reliable* about it. Pydantic turns "usually valid" into a hard boundary:
either the object validates, or the pipeline retries, or it fails loudly.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Answer(BaseModel):
    """General-purpose Q&A envelope."""

    answer: str = Field(..., description="Direct answer, 1-3 sentences.")
    key_points: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_more_info: bool = False

    @field_validator("answer")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer must not be empty")
        return v.strip()


class Summary(BaseModel):
    title: str = Field(..., max_length=120)
    summary: str
    bullet_points: list[str] = Field(..., min_length=2, max_length=6)
    topics: list[str] = Field(default_factory=list, max_length=5)


class Classification(BaseModel):
    label: Literal["positive", "negative", "neutral", "mixed"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., max_length=300)


class Entities(BaseModel):
    people: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)


class Step(BaseModel):
    step_number: int = Field(..., ge=1)
    action: str
    tool: str | None = None


class ActionPlan(BaseModel):
    goal: str
    steps: list[Step] = Field(..., min_length=1, max_length=8)
    estimated_minutes: int = Field(..., ge=1, le=600)


class CommandIntent(BaseModel):
    """The classic edge-device use case: natural language -> device command."""

    intent: Literal["turn_on", "turn_off", "set_value", "query_status", "unknown"]
    device: str
    value: float | None = None
    unit: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)


SCHEMAS: dict[str, type[BaseModel]] = {
    "answer": Answer,
    "summary": Summary,
    "classification": Classification,
    "entities": Entities,
    "plan": ActionPlan,
    "command": CommandIntent,
}


def get_schema(name: str) -> type[BaseModel]:
    try:
        return SCHEMAS[name]
    except KeyError:
        raise KeyError(f"unknown schema '{name}'. Available: {', '.join(SCHEMAS)}") from None


def schema_instructions(model: type[BaseModel]) -> str:
    """The system-prompt fragment that tells the SLM exactly what shape to emit."""
    import json

    return (
        "You must reply with a single JSON object and nothing else. "
        "No prose, no markdown fences, no explanation before or after.\n"
        "The object must validate against this JSON Schema:\n"
        f"{json.dumps(model.model_json_schema(), indent=2)}"
    )
