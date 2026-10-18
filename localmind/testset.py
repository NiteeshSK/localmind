"""Loader for the standardised evaluation prompts."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import TESTSET_PATH


@dataclass
class PromptSpec:
    id: str
    category: str
    prompt: str
    expected_keywords: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    num_predict: int = 256
    schema: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "PromptSpec":
        return cls(
            id=d["id"], category=d["category"], prompt=d["prompt"],
            expected_keywords=d.get("expected_keywords", []),
            checks=d.get("checks", {}),
            num_predict=d.get("num_predict", 256),
            schema=d.get("schema"),
        )


@dataclass
class TestSet:
    name: str
    version: str
    description: str
    prompts: list[PromptSpec]

    def __len__(self) -> int:
        return len(self.prompts)

    def categories(self) -> list[str]:
        return sorted({p.category for p in self.prompts})

    def filter(self, categories: list[str] | None = None,
               limit: int | None = None) -> "TestSet":
        ps = self.prompts
        if categories:
            wanted = {c.lower() for c in categories}
            ps = [p for p in ps if p.category.lower() in wanted]
        if limit:
            ps = ps[:limit]
        return TestSet(self.name, self.version, self.description, ps)


def load_testset(path: str | Path | None = None) -> TestSet:
    p = Path(path) if path else TESTSET_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return TestSet(
        name=data.get("name", p.stem),
        version=data.get("version", "0"),
        description=data.get("description", ""),
        prompts=[PromptSpec.from_dict(d) for d in data["prompts"]],
    )
