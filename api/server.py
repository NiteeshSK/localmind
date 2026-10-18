"""FastAPI wrapper around LocalMind — the same engine, over HTTP.

Everything stays on the machine: this binds to localhost by default and talks
only to the local Ollama daemon. Useful for wiring the assistant into a
desktop app, a browser extension, or a Raspberry Pi kiosk.

Run with:  localmind serve   (or)   uvicorn api.server:app --reload
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from localmind import __version__
from localmind.client import OllamaClient
from localmind.config import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, RunConfig
from localmind.schemas import SCHEMAS, get_schema
from localmind.structured import StructuredGenerator
from localmind.sysinfo import hardware_snapshot

app = FastAPI(
    title="LocalMind",
    version=__version__,
    description="Offline SLM assistant with first-class latency metrics.",
)
client = OllamaClient(DEFAULT_HOST)


class GenerateRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=8192)
    seed: int | None = None
    system: str | None = DEFAULT_SYSTEM_PROMPT
    stream: bool = False


class StructuredRequest(GenerateRequest):
    schema_name: Literal["answer", "summary", "classification",
                         "entities", "plan", "command"] = "answer"
    max_attempts: int = Field(3, ge=1, le=6)


class GenerateResponse(BaseModel):
    text: str
    metrics: dict[str, Any]


def _cfg(req: GenerateRequest) -> RunConfig:
    return RunConfig(model=req.model, temperature=req.temperature,
                     num_predict=req.max_tokens, seed=req.seed, system=req.system)


@app.get("/health")
def health() -> dict:
    up = client.is_up()
    return {"status": "ok" if up else "degraded", "ollama": up,
            "ollama_version": client.version(), "localmind": __version__}


@app.get("/hardware")
def hardware() -> dict:
    return hardware_snapshot()


@app.get("/models")
def models() -> dict:
    return {"installed": client.list_models(), "resident": client.running()}


@app.get("/schemas")
def schemas() -> dict:
    return {name: m.model_json_schema() for name, m in SCHEMAS.items()}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.stream:
        raise HTTPException(400, "use /generate/stream for streaming")
    res = client.generate(req.prompt, _cfg(req), sample_memory=True)
    if not res.ok:
        raise HTTPException(502, res.metrics.error or "generation failed")
    return GenerateResponse(text=res.text, metrics=res.metrics.to_dict())


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    """Server-sent-events style token stream; the final event carries metrics."""
    import json
    import queue
    import threading

    q: "queue.Queue[str | None]" = queue.Queue()

    def worker():
        res = client.generate(req.prompt, _cfg(req), on_token=lambda t: q.put(t))
        q.put("\n__METRICS__" + json.dumps(res.metrics.to_dict()))
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(gen(), media_type="text/plain")


@app.post("/structured")
def structured(req: StructuredRequest) -> dict:
    schema = get_schema(req.schema_name)
    gen = StructuredGenerator(client, max_attempts=req.max_attempts)
    result = gen.generate(req.prompt, schema, _cfg(req), req.schema_name)
    body = result.envelope()
    body["total_ms"] = round(result.total_ms, 1)
    if not result.ok:
        # 422: the model answered, but never satisfied the contract.
        raise HTTPException(422, body)
    return body
