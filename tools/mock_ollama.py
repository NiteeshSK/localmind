"""A fake Ollama daemon, for testing LocalMind without downloading 12 GB of weights.

It speaks enough of the real API (`/api/version`, `/api/tags`, `/api/ps`,
`/api/generate`, `/api/chat`, `/api/pull`) to exercise every code path in this
repo: streaming, nanosecond timing counters, JSON mode, and native schema mode
(it synthesises a valid instance straight from the JSON Schema you send).

    python tools/mock_ollama.py --port 11500
    localmind doctor --host http://127.0.0.1:11500

⚠️  Numbers produced against this server are SIMULATED. They test the harness,
    not any model. Never put them in a report.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# name -> (tokens/sec, load seconds, disk bytes, params, quant)
PROFILES = {
    "mock-fast:1b":   (48.0, 0.35, 1_100_000_000, "1.2B", "Q4_K_M"),
    "mock-mid:3b":    (22.0, 0.80, 2_000_000_000, "3.2B", "Q4_K_M"),
    "mock-slow:7b":   (7.5, 2.10, 4_400_000_000, "7.2B", "Q4_K_M"),
    "mock-mid:3b-q8": (14.0, 1.10, 3_400_000_000, "3.2B", "Q8_0"),
}
DEFAULT_PROFILE = (20.0, 0.7, 2_000_000_000, "3B", "Q4_K_M")

FILLER = (
    "This is simulated output from the mock server. It streams token by token so "
    "the client can measure time to first token and decode throughput honestly. "
    "The content is meaningless; only the timing behaviour is realistic."
).split(" ")


def synth_from_schema(schema: dict, depth: int = 0) -> object:
    """Build a minimal valid instance of a JSON Schema (enough for our models)."""
    defs = schema.get("$defs", {})

    def resolve(s: dict) -> dict:
        if "$ref" in s:
            key = s["$ref"].split("/")[-1]
            return defs.get(key, {})
        for k in ("anyOf", "oneOf"):
            if k in s:
                for opt in s[k]:
                    if opt.get("type") != "null":
                        return resolve(opt)
        return s

    s = resolve(schema)
    t = s.get("type")
    if "enum" in s:
        return s["enum"][0]
    if "const" in s:
        return s["const"]
    if t == "object" or "properties" in s:
        out = {}
        for name, sub in (s.get("properties") or {}).items():
            sub = dict(sub)
            sub.setdefault("$defs", defs)
            out[name] = synth_from_schema(sub, depth + 1)
        return out
    if t == "array":
        item = dict(s.get("items") or {"type": "string"})
        item.setdefault("$defs", defs)
        n = max(int(s.get("minItems", 2)), 2)
        n = min(n, int(s.get("maxItems", n)))
        return [synth_from_schema(item, depth + 1) for _ in range(n)]
    if t == "integer":
        lo = int(s.get("minimum", s.get("exclusiveMinimum", 0) or 0)) or 1
        return max(lo, 1)
    if t == "number":
        lo, hi = s.get("minimum", 0.0), s.get("maximum", 1.0)
        return round((lo + hi) / 2, 2)
    if t == "boolean":
        return False
    max_len = s.get("maxLength", 40)
    return "mock value"[:max_len] if max_len else "mock value"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep the test output clean
        pass

    # ------------------------------------------------------------------ #
    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/api/version"):
            return self._json({"version": "0.0.0-mock"})
        if self.path.startswith("/api/tags"):
            return self._json({"models": [
                {"name": n, "size": p[2],
                 "details": {"parameter_size": p[3], "quantization_level": p[4],
                             "family": "mock"}}
                for n, p in PROFILES.items()]})
        if self.path.startswith("/api/ps"):
            return self._json({"models": []})
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path.startswith("/api/pull"):
            return self._json({"status": "success"})
        if self.path.startswith(("/api/generate", "/api/chat")):
            return self._generate(req, chat=self.path.startswith("/api/chat"))
        self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------------ #
    def _generate(self, req: dict, chat: bool) -> None:
        model = req.get("model", "mock-mid:3b")
        tps, load_s, *_ = PROFILES.get(model, DEFAULT_PROFILE)
        opts = req.get("options") or {}
        seed = opts.get("seed")
        rng = random.Random(seed if seed is not None else time.time_ns())
        temp = float(opts.get("temperature", 0.0))
        limit = int(opts.get("num_predict", 128))
        fmt = req.get("format")

        # Empty prompt with keep_alive 0 == unload request.
        if not req.get("prompt") and not req.get("messages"):
            return self._json({"model": model, "done": True, "response": "",
                               "total_duration": 1_000_000})

        if isinstance(fmt, dict):
            text = json.dumps(synth_from_schema(fmt))
        elif fmt == "json":
            text = json.dumps({"answer": "mock", "confidence": 0.5,
                               "key_points": ["a", "b"], "needs_more_info": False})
        else:
            n_words = max(8, int(rng.gauss(40, 6 + 25 * temp)))
            words = [rng.choice(FILLER) if temp > 0 else FILLER[i % len(FILLER)]
                     for i in range(n_words)]
            text = " ".join(words) + "."

        tokens = [w + " " for w in text.split()][:limit]
        delay = 1.0 / tps

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        time.sleep(load_s * 0.15)  # simulated prefill
        t0 = time.perf_counter()
        for tok in tokens:
            time.sleep(delay)
            self._chunk({"model": model, "done": False,
                         **({"message": {"role": "assistant", "content": tok}}
                            if chat else {"response": tok})})
        eval_ns = int((time.perf_counter() - t0) * 1e9)
        prompt_tokens = max(1, len(str(req.get("prompt", "")).split()))
        self._chunk({
            "model": model, "done": True, "done_reason": "stop",
            **({"message": {"role": "assistant", "content": ""}} if chat else {"response": ""}),
            "total_duration": eval_ns + int(load_s * 0.15 * 1e9),
            "load_duration": int(load_s * 0.15 * 1e9),
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": int(prompt_tokens / 250 * 1e9),
            "eval_count": len(tokens),
            "eval_duration": eval_ns,
        })
        self.wfile.write(b"0\r\n\r\n")

    def _chunk(self, obj: dict) -> None:
        data = (json.dumps(obj) + "\n").encode()
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock ollama on http://{args.host}:{args.port}  models: {', '.join(PROFILES)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
