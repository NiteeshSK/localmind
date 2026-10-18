"""End-to-end tests against the mock Ollama daemon.

These exercise the real HTTP paths — streaming, timing counters, schema mode,
retry ladder, benchmark harness, comparison harness and report rendering —
without needing a single gigabyte of downloaded weights.
"""
import pytest

from localmind.benchmark import run_benchmark
from localmind.client import OllamaClient
from localmind.compare import run_comparison
from localmind.config import RunConfig
from localmind.report import render_benchmark, render_comparison, render_tempstudy
from localmind.schemas import Answer, CommandIntent
from localmind.structured import StructuredGenerator
from localmind.tempstudy import run_temperature_study
from localmind.testset import load_testset

MODEL = "mock-fast:1b"


@pytest.fixture
def client(mock_host):
    return OllamaClient(mock_host)


class TestClientContract:
    def test_daemon_is_reachable(self, client):
        assert client.is_up() and client.version() == "0.0.0-mock"

    def test_lists_and_finds_models(self, client):
        assert client.has_model(MODEL)
        assert not client.has_model("not-a-real-model:99b")

    def test_streaming_populates_every_metric(self, client):
        res = client.generate("hello", RunConfig(model=MODEL, num_predict=24))
        m = res.metrics
        assert m.ok and res.text
        assert m.ttft_ms is not None and m.ttft_ms > 0
        assert m.completion_tokens > 0
        assert m.decode_tps > 0
        assert m.wall_ms >= m.ttft_ms          # TTFT is part of the total
        assert m.end_to_end_tps <= m.decode_tps + 1e-6   # prefill drags e2e down

    def test_tokens_arrive_incrementally(self, client):
        seen = []
        client.generate("hello", RunConfig(model=MODEL, num_predict=20),
                        on_token=seen.append)
        assert len(seen) > 3, "streaming should deliver many chunks, not one blob"

    def test_chat_endpoint_works(self, client):
        res = client.chat([{"role": "user", "content": "hi"}],
                          RunConfig(model=MODEL, num_predict=16))
        assert res.ok and res.text

    def test_unreachable_host_fails_softly(self):
        bad = OllamaClient("http://127.0.0.1:1")
        assert not bad.is_up()
        res = bad.generate("hi", RunConfig(model=MODEL))
        assert not res.ok and res.metrics.error   # no exception escapes

    def test_num_predict_is_respected(self, client):
        res = client.generate("hello", RunConfig(model=MODEL, num_predict=5))
        assert res.metrics.completion_tokens <= 5


class TestStructuredPipeline:
    def test_native_schema_mode_validates_first_try(self, client):
        gen = StructuredGenerator(client, max_attempts=3)
        r = gen.generate("set the lamp to 40%", CommandIntent,
                         RunConfig(model=MODEL), "command")
        assert r.ok and r.n_attempts == 1
        assert r.data["intent"] in {"turn_on", "turn_off", "set_value",
                                    "query_status", "unknown"}

    def test_retry_ladder_gives_up_gracefully(self, client):
        # JSON mode on the mock returns an Answer-shaped object, which can never
        # satisfy CommandIntent — so this must exhaust attempts, not raise.
        gen = StructuredGenerator(client, max_attempts=3, use_native_format=False)
        r = gen.generate("set the lamp", CommandIntent, RunConfig(model=MODEL), "command")
        assert not r.ok
        assert r.n_attempts == 3
        env = r.envelope()
        assert env["ok"] is False and env["error"] == "schema_validation_failed"
        assert env["last_error"]

    def test_repair_attempt_forces_temperature_zero(self, client):
        gen = StructuredGenerator(client, max_attempts=3, use_native_format=False)
        r = gen.generate("x", CommandIntent,
                         RunConfig(model=MODEL, temperature=0.9), "command")
        repair = [a for a in r.attempts if a.strategy == "repair"]
        assert repair and repair[0].metrics["temperature"] == 0.0

    def test_json_mode_satisfies_a_matching_schema(self, client):
        gen = StructuredGenerator(client, max_attempts=2, use_native_format=False)
        r = gen.generate("anything", Answer, RunConfig(model=MODEL), "answer")
        assert r.ok and 0.0 <= r.data["confidence"] <= 1.0


class TestPhaseOne:
    def test_benchmark_produces_a_full_summary(self, client):
        ts = load_testset().filter(limit=2)
        res = run_benchmark(client, RunConfig(model=MODEL, num_predict=16), ts,
                            repeats=2, warmup=0, measure_cold=True, sample_memory=False)
        assert len(res.runs) == 4
        assert res.summary["success_rate"] == 1.0
        assert res.summary["decode_tps"]["median"] > 0
        assert res.cold_start and res.cold_start["ok"]
        assert res.per_category

    def test_benchmark_report_renders(self, client):
        ts = load_testset().filter(limit=1)
        res = run_benchmark(client, RunConfig(model=MODEL, num_predict=12), ts,
                            repeats=1, warmup=0, measure_cold=False, sample_memory=False)
        md = render_benchmark({"model": MODEL, "hardware": {}, **res.to_dict()})
        assert "Steady-state performance" in md and "Time to first token" in md


class TestPhaseTwo:
    def test_temperature_sweep_shapes(self, client):
        out = run_temperature_study(client, RunConfig(model=MODEL, num_predict=24),
                                    "say something", [0.0, 1.0], samples=3)
        assert len(out["points"]) == 2
        assert all(p["n_samples"] == 3 for p in out["points"])
        assert out["conclusion"]

    def test_higher_temperature_lowers_overlap(self, client):
        out = run_temperature_study(client, RunConfig(model=MODEL, num_predict=40),
                                    "say something", [0.0, 1.0], samples=4)
        cold, hot = out["points"][0], out["points"][1]
        assert cold["mean_pairwise_jaccard"] >= hot["mean_pairwise_jaccard"]

    def test_tempstudy_report_renders(self, client):
        out = run_temperature_study(client, RunConfig(model=MODEL, num_predict=16),
                                    "hello", [0.0], samples=2)
        md = render_tempstudy({"created_at": "now", **out})
        assert "Temperature Determinism Study" in md


class TestPhaseThree:
    def test_comparison_covers_every_model(self, client):
        ts = load_testset().filter(limit=2)
        out = run_comparison(client, ["mock-fast:1b", "mock-slow:7b"], ts,
                             RunConfig(num_predict=16), repeats=1)
        assert len(out["headlines"]) == 2
        assert out["rankings"]["fastest_decode"][0] == "mock-fast:1b"
        assert out["findings"]

    def test_missing_model_is_reported_not_crashed(self, client):
        ts = load_testset().filter(limit=1)
        out = run_comparison(client, ["ghost:1b"], ts, RunConfig(num_predict=8))
        rep = out["reports"][0]
        assert rep["available"] is False and "ollama pull" in rep["error"]

    def test_comparison_report_renders(self, client):
        ts = load_testset().filter(limit=2)
        out = run_comparison(client, ["mock-fast:1b", "mock-mid:3b"], ts,
                             RunConfig(num_predict=12))
        md = render_comparison({"created_at": "now", "hardware": {}, **out})
        assert "Model Comparison Study" in md and "Selection guidance" in md
