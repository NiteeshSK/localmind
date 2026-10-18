"""Statistics and variance-math tests."""
from localmind.metrics import aggregate, describe, pct
from localmind.tempstudy import jaccard, mean_pairwise_jaccard


def test_describe_handles_empty():
    assert describe([])["n"] == 0
    assert describe([None, None])["n"] == 0


def test_describe_basic_stats():
    d = describe([1, 2, 3, 4])
    assert d["median"] == 2.5 and d["min"] == 1 and d["max"] == 4 and d["n"] == 4


def test_cv_is_zero_for_constant_series():
    assert describe([5, 5, 5])["cv"] == 0.0


def test_percentile_bounds():
    vals = list(range(1, 101))
    assert pct(vals, 50) in (50, 51)
    assert pct(vals, 95) in (95, 96)
    assert pct([], 95) is None


def test_aggregate_counts_failures():
    rows = [{"ok": True, "ttft_ms": 10, "decode_tps": 5, "wall_ms": 100},
            {"ok": False, "ttft_ms": None, "decode_tps": 0, "wall_ms": 0}]
    a = aggregate(rows)
    assert a["runs"] == 2 and a["failed"] == 1 and a["success_rate"] == 0.5
    assert a["ttft_ms"]["n"] == 1  # failed runs excluded from timing stats


def test_jaccard_extremes():
    assert jaccard("a b", "a b") == 1.0
    assert jaccard("a b", "c d") == 0.0
    assert jaccard("", "") == 1.0


def test_mean_pairwise_jaccard_single_sample():
    assert mean_pairwise_jaccard(["only one"]) == 1.0
