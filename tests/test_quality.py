"""Phase 3 unit tests — a scorer you cannot test is a scorer you cannot trust."""
from localmind.quality import HeuristicScorer, compliance_score, keyword_coverage, repetition_score
from localmind.testset import PromptSpec, load_testset


def spec(**kw) -> PromptSpec:
    base = dict(id="t", category="test", prompt="p", expected_keywords=[], checks={})
    base.update(kw)
    return PromptSpec(**base)


def test_keyword_coverage_is_fractional():
    assert keyword_coverage("the cat sat", ["cat", "dog"]) == 0.5
    assert keyword_coverage("nothing", []) is None


def test_repetition_detects_loops():
    assert repetition_score("a b c d e f g h i j k l") == 1.0
    looped = "the same thing " * 12
    assert repetition_score(looped) < 0.3


def test_word_cap_compliance():
    score, fails = compliance_score("one two three", {"max_words": 2})
    assert score == 0.0 and "word cap" in fails[0]
    score, fails = compliance_score("one two", {"max_words": 2})
    assert score == 1.0 and not fails


def test_banned_words_are_caught():
    score, fails = compliance_score("this is a protocol", {"must_not_contain": ["protocol"]})
    assert score == 0.0 and "banned" in fails[0]


def test_json_requirement():
    assert compliance_score('{"a":1}', {"must_be_json": True})[0] == 1.0
    assert compliance_score("nope", {"must_be_json": True})[0] == 0.0


def test_bullet_counting_handles_dashes_and_numbers():
    text = "- one\n- two\n- three"
    assert compliance_score(text, {"min_bullets": 3})[0] == 1.0
    assert compliance_score("1. a\n2. b", {"min_bullets": 3})[0] == 0.0


def test_empty_output_scores_zero():
    s = HeuristicScorer().score(spec(expected_keywords=["x"]), "")
    assert s.composite == 0.0 and s.empty


def test_refusal_is_penalised():
    sp = spec(expected_keywords=["cannot"])
    s = HeuristicScorer().score(sp, "I cannot help with that.")
    assert s.refusal and s.composite < 100


def test_perfect_answer_scores_high():
    sp = spec(expected_keywords=["ram", "disk"], checks={"max_words": 20})
    s = HeuristicScorer().score(sp, "RAM is volatile; a disk keeps data without power.")
    assert s.composite > 95


def test_scores_are_bounded():
    sp = spec(expected_keywords=["a"], checks={"max_words": 5})
    for text in ["", "a", "a " * 200, "x" * 500]:
        assert 0.0 <= HeuristicScorer().score(sp, text).composite <= 100.0


class TestTestSet:
    def test_loads_at_least_thirty_prompts(self):
        ts = load_testset()
        assert len(ts) >= 30, "Phase 3 requires 30-50 standardised prompts"
        assert len(ts) <= 50

    def test_ids_are_unique(self):
        ids = [p.id for p in load_testset().prompts]
        assert len(ids) == len(set(ids))

    def test_every_prompt_has_a_signal(self):
        for p in load_testset().prompts:
            assert p.expected_keywords or p.checks, f"{p.id} is unscoreable"

    def test_filtering(self):
        ts = load_testset()
        code = ts.filter(["code"])
        assert code.prompts and all(p.category == "code" for p in code.prompts)
        assert len(ts.filter(limit=5)) == 5
