"""Phase 2 unit tests — the JSON boundary is where small models break."""
import pytest
from pydantic import ValidationError

from localmind.schemas import Answer, CommandIntent, get_schema
from localmind.structured import extract_json, parse_json


class TestExtraction:
    def test_clean_json(self):
        assert parse_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_chatty_preamble_and_postamble(self):
        raw = 'Sure! Here is the object:\n{"a": 1}\nLet me know if you need more.'
        assert parse_json(raw) == {"a": 1}

    def test_brace_inside_string_does_not_end_scan(self):
        assert parse_json('{"a": "closing } brace", "b": 2}')["b"] == 2

    def test_trailing_comma_is_repaired(self):
        assert parse_json('{"a": 1,}') == {"a": 1}

    def test_nested_objects(self):
        out = parse_json('{"a": {"b": {"c": [1, 2]}}}')
        assert out["a"]["b"]["c"] == [1, 2]

    def test_top_level_array(self):
        assert parse_json('[1, 2, 3]') == [1, 2, 3]

    def test_no_json_returns_none(self):
        assert parse_json("there is no object here") is None
        assert extract_json("") is None

    def test_unterminated_object_returns_none(self):
        assert parse_json('{"a": 1') is None


class TestSchemas:
    def test_registry_round_trip(self):
        assert get_schema("command") is CommandIntent

    def test_unknown_schema_raises(self):
        with pytest.raises(KeyError):
            get_schema("does-not-exist")

    def test_confidence_bounds_enforced(self):
        with pytest.raises(ValidationError):
            Answer(answer="x", confidence=1.4)

    def test_empty_answer_rejected(self):
        with pytest.raises(ValidationError):
            Answer(answer="   ", confidence=0.5)

    def test_literal_intent_enforced(self):
        with pytest.raises(ValidationError):
            CommandIntent(intent="explode", device="lamp", confidence=0.9)

    def test_valid_object_passes(self):
        c = CommandIntent(intent="set_value", device="lamp", value=40,
                          unit="percent", confidence=0.9)
        assert c.model_dump()["value"] == 40

    def test_every_schema_emits_json_schema(self):
        from localmind.schemas import SCHEMAS
        for name, model in SCHEMAS.items():
            js = model.model_json_schema()
            assert "properties" in js, name
