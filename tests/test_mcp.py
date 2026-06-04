"""Unit tests for MCP client helpers (no live server needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.mcp_client import _content_to_text, _schema_to_model


class _Block:
    def __init__(self, text):
        self.text = text


class _Result:
    def __init__(self, content, is_error=False):
        self.content = content
        self.isError = is_error


def test_content_to_text_joins_blocks():
    r = _Result([_Block("hello"), _Block("world")])
    assert _content_to_text(r) == "hello\nworld"


def test_content_to_text_marks_errors():
    r = _Result([_Block("boom")], is_error=True)
    assert _content_to_text(r).startswith("ERROR: ")


def test_content_to_text_empty():
    assert _content_to_text(_Result([])) == "(no output)"


def test_schema_to_model_required_and_optional():
    model = _schema_to_model("t", {
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
        "required": ["a"],
    })
    # required field present validates; optional missing is fine
    inst = model(a=3)
    assert inst.a == 3 and inst.b is None
    # missing required field raises
    import pytest
    with pytest.raises(Exception):
        model(b="x")


def test_schema_to_model_empty_schema():
    model = _schema_to_model("empty", {})
    assert model() is not None  # no fields, constructs cleanly
