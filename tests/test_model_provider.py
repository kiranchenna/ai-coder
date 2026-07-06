"""Tests for core/model.py's provider branching (ollama vs openai_compatible)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.model import _build_openai_compatible, get_chat_model


def test_default_provider_builds_chat_ollama(tmp_path, monkeypatch):
    from core.config import get_config

    cfg = get_config()
    saved = dict(cfg.raw()["model"])
    try:
        cfg.raw()["model"]["provider"] = "ollama"
        llm = get_chat_model()
        assert type(llm).__name__ == "ChatOllama"
    finally:
        cfg.raw()["model"] = saved


def test_openai_compatible_provider_builds_chat_openai(monkeypatch):
    from core.config import get_config

    cfg = get_config()
    saved = dict(cfg.raw()["model"])
    try:
        cfg.raw()["model"]["provider"] = "openai_compatible"
        cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"
        cfg.raw()["model"]["api_key"] = "sk-test"
        llm = get_chat_model()
        assert type(llm).__name__ == "ChatOpenAI"
    finally:
        cfg.raw()["model"] = saved


def test_openai_compatible_blank_api_key_uses_placeholder(monkeypatch):
    from core.config import get_config

    cfg = get_config()
    saved = dict(cfg.raw()["model"])
    try:
        cfg.raw()["model"]["provider"] = "openai_compatible"
        cfg.raw()["model"]["base_url"] = "http://localhost:8080/v1"
        cfg.raw()["model"]["api_key"] = ""
        llm = get_chat_model()  # must not raise for a local server with no auth
        assert type(llm).__name__ == "ChatOpenAI"
    finally:
        cfg.raw()["model"] = saved


def test_missing_langchain_openai_raises_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_openai":
            raise ImportError("No module named 'langchain_openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match=r'pip install "ai-coder\[openai\]"'):
        _build_openai_compatible("some-model", "http://localhost:8080/v1", "", 0.3)
