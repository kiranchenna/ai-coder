"""Tests for core/model.py's chat model factory (always openai_compatible)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.model import _build_openai_compatible, get_chat_model


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
    with pytest.raises(RuntimeError, match=r"pip install langchain-openai"):
        _build_openai_compatible("some-model", "http://localhost:8080/v1", "", 0.3)


# ── LM Studio discovery (lms CLI + /v1/models) ──────────────────────────────────
# subprocess.run and httpx.get are mocked throughout — these must never
# depend on `lms` or a real LM Studio server being present on the test
# machine (and, per this session's own live testing, `lms` can hang on
# interactive prompts if a mock is ever missed — a real regression risk to
# guard against explicitly, not just for portability).

class _FakeCompletedProcess:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _mock_lms(monkeypatch, *, stdout="[]", returncode=0, stderr="", raise_missing=False, raise_timeout=False):
    import subprocess

    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if raise_missing:
            raise FileNotFoundError("lms not found")
        if raise_timeout:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))
        return _FakeCompletedProcess(stdout=stdout, stderr=stderr, returncode=returncode)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_list_lmstudio_models_parses_lms_ls_json(monkeypatch):
    from core.model import list_lmstudio_models

    payload = (
        '[{"modelKey": "qwen2.5-coder-7b-instruct-mlx", "sizeBytes": 4300277939, "vision": false},'
        ' {"modelKey": "some-vl-model", "sizeBytes": 1000, "vision": true}]'
    )
    calls = _mock_lms(monkeypatch, stdout=payload)
    models = list_lmstudio_models()
    assert calls[0][:3] == ["lms", "ls", "--llm"]
    assert models == [
        {"name": "qwen2.5-coder-7b-instruct-mlx", "size": 4300277939, "vision": False},
        {"name": "some-vl-model", "size": 1000, "vision": True},
    ]


def test_list_lmstudio_models_vision_only_filters(monkeypatch):
    from core.model import list_lmstudio_models

    payload = (
        '[{"modelKey": "a", "sizeBytes": 1, "vision": false},'
        ' {"modelKey": "b", "sizeBytes": 2, "vision": true}]'
    )
    _mock_lms(monkeypatch, stdout=payload)
    models = list_lmstudio_models(vision_only=True)
    assert [m["name"] for m in models] == ["b"]


def test_list_lmstudio_models_raises_when_lms_not_on_path(monkeypatch):
    from core.model import list_lmstudio_models

    _mock_lms(monkeypatch, raise_missing=True)
    with pytest.raises(RuntimeError, match="lms"):
        list_lmstudio_models()


def test_list_lmstudio_models_raises_on_nonzero_exit(monkeypatch):
    from core.model import list_lmstudio_models

    _mock_lms(monkeypatch, returncode=1, stderr="boom")
    with pytest.raises(RuntimeError, match="boom"):
        list_lmstudio_models()


def test_list_lmstudio_embedding_models_parses_ls_embedding_json(monkeypatch):
    from core.model import list_lmstudio_embedding_models

    payload = '[{"modelKey": "text-embedding-nomic-embed-text-v1.5", "sizeBytes": 84106624}]'
    calls = _mock_lms(monkeypatch, stdout=payload)
    models = list_lmstudio_embedding_models()
    assert calls[0][:3] == ["lms", "ls", "--embedding"]
    assert models == [{"name": "text-embedding-nomic-embed-text-v1.5", "size": 84106624}]


def test_is_lmstudio_model_downloaded_true_and_false(monkeypatch):
    from core.model import is_lmstudio_model_downloaded

    _mock_lms(monkeypatch, stdout='[{"modelKey": "present", "sizeBytes": 1, "vision": false}]')
    assert is_lmstudio_model_downloaded("present") is True
    assert is_lmstudio_model_downloaded("absent") is False


def test_is_lmstudio_model_downloaded_none_when_unreachable(monkeypatch):
    from core.model import is_lmstudio_model_downloaded

    _mock_lms(monkeypatch, raise_missing=True)
    assert is_lmstudio_model_downloaded("anything") is None


def test_switch_lmstudio_model_unloads_other_llms_and_loads_target(monkeypatch):
    import json
    import subprocess

    from core.model import switch_lmstudio_model

    ps_payload = json.dumps([
        {"type": "llm", "identifier": "old-model"},
        {"type": "embedding", "identifier": "embed-model"},  # must be left alone
    ])
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "ps":
            return _FakeCompletedProcess(stdout=ps_payload)
        return _FakeCompletedProcess(stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    switch_lmstudio_model("new-model")

    unload_calls = [c for c in calls if c[1] == "unload"]
    load_calls = [c for c in calls if c[1] == "load"]
    assert unload_calls == [["lms", "unload", "old-model"]]  # not the embedding model
    assert load_calls == [["lms", "load", "new-model", "-y"]]


def test_switch_lmstudio_model_is_a_noop_when_already_loaded(monkeypatch):
    # Live-reproduced bug: `lms load` on an already-loaded modelKey doesn't
    # reuse it — it spins up a second, separately-identified instance,
    # silently doubling RAM usage. Must skip the load call entirely instead.
    import json
    import subprocess

    from core.model import switch_lmstudio_model

    ps_payload = json.dumps([{"type": "llm", "identifier": "already-loaded"}])
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "ps":
            return _FakeCompletedProcess(stdout=ps_payload)
        return _FakeCompletedProcess(stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    switch_lmstudio_model("already-loaded")

    assert not any(c[1] == "load" for c in calls)
    assert not any(c[1] == "unload" for c in calls)  # the target itself is never unloaded


def test_is_lmstudio_reachable_returns_model_ids(monkeypatch):
    import httpx

    from core.model import is_lmstudio_reachable

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "model-a"}, {"id": "model-b"}]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())
    assert is_lmstudio_reachable("http://localhost:1234/v1") == {"model-a", "model-b"}


def test_is_lmstudio_reachable_none_when_unreachable(monkeypatch):
    import httpx

    from core.model import is_lmstudio_reachable

    def raise_connect_error(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", raise_connect_error)
    assert is_lmstudio_reachable("http://localhost:1234/v1") is None


# ── is_lmstudio_endpoint / ensure_lmstudio_running (auto-start) ─────────────────

def test_is_lmstudio_endpoint_matches_default_url_only():
    from core.model import is_lmstudio_endpoint

    assert is_lmstudio_endpoint("http://localhost:1234/v1") is True
    assert is_lmstudio_endpoint("http://localhost:1234/v1/") is True  # trailing slash
    assert is_lmstudio_endpoint("https://api.example.com/v1") is False
    assert is_lmstudio_endpoint("http://localhost:8080/v1") is False  # a different local server


def test_ensure_lmstudio_running_is_a_noop_when_already_reachable(monkeypatch):
    import subprocess

    import core.model as model_mod

    monkeypatch.setattr(model_mod, "is_lmstudio_reachable", lambda base_url: {"m"})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not shell out when already reachable")))
    assert model_mod.ensure_lmstudio_running("m") is True


def test_ensure_lmstudio_running_starts_the_server_and_loads_the_model(monkeypatch):
    import subprocess

    import core.model as model_mod

    calls = []
    reachable = iter([None, {"target-model"}])  # unreachable, then reachable post-start

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _FakeCompletedProcess(stdout="[]")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(model_mod, "is_lmstudio_reachable", lambda base_url: next(reachable))
    assert model_mod.ensure_lmstudio_running("target-model") is True
    assert calls[0] == ["lms", "server", "start"]
    # switch_lmstudio_model("target-model") ran too — it `lms ps`'d then loaded it
    assert ["lms", "load", "target-model", "-y"] in calls


def test_ensure_lmstudio_running_reports_progress_via_on_status(monkeypatch):
    """The caller-supplied on_status callback should narrate both steps, in
    order, so a caller (cli.py) can surface what's happening behind the
    scenes instead of this multi-second operation looking like a hang."""
    import subprocess

    import core.model as model_mod

    reachable = iter([None, {"target-model"}])
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _FakeCompletedProcess(stdout="[]"))
    monkeypatch.setattr(model_mod, "is_lmstudio_reachable", lambda base_url: next(reachable))

    messages = []
    model_mod.ensure_lmstudio_running("target-model", on_status=messages.append)
    assert len(messages) == 2
    assert "starting" in messages[0].lower()
    assert "target-model" in messages[1]


def test_ensure_lmstudio_running_on_status_not_called_when_already_reachable(monkeypatch):
    import core.model as model_mod

    monkeypatch.setattr(model_mod, "is_lmstudio_reachable", lambda base_url: {"m"})
    messages = []
    assert model_mod.ensure_lmstudio_running("m", on_status=messages.append) is True
    assert messages == []


def test_ensure_lmstudio_running_false_when_server_start_fails(monkeypatch):
    import subprocess

    import core.model as model_mod

    monkeypatch.setattr(model_mod, "is_lmstudio_reachable", lambda base_url: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(
        stderr="no LM Studio instance running", returncode=1))
    messages = []
    assert model_mod.ensure_lmstudio_running("target-model", on_status=messages.append) is False
    # The specific failure reason must be surfaced, not just swallowed into a bare False.
    assert "no LM Studio instance running" in messages[-1]


def test_ensure_lmstudio_running_reports_lms_not_installed(monkeypatch):
    """Live-reproduced gap: a user with no LM Studio installed at all (`lms`
    not on PATH) used to just see a generic "open LM Studio" warning with no
    mention that it isn't installed. `_run_lms` already raises an actionable
    RuntimeError for this (see its FileNotFoundError handling) — it must
    reach the caller via on_status instead of being discarded."""
    import subprocess

    import core.model as model_mod

    monkeypatch.setattr(model_mod, "is_lmstudio_reachable", lambda base_url: None)

    def raise_not_found(*a, **k):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", raise_not_found)
    messages = []
    assert model_mod.ensure_lmstudio_running("target-model", on_status=messages.append) is False
    assert any("install LM Studio" in m for m in messages)


def test_ensure_lmstudio_running_false_when_still_unreachable_after_start(monkeypatch):
    import subprocess

    import core.model as model_mod

    monkeypatch.setattr(model_mod, "is_lmstudio_reachable", lambda base_url: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(stdout=""))
    messages = []
    assert model_mod.ensure_lmstudio_running("target-model", on_status=messages.append) is False
    assert "still not reachable" in messages[-1]
