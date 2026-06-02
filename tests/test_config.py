"""Tests for core/config.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import yaml
from unittest.mock import patch


def _make_config(tmp_path, overrides=None):
    """Helper: create a Config with a temp config.yaml."""
    import copy
    import core.config as cfg_mod

    data = copy.deepcopy(cfg_mod.DEFAULT_CONFIG)
    if overrides:
        cfg_mod._deep_merge(data, overrides)

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")

    # Patch paths so we use the temp directory
    with patch.object(cfg_mod, "AICODER_HOME", tmp_path), \
         patch.object(cfg_mod, "CONFIG_PATH", config_file), \
         patch.object(cfg_mod, "MEMORY_DIR", tmp_path / "memory"):
        cfg = cfg_mod.load_config()

    return cfg


# ─── Default values ───────────────────────────────────────────────────────────

def test_default_model_name(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.model_name == "qwen2.5-coder:7b"


def test_default_shell_mode(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.shell_confirmation == "always"


def test_default_file_confirmation(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.file_confirmation == "auto"


def test_default_file_backup(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.file_backup is True


def test_default_memory_enabled(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.memory_enabled is True


# ─── Overrides ────────────────────────────────────────────────────────────────

def test_override_model_name(tmp_path):
    cfg = _make_config(tmp_path, overrides={"model": {"name": "qwen2.5-coder:7b"}})
    assert cfg.model_name == "qwen2.5-coder:7b"


def test_override_shell_mode(tmp_path):
    cfg = _make_config(tmp_path, overrides={"shell": {"confirmation": "smart"}})
    assert cfg.shell_confirmation == "smart"


# ─── set_shell_confirmation ───────────────────────────────────────────────────

def test_set_shell_confirmation_valid(tmp_path):
    import copy
    import core.config as cfg_mod

    data = copy.deepcopy(cfg_mod.DEFAULT_CONFIG)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(data, sort_keys=False))

    with patch.object(cfg_mod, "AICODER_HOME", tmp_path), \
         patch.object(cfg_mod, "CONFIG_PATH", config_file), \
         patch.object(cfg_mod, "MEMORY_DIR", tmp_path / "memory"):
        cfg = cfg_mod.load_config()
        cfg.set_shell_confirmation("never")
        assert cfg.shell_confirmation == "never"


def test_set_shell_confirmation_invalid(tmp_path):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError):
        cfg.set_shell_confirmation("maybe")


# ─── deep_merge ──────────────────────────────────────────────────────────────

def test_deep_merge_adds_keys():
    import core.config as cfg_mod
    base = {"a": {"x": 1}}
    cfg_mod._deep_merge(base, {"a": {"y": 2}})
    assert base == {"a": {"x": 1, "y": 2}}


def test_deep_merge_override_scalar():
    import core.config as cfg_mod
    base = {"a": 1}
    cfg_mod._deep_merge(base, {"a": 2})
    assert base["a"] == 2


def test_deep_merge_nested():
    import core.config as cfg_mod
    base = {"model": {"name": "a", "temp": 0.3}}
    cfg_mod._deep_merge(base, {"model": {"name": "b"}})
    assert base["model"]["name"] == "b"
    assert base["model"]["temp"] == 0.3  # preserved
