"""Tests for core/config.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
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
    assert cfg.model_name == "qwen2.5-coder-7b-instruct"


def test_default_model_provider_is_openai_compatible(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.model_provider == "openai_compatible"


def test_model_provider_normalizes_case_and_whitespace(tmp_path):
    cfg = _make_config(tmp_path, {"model": {"provider": " OpenAI_Compatible "}})
    assert cfg.model_provider == "openai_compatible"


def test_default_model_api_key_is_empty(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.model_api_key == ""


def test_model_api_key_explicit_value(tmp_path):
    cfg = _make_config(tmp_path, {"model": {"api_key": "sk-test-123"}})
    assert cfg.model_api_key == "sk-test-123"


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


def test_model_context_length_defaults_to_128k(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.model_context_length == 131072


def test_set_model_context_length_persists(tmp_path):
    import copy
    import core.config as cfg_mod

    data = copy.deepcopy(cfg_mod.DEFAULT_CONFIG)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(data, sort_keys=False))

    with patch.object(cfg_mod, "AICODER_HOME", tmp_path), \
         patch.object(cfg_mod, "CONFIG_PATH", config_file), \
         patch.object(cfg_mod, "MEMORY_DIR", tmp_path / "memory"):
        cfg = cfg_mod.load_config()
        cfg.set_model_context_length(32768)
        assert cfg.model_context_length == 32768
        # Reload from disk — confirms it was actually persisted, not just
        # updated in memory.
        reloaded = cfg_mod.load_config()
        assert reloaded.model_context_length == 32768


def test_set_model_context_length_rejects_non_positive(tmp_path):
    cfg = _make_config(tmp_path)
    with pytest.raises(ValueError):
        cfg.set_model_context_length(0)
    with pytest.raises(ValueError):
        cfg.set_model_context_length(-1)


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


# ─── Developer Mode profiles & lever resolution ───────────────────────────────

def test_default_profile_is_balanced(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.devmode_profile() == "balanced"


def test_balanced_profile_enables_reflect_not_best_of(tmp_path):
    # The eval's key finding, encoded as the default: reflect on, best_of off.
    cfg = _make_config(tmp_path)
    assert cfg.devmode_lever("reflect") is True
    assert cfg.devmode_lever("consistency_check") is True
    assert cfg.devmode_lever("best_of") is False


def test_fast_profile_is_reflect_only(tmp_path):
    cfg = _make_config(tmp_path, {"devmode": {"profile": "fast"}})
    assert cfg.devmode_lever("reflect") is True
    assert cfg.devmode_lever("consistency_check") is False
    assert cfg.devmode_lever("build_review") is False


def test_unknown_profile_falls_back_to_balanced(tmp_path):
    cfg = _make_config(tmp_path, {"devmode": {"profile": "turbo"}})
    assert cfg.devmode_profile() == "balanced"


def test_explicit_lever_overrides_profile(tmp_path):
    # An explicit bool wins over the profile in either direction.
    cfg = _make_config(tmp_path, {"devmode": {"profile": "fast", "consistency_check": True}})
    assert cfg.devmode_lever("consistency_check") is True


def test_best_of_gated_off_without_judge_model(tmp_path):
    # thorough wants best_of, but with no judge_model it must not fire.
    cfg = _make_config(tmp_path, {"devmode": {"profile": "thorough"}})
    assert cfg._devmode_lever_intended("best_of") is True
    assert cfg.devmode_lever("best_of") is False
    assert cfg.devmode_best_of_gated() is True


def test_best_of_fires_with_judge_model(tmp_path):
    cfg = _make_config(tmp_path, {"devmode": {"profile": "thorough", "judge_model": "big:14b"}})
    assert cfg.devmode_lever("best_of") is True
    assert cfg.devmode_best_of_gated() is False
