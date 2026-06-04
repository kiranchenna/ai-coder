"""
core/config.py — Configuration management for aicoder
======================================================
Config is stored at ~/.aicoder/config.yaml
Created automatically on first run with sensible defaults.
"""

import hashlib
import sys
from pathlib import Path
from typing import Any
import yaml

# ─── Paths ────────────────────────────────────────────────────────────────────

AICODER_HOME = Path.home() / ".aicoder"
CONFIG_PATH  = AICODER_HOME / "config.yaml"
MEMORY_DIR   = AICODER_HOME / "memory"


def project_id(root: Path) -> str:
    """Stable per-project identifier (name + short path hash). Used to key
    per-project data (memory, plans) under MEMORY_DIR."""
    resolved = root.resolve()
    digest = hashlib.md5(str(resolved).encode()).hexdigest()[:8]
    return f"{resolved.name}_{digest}"

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        # NOTE: Change 'name' to any model you have pulled via `ollama pull <model>`.
        # Good coding models: qwen2.5-coder:4b, qwen2.5-coder:7b, deepseek-coder:6.7b
        "provider": "ollama",
        "name": "qwen2.5-coder:7b",
        "base_url": "http://localhost:11434",
        "temperature": 0.3,
        "temperature_precise": 0.1,
        "context_length": 16384,
    },
    "shell": {
        # Confirmation mode for shell commands:
        #   always  — always ask before running (safe default)
        #   never   — auto-run without asking
        #   smart   — ask for potentially destructive commands only
        "confirmation": "always",
    },
    "files": {
        # How to handle AI-generated file writes:
        #   always  — show diff and ask [y/N] before each changed file
        #   auto    — show diff but apply automatically (informational)
        #   never   — skip diff, write immediately
        "confirmation": "auto",
        # Create a .bak copy of every file before overwriting it
        "backup": True,
    },
    "workspace": {
        "auto_scan": True,
        "ignore_dirs": [
            ".git", "venv", ".venv", "__pycache__", "node_modules",
            ".next", "dist", "build", ".cache", ".mypy_cache",
            ".pytest_cache", "coverage", ".tox",
        ],
        "ignore_extensions": [
            ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe",
            ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
            ".mp4", ".mp3", ".zip", ".tar", ".gz", ".lock",
        ],
        "max_file_size_kb": 200,
        "max_context_files": 30,
    },
    "search": {
        "max_results": 5,
        "timeout_seconds": 10,
    },
    "memory": {
        "enabled": True,
        "max_history": 50,
    },
    "knowledge": {
        # Dedicated embedding model for the vector RAG knowledge base.
        # Uses a separate fast model instead of the main chat model.
        # nomic-embed-text-v2-moe: MoE architecture, multilingual, 523MB, best quality.
        # Set to "" to use the main chat model (no extra download needed).
        "embedding_model": "nomic-embed-text-v2-moe",
    },
    "mcp": {
        # Optional MCP (Model Context Protocol) servers. Their tools are exposed
        # to the agent alongside the built-ins. Requires: pip install "ai-coder[mcp]".
        # Example:
        #   servers:
        #     filesystem:
        #       command: npx
        #       args: ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        "servers": {},
    },
    "hooks": {
        # Optional shell commands run on agent events. Opt-in. Example:
        #   PreToolUse:   [{matcher: "run_shell", command: "guard.sh"}]  # non-zero blocks
        #   PostToolUse:  [{matcher: "write_file|edit_file", command: "ruff format ."}]
        #   Stop:         [{command: "notify-send done"}]
    },
    "devmode": {
        # Developer Mode: refine each phase decision with a draft→critique→revise
        # pass (better depth from a small model, at the cost of an extra call).
        "reflect": True,
        # `dev build`: after generating each file, run a self-review pass that
        # checks it against the spec/conventions and fixes bugs before writing.
        "build_review": True,
        # After each phase, check its decision against earlier phases for
        # contradictions (e.g. a schema that violates the security model) and
        # surface them immediately instead of waiting for the final review.
        "consistency_check": True,
        # For critical phases, generate several candidate decisions and let a
        # judge pick the strongest (more calls, better worst-case quality).
        "best_of": True,
    },
}


# ─── Config class ─────────────────────────────────────────────────────────────

class Config:
    """Thin wrapper around a nested dict that provides attribute-style access."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # ── Model settings ────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._data["model"]["name"]

    @property
    def model_base_url(self) -> str:
        return self._data["model"]["base_url"]

    @property
    def model_temperature(self) -> float:
        return float(self._data["model"]["temperature"])

    @property
    def model_temperature_precise(self) -> float:
        return float(self._data["model"]["temperature_precise"])

    @property
    def model_context_length(self) -> int:
        return int(self._data["model"].get("context_length", 8192))

    # ── Shell settings ────────────────────────────────────────────────────────

    @property
    def shell_confirmation(self) -> str:
        return self._data["shell"]["confirmation"]

    def set_shell_confirmation(self, mode: str) -> None:
        """Update shell confirmation mode and persist to disk."""
        if mode not in ("always", "never", "smart"):
            raise ValueError(f"Invalid mode: {mode!r}. Use: always | never | smart")
        self._data["shell"]["confirmation"] = mode
        save_config(self._data)

    # ── File settings ──────────────────────────────────────────────────

    @property
    def file_confirmation(self) -> str:
        return self._data.get("files", {}).get("confirmation", "auto")

    @property
    def file_backup(self) -> bool:
        return bool(self._data.get("files", {}).get("backup", True))

    def set_file_confirmation(self, mode: str) -> None:
        """Update file review mode and persist to disk."""
        if mode not in ("always", "auto", "never"):
            raise ValueError(f"Invalid mode: {mode!r}. Use: always | auto | never")
        if "files" not in self._data:
            self._data["files"] = {}
        self._data["files"]["confirmation"] = mode
        save_config(self._data)

    # ── Workspace settings ────────────────────────────────────────────────────

    @property
    def ignore_dirs(self) -> list[str]:
        return self._data["workspace"]["ignore_dirs"]

    @property
    def ignore_extensions(self) -> list[str]:
        return self._data["workspace"]["ignore_extensions"]

    @property
    def max_file_size_kb(self) -> int:
        return int(self._data["workspace"]["max_file_size_kb"])

    @property
    def max_context_files(self) -> int:
        return int(self._data["workspace"]["max_context_files"])

    # ── Search settings ───────────────────────────────────────────────────────

    @property
    def search_max_results(self) -> int:
        return int(self._data["search"]["max_results"])

    @property
    def search_timeout(self) -> int:
        return int(self._data["search"]["timeout_seconds"])

    # ── Memory settings ───────────────────────────────────────────────────────

    @property
    def memory_enabled(self) -> bool:
        return bool(self._data["memory"]["enabled"])

    @property
    def memory_max_history(self) -> int:
        return int(self._data["memory"]["max_history"])

    # ── Knowledge / RAG settings ──────────────────────────────────────────────

    @property
    def embedding_model(self) -> str:
        """
        Model used for generating embeddings in the vector knowledge base.
        Falls back to the main chat model if empty or not set.
        """
        return (
            self._data.get("knowledge", {}).get("embedding_model", "")
            or self._data["model"]["name"]
        )

    # ── Raw access ────────────────────────────────────────────────────────────

    def get(self, *keys: str, default: Any = None) -> Any:
        """Navigate nested keys: config.get('model', 'name')"""
        d = self._data
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    def raw(self) -> dict:
        return self._data


# ─── Load / Save ──────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (base is mutated in-place)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def load_config() -> Config:
    """
    Load config from ~/.aicoder/config.yaml.
    Creates the file with defaults if it does not exist.
    """
    AICODER_HOME.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    import copy
    data = copy.deepcopy(DEFAULT_CONFIG)

    if CONFIG_PATH.exists():
        try:
            user_data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            _deep_merge(data, user_data)
        except yaml.YAMLError as e:
            print(f"[WARNING] Config parse error: {e}. Using defaults.", file=sys.stderr)
    else:
        # First run: write default config
        save_config(data)

    return Config(data)


def save_config(data: dict) -> None:
    """Write config dict to ~/.aicoder/config.yaml."""
    AICODER_HOME.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


# Singleton — loaded once at import time
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config
