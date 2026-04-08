"""aicoder_cli — installable package entry point."""
import sys
from pathlib import Path

# Make the project root (parent of this package) importable
# so that core/, tools/, commands/ can be imported directly.
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from cli import main  # noqa: E402

__all__ = ["main"]
