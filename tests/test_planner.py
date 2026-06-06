"""Tests for the resumable planner's task-status handling."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.planner import Planner


class FakeSession:
    """Stands in for AgentSession: records sends, reports turn completion."""

    def __init__(self, complete: bool):
        self.last_turn_complete = complete
        self.sent: list[str] = []

    def send(self, msg: str) -> str:
        self.sent.append(msg)
        return ""


def _planner(session) -> Planner:
    ws = Path(tempfile.mkdtemp())
    p = Planner(ws, session)
    p.state_file = ws / "plan.json"   # keep state out of ~/.aicoder
    return p


def _one_task_plan() -> dict:
    return {"goal": "g", "tasks": [
        {"id": 1, "title": "t1", "description": "d", "status": "pending"},
    ]}


def test_task_left_pending_when_step_cap_hit(monkeypatch):
    # The agent ran out of steps (last_turn_complete False) → task must NOT be
    # marked done, so 'resume' can give it more steps.
    p = _planner(FakeSession(complete=False))
    p.save(_one_task_plan())
    p.run()
    reloaded = p.load()
    assert reloaded["tasks"][0]["status"] == "pending"


def test_task_marked_done_when_turn_completes(monkeypatch):
    # A genuine completion → task is done, and we don't get stuck re-asking.
    from rich.prompt import Prompt
    monkeypatch.setattr(Prompt, "ask", staticmethod(lambda *a, **k: "n"))  # stop after task 1
    p = _planner(FakeSession(complete=True))
    p.save(_one_task_plan())
    p.run()
    reloaded = p.load()
    assert reloaded["tasks"][0]["status"] == "done"
