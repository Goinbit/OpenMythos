"""Keep the Claude Code instruction mirror exact after harness normalization."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
_PAIR = "AGENTS.md/CLAUDE.md"
_COLLAPSED_PAIR = "CLAUDE.md/CLAUDE.md"
_MARK = "\u0000PAIR\u0000"
_COLLAPSED_MARK = "\u0000COLLAPSED_PAIR\u0000"


def _to_claude(source: str) -> str:
    for old, new in (
        (_PAIR, _MARK),
        (_COLLAPSED_PAIR, _COLLAPSED_MARK),
        ("# AGENTS.md", "# CLAUDE.md"),
        ("AGENTS.md", "CLAUDE.md"),
        (".Codex", ".claude"),
        ("Codex harness", "Claude Code harness"),
        (_MARK, _PAIR),
        (_COLLAPSED_MARK, _COLLAPSED_PAIR),
    ):
        source = source.replace(old, new)
    return source


def _to_agents(source: str) -> str:
    for old, new in (
        (_PAIR, _MARK),
        (_COLLAPSED_PAIR, _COLLAPSED_MARK),
        ("Claude Code harness", "Codex harness"),
        (".claude", ".Codex"),
        ("CLAUDE.md", "AGENTS.md"),
        (_MARK, _PAIR),
        (_COLLAPSED_MARK, _COLLAPSED_PAIR),
    ):
        source = source.replace(old, new)
    return source


def test_claude_mirror_is_exact() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text()
    claude = (REPO_ROOT / "CLAUDE.md").read_text()
    assert claude == _to_claude(agents)
    assert _to_agents(claude) == agents
