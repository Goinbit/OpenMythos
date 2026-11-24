"""Assert the CI gate ran on the stack the local gate actually measures.

A green check mark is only external evidence when the stack behind it is the
canonical one (`.venv-arm64`). The first real CI run proved otherwise: the
"4/4" came from torch 2.14.1/2.15.0 while the local green was measured on
2.12.0. So the workflow keeps its pins in one env block, installs from them,
and verifies what actually landed before running its verdict.

Usage:
    python -m examples.ci_gate_stack torch==2.12.0 mypy==2.4.1

A pin with a local segment (``torch==2.12.0+cpu``) demands that exact build;
without one, the build tag is reported but not constrained.
"""

from __future__ import annotations

import argparse
import importlib.metadata


def _installed_version(dist: str) -> str:
    return importlib.metadata.version(dist)


def parse_requirement(spec: str) -> tuple[str, str]:
    parts = spec.split("==")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"expected 'name==version', got {spec!r}")
    return parts[0], parts[1]


def check_stack(
    requirements: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Return ``(problems, report)`` for ``[(dist, pinned_version), ...]``."""
    problems: list[str] = []
    report: list[str] = []
    for dist, want in requirements:
        try:
            got = _installed_version(dist)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{dist}: not installed (pin {want})")
            continue
        matches = got == want if "+" in want else got.split("+", 1)[0] == want
        if matches:
            report.append(f"{dist} {got} matches pin {want}")
        else:
            problems.append(f"{dist}: resolved {got}, pin {want}")
    return problems, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("specs", nargs="+", metavar="name==version")
    args = parser.parse_args(argv)
    try:
        requirements = [parse_requirement(spec) for spec in args.specs]
    except ValueError as exc:
        parser.error(str(exc))  # NoReturn
    problems, report = check_stack(requirements)
    for line in report:
        print(f"[gate-stack] OK {line}")
    for problem in problems:
        print(f"[gate-stack] FAIL {problem}", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
