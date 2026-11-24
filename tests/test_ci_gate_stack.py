"""Tests for the CI gate-stack assertion (examples.ci_gate_stack).

The gate's green mark is only meaningful on the pinned stack, so the workflow
must fail rather than describe a stack nobody measures locally.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from examples import ci_gate_stack
from examples.ci_gate_stack import check_stack, main, parse_requirement

CI_YAML = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


class TestParseRequirement:
    def test_parses_pip_style_spec(self) -> None:
        assert parse_requirement("torch==2.12.0") == ("torch", "2.12.0")

    def test_parses_local_version(self) -> None:
        assert parse_requirement("torch==2.12.0+cpu") == ("torch", "2.12.0+cpu")

    @pytest.mark.parametrize(
        "spec",
        ["torch", "torch>=2.12", "==2.12.0", "torch==", "torch==2.12==0.0"],
    )
    def test_rejects_malformed_spec(self, spec: str) -> None:
        with pytest.raises(ValueError):
            parse_requirement(spec)


class TestCheckStack:
    def test_base_match_ignores_build_suffix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ci_gate_stack, "_installed_version", lambda _d: "2.12.0+cpu")
        problems, report = check_stack([("torch", "2.12.0")])
        assert problems == []
        assert "2.12.0+cpu" in report[0]

    def test_drift_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ci_gate_stack, "_installed_version", lambda _d: "2.14.1+cpu")
        problems, _report = check_stack([("torch", "2.12.0")])
        assert len(problems) == 1
        assert "2.12.0" in problems[0] and "2.14.1+cpu" in problems[0]

    def test_exact_spec_requires_matching_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ci_gate_stack, "_installed_version", lambda _d: "2.12.0+cpu")
        assert check_stack([("torch", "2.12.0+cu126")])[0] != []
        assert check_stack([("torch", "2.12.0+cpu")])[0] == []

    def test_missing_distribution_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_dist: str) -> str:
            raise importlib_metadata_not_found()

        monkeypatch.setattr(ci_gate_stack, "_installed_version", boom)
        problems, _report = check_stack([("torch", "2.12.0")])
        assert len(problems) == 1
        assert "not installed" in problems[0]


class TestMain:
    def test_returns_zero_when_pinned_stack_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ci_gate_stack, "_installed_version", lambda _d: "0.15.8")
        assert main(["ruff==0.15.8"]) == 0

    def test_returns_nonzero_on_drift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ci_gate_stack, "_installed_version", lambda _d: "0.17.6")
        assert main(["ruff==0.15.8"]) == 1

    def test_rejects_malformed_spec(self) -> None:
        with pytest.raises(SystemExit):
            main(["torch"])


def importlib_metadata_not_found() -> Exception:
    import importlib.metadata

    return importlib.metadata.PackageNotFoundError("torch")


class TestWorkflowSingleSource:
    """Pins live in one workflow-level env block; installs and assertions read it."""

    def test_env_block_defines_every_verdict_layer(self) -> None:
        text = CI_YAML.read_text()
        for var in ("GATE_TORCH", "GATE_MYPY", "GATE_RUFF"):
            assert re.search(rf"^\s+{var}: ", text, re.MULTILINE), var

    def test_version_literals_only_appear_in_the_env_block(self) -> None:
        text = CI_YAML.read_text()
        offenders = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"(?:torch|mypy|ruff)==[0-9]", line)
            and not line.strip().startswith("GATE_")
        ]
        assert offenders == []

    def test_installs_reference_the_env_pins(self) -> None:
        text = CI_YAML.read_text()
        installs = [
            line
            for line in text.splitlines()
            if "pip install" in line and any(k in line for k in ("torch", "mypy", "ruff"))
        ]
        assert installs, "expected the three pinned layers to be installed"
        assert all("${{" in line or "${" in line for line in installs), installs

    def test_every_job_asserts_its_stack(self) -> None:
        text = CI_YAML.read_text()
        # test (x2 matrix legs share one step) / lint / typecheck / benchmark.
        assert text.count("examples.ci_gate_stack") == 4

    def test_assertions_cover_the_layers_that_job_installs(self) -> None:
        text = CI_YAML.read_text()
        asserts = [line for line in text.splitlines() if "examples.ci_gate_stack" in line]
        assert sum("GATE_TORCH" in line for line in asserts) == 3
        assert any("GATE_MYPY" in line for line in asserts)
        assert any("GATE_RUFF" in line for line in asserts)

    def test_benchmark_job_is_advisory_yet_version_anchored(self) -> None:
        """Timings record a baseline; they do not gate, but they do name a stack."""
        text = CI_YAML.read_text()
        job = text.split("  benchmark:\n", 1)[1]
        assert "continue-on-error: true" in job
        assert 'python -m examples.ci_gate_stack "${GATE_TORCH}"' in job
        assert "benchmarks.gate_bench" in job
