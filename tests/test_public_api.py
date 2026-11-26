"""Public package import contract (P0.1)."""
from __future__ import annotations

import subprocess
import sys

import open_mythos


def test_public_api_importable() -> None:
    namespace: dict[str, object] = {}
    exec("from open_mythos import *", namespace)
    for name in open_mythos.__all__:
        assert name in namespace, name
        assert getattr(open_mythos, name) is namespace[name], name


def test_public_api_star_import_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from open_mythos import *"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
