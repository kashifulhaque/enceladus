"""Checks that the user guide stays in sync with the code."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"guide_{name}", GUIDE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_language_reference_is_current():
    """Fails when a builtin or its docstring changed without regenerating the reference."""
    gen = _load("gen_reference")
    assert gen.OUTPUT.read_text() == gen.generate(), (
        "docs/guide/language-reference.md is out of date; run "
        "`uv run python docs/guide/gen_reference.py`"
    )


@pytest.mark.slow
def test_guide_snippets_run():
    """Runs every Python code block in the guide and the README (5-10 s)."""
    proc = subprocess.run([sys.executable, str(GUIDE / "check_snippets.py")],
                          capture_output=True, text=True, timeout=900)  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
