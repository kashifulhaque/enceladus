"""Runs the Python code blocks in the guide's Markdown pages.

Each ```python block runs as its own script in a fresh subprocess, in a temporary
directory, with the interpreter that runs this script. A kernel needs its source in a
file, so the checker writes each program to disk before running it. An HTML comment
right before a block changes how it runs:

- `<!-- snippet: skip -->` doesn't run the block, for fragments and pseudocode.
- `<!-- snippet: continue -->` appends the block to the previous program and runs the
  result, for a tutorial that builds one program in steps.
- `<!-- snippet: env NAME=VALUE ... -->` sets environment variables for the block.

Run it from anywhere, for example against an installed wheel:

    uv run python docs/guide/check_snippets.py docs/guide/quickstart.md

With no arguments, it checks every page in `docs/guide/` and `README.md`. It exits with
status 1 if any block fails.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

GUIDE = Path(__file__).resolve().parent
ROOT = GUIDE.parents[1]
FENCE = re.compile(r"^```(\w*)\s*$")
DIRECTIVE = re.compile(r"^<!--\s*snippet:\s*(skip|continue|env)\b(.*?)-->\s*$")


@dataclass
class Snippet:
    path: Path
    line: int
    code: str
    env: dict[str, str] = field(default_factory=dict)


def extract(path: Path) -> list[Snippet]:
    """Returns the programs to run from one Markdown file, in order."""
    lines = path.read_text().splitlines()
    programs: list[Snippet] = []
    directives: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        m = DIRECTIVE.match(lines[i].strip())
        if m:
            directives.append((m.group(1), m.group(2).strip()))
            i += 1
            continue
        f = FENCE.match(lines[i])
        if not f:
            if lines[i].strip():
                directives = []  # a directive applies only to the block right after it
            i += 1
            continue
        lang, start = f.group(1), i + 1
        i += 1
        body = []
        while i < len(lines) and not lines[i].startswith("```"):
            body.append(lines[i])
            i += 1
        i += 1
        kinds = {k for k, _ in directives}
        env = {}
        for k, arg in directives:
            if k == "env":
                env.update(kv.split("=", 1) for kv in arg.split())
        directives = []
        if lang != "python" or "skip" in kinds:
            continue
        code = "\n".join(body) + "\n"
        if "continue" in kinds and programs:
            prev = programs[-1]
            prev.code += "\n" + code
            prev.env.update(env)
        else:
            programs.append(Snippet(path, start, code, env))
    return programs


def run(snippet: Snippet, workdir: Path) -> tuple[bool, str]:
    """Runs one program and returns whether it succeeded, and its output."""
    script = workdir / f"snippet_{snippet.path.stem}_{snippet.line}.py"
    script.write_text(snippet.code)
    env = {**os.environ, **snippet.env}
    proc = subprocess.run([sys.executable, str(script)], cwd=workdir, env=env,
                          capture_output=True, text=True, timeout=600)  # fmt: skip
    return proc.returncode == 0, proc.stdout + proc.stderr


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [*sorted(GUIDE.glob("*.md")), ROOT / "README.md"]
    failures = 0
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for path in paths:
            for s in extract(path):
                total += 1
                ok, out = run(s, Path(tmp))
                where = f"{path.name}:{s.line}"
                print(f"{'ok  ' if ok else 'FAIL'} {where}", flush=True)
                if not ok:
                    failures += 1
                    print(out, file=sys.stderr)
    print(f"{total - failures} of {total} snippets passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
