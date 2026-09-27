"""Runs every benchmark and writes a Markdown report to `benchmarks/results/`.

Run with `uv run python benchmarks/run_all.py`. The runner discovers every
`benchmarks/bench_*.py` script. The report is named `<date>-<architecture>.md`; when a
report of that name exists, the new one gets a numeric suffix, such as
`<date>-<architecture>-2.md`, so an earlier run is never overwritten. Pass `--dry-run` to
print the benchmarks and the report path without running anything.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

import enceladus

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def benchmarks() -> list[str]:
    """Returns the name of every `bench_*.py` script, in alphabetical order."""
    return sorted(p.stem for p in HERE.glob("bench_*.py"))


def report_path(date: str, arch: str) -> Path:
    """Returns the first report path for `date` and `arch` that doesn't exist yet."""
    out = RESULTS / f"{date}-{arch}.md"
    n = 2
    while out.exists():
        out = RESULTS / f"{date}-{arch}-{n}.md"
        n += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the benchmarks and the report path, and exit")  # fmt: skip
    args = parser.parse_args()
    caps = enceladus.get_device().caps
    date = datetime.date.today().isoformat()
    names = benchmarks()
    out = report_path(date, caps.architecture)
    if args.dry_run:
        print("\n".join(names))
        print(f"would write {out}")
        return
    lines = [
        f"# Enceladus benchmarks, {date}",
        "",
        f"Device: {caps.name} ({caps.architecture}). Enceladus {enceladus.__version__}.",
        "Enceladus times use GPU timestamps; MLX and PyTorch times use the wall clock.",
        "",
    ]
    for name in names:
        print(f"running {name}...", flush=True)
        r = subprocess.run([sys.executable, str(HERE / f"{name}.py")], capture_output=True,
                           text=True)  # fmt: skip
        status = [] if r.returncode == 0 else [f"The script exited with status {r.returncode}.",
                                               ""]  # fmt: skip
        lines += [f"## {name}", "", *status, "```text", (r.stdout + r.stderr).strip(), "```", ""]
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
