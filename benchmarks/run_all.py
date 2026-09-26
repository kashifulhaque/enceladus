"""Runs every benchmark and writes a Markdown report to `benchmarks/results/`.

Run with `uv run python benchmarks/run_all.py`. The report is named
`<date>-<architecture>.md`.
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

import enceladus

HERE = Path(__file__).resolve().parent
BENCHMARKS = ["bench_dispatch", "bench_elementwise", "bench_softmax", "bench_norms",
              "bench_matmul", "bench_attention"]  # fmt: skip


def main() -> None:
    caps = enceladus.get_device().caps
    date = datetime.date.today().isoformat()
    out = HERE / "results" / f"{date}-{caps.architecture}.md"
    out.parent.mkdir(exist_ok=True)
    lines = [
        f"# Enceladus benchmarks, {date}",
        "",
        f"Device: {caps.name} ({caps.architecture}). Enceladus {enceladus.__version__}.",
        "Enceladus times use GPU timestamps; MLX and PyTorch times use the wall clock.",
        "",
    ]
    for name in BENCHMARKS:
        print(f"running {name}...", flush=True)
        r = subprocess.run([sys.executable, str(HERE / f"{name}.py")], capture_output=True,
                           text=True)  # fmt: skip
        lines += [f"## {name}", "", "```text", (r.stdout + r.stderr).strip(), "```", ""]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
