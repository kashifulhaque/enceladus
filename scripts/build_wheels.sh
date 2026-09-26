#!/usr/bin/env bash
# Builds the macosx_15_0_arm64 release wheels for CPython 3.11 to 3.14, checks
# their metadata, and runs the quickstart against each installed wheel, outside the
# source tree.
#
# Usage: scripts/build_wheels.sh [OUT_DIR]   (OUT_DIR defaults to dist/)
#
# Each wheel is version-specific (cp311 to cp314). nanobind's stable-ABI mode
# isn't used; docs/progress.md records why.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/dist}" && cd "${1:-$ROOT/dist}" && pwd)"
PYTHONS=(3.11 3.12 3.13 3.14)

cd "$ROOT"
for py in "${PYTHONS[@]}"; do
  uv build --wheel --python "$py" --out-dir "$OUT"
done

# Check the tag, metadata, and contents of each wheel.
uv run --no-project --python 3.13 python - "$OUT" <<'EOF'
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

out = Path(sys.argv[1])
for tag in ("cp311-cp311", "cp312-cp312", "cp313-cp313", "cp314-cp314"):
    wheels = sorted(out.glob(f"enceladus-*-{tag}-macosx_15_0_arm64.whl"))
    assert wheels, f"no {tag} wheel in {out}"
    whl = wheels[-1]
    with zipfile.ZipFile(whl) as z:
        names = z.namelist()
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        meta = Parser().parsestr(z.read(meta_name).decode())
    assert meta["Name"] == "enceladus", meta["Name"]
    assert meta["Requires-Python"] == ">=3.11", meta["Requires-Python"]
    extras = set(meta.get_all("Provides-Extra") or [])
    assert {"torch", "mlx"} <= extras, extras
    assert "enceladus/compiler/codegen/prelude.metal" in names, "prelude.metal is missing"
    assert any(n.startswith("enceladus/_C.") and n.endswith(".so") for n in names)
    bad = [n for n in names if n.startswith(("tests/", "examples/", "benchmarks/", "docs/"))
           or n.endswith((".mm", ".h"))]
    assert not bad, f"unexpected files: {bad}"
    print(f"{whl.name}: metadata and contents OK ({len(names)} files)")
EOF

# Run the quickstart against each installed wheel. The snippets run from a temporary
# directory, so `import enceladus` resolves to the wheel, not to src/.
for py in "${PYTHONS[@]}"; do
  tag="cp${py/./}-cp${py/./}"
  whl="$(ls "$OUT"/enceladus-*-"$tag"-macosx_15_0_arm64.whl | tail -n 1)"
  echo "== Python $py: $(basename "$whl")"
  (cd "$(mktemp -d)" &&
    uv run --isolated --no-project --python "$py" --with "$whl" \
      python -c "import enceladus; print('using', enceladus.__file__)" &&
    uv run --isolated --no-project --python "$py" --with "$whl" \
      python "$ROOT/docs/guide/check_snippets.py" "$ROOT/docs/guide/quickstart.md")
done
