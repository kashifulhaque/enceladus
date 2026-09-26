"""Common-subexpression elimination and dead-code elimination."""

from __future__ import annotations

from typing import Any

from enceladus.compiler import ir

# Ops with no side effects whose results depend only on operands and attributes.
PURE_OPS = frozenset(
    """const splat arange full program_id num_programs binary cmp unary fma select cast
    bitcast broadcast expand_dims reshape trans addptr make_desc""".split()
)
# Ops that are safe to delete when their results are unused.
REMOVABLE_OPS = PURE_OPS | {"load", "dot", "reduce", "scan", "desc_load", "local_load"}


def _freeze(v: Any) -> Any:
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, float) and v != v:
        return "nan"
    return v


def cse(module: ir.Module) -> int:
    """Merges identical pure ops. Returns the number of ops removed."""
    replace: dict[int, ir.Value] = {}
    removed = 0

    def run(block: ir.Block, scope: list[dict]) -> None:
        nonlocal removed
        table: dict = {}
        scope.append(table)
        kept = []
        for op in block.ops:
            op.operands = [replace.get(id(v), v) for v in op.operands]
            for r in op.regions:
                run(r.block, scope)
            if op.name in PURE_OPS and not op.regions:
                key = (
                    op.name,
                    tuple(id(v) for v in op.operands),
                    _freeze(op.attrs),
                    tuple(str(r.type) for r in op.results),
                )
                prev = next((t[key] for t in reversed(scope) if key in t), None)
                if prev is not None:
                    for a, b in zip(op.results, prev.results, strict=True):
                        replace[id(a)] = b
                        if b.name_hint is None:
                            b.name_hint = a.name_hint
                    removed += 1
                    continue
                table[key] = op
            kept.append(op)
        block.ops = kept
        scope.pop()

    run(module.body, [])
    return removed


def dce(module: ir.Module) -> int:
    """Deletes removable ops whose results are unused. Returns the number removed."""
    total = 0
    while True:
        used: set[int] = set()
        for op in module.walk():
            used.update(id(v) for v in op.operands)
        removed = 0

        def run(block: ir.Block, used: set[int] = used) -> None:
            nonlocal removed
            kept = []
            for op in block.ops:
                for r in op.regions:
                    run(r.block)
                if op.name in REMOVABLE_OPS and not any(id(v) in used for v in op.results):
                    removed += 1
                    continue
                kept.append(op)
            block.ops = kept

        run(module.body)
        total += removed
        if not removed:
            return total


def simplify(module: ir.Module) -> None:
    """Runs CSE, then DCE."""
    cse(module)
    dce(module)
