"""Edge versioning: split a `tl.dot` loop into interior and edge versions.

A direct `tl.dot` operand (see `codegen.dot`) reads its fragments straight from device
memory. Without versioning, every `dot` tests whether its whole block is in bounds and
branches to unmasked or checked loads. This pass finds the `for` loops where that test
can move out of the loop:

- Every direct operand of a covered `dot` loads from a descriptor defined before the
  loop, at offsets that are either loop-invariant or the loop counter plus a
  loop-invariant value.
- The loop counts up by a constant step with a 32-bit counter.

Codegen then emits the loop three times:

```
if (every invariant offset is in bounds && the first iteration's offsets are >= 0) {
  for (i = lb; i < k_end; i += step) { unmasked loads }  // full blocks along K
  for (; i < ub; i += step) { checked loads }             // the ragged K tail
} else {
  for (i = lb; i < ub; i += step) { checked loads }       // edge blocks along M or N
}
```

`k_end` is the first counter value at which some counter-dependent block would cross
the end of its dimension, so the tail is empty when K is a multiple of the block size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from enceladus.compiler import ir


@dataclass
class Bound:
    """One dimension of one direct operand's block.

    Attributes:
        desc: The descriptor value, defined before the loop.
        dim: The descriptor dimension.
        offset: The loop-invariant part of the offset: a value defined before the loop,
            or None for 0.
        uses_counter: Whether the loop counter is added to `offset`.
    """

    desc: ir.Value
    dim: int
    offset: ir.Value | None
    uses_counter: bool


@dataclass
class LoopPlan:
    """How to version one loop: the dots it covers and the bounds that decide the version."""

    dots: set[int] = field(default_factory=set)
    bounds: list[Bound] = field(default_factory=list)


def _inside(v: ir.Value, loop: ir.Op) -> bool:
    """Returns whether `v` is defined in the body of `loop`, at any depth."""
    owner = v.owner
    blk = owner if isinstance(owner, ir.Block) else owner.parent
    while blk is not None:
        region = blk.parent
        parent = region.parent if region is not None else None
        if parent is loop:
            return True
        if parent is None:
            return False
        blk = parent.parent
    return False


def _innermost_loop(op: ir.Op) -> ir.Op | None:
    blk = op.parent
    while blk is not None:
        region = blk.parent
        parent = region.parent if region is not None else None
        if parent is None:
            return None
        if parent.name == "for":
            return parent
        blk = parent.parent
    return None


def _bounds(load: ir.Op, loop: ir.Op) -> list[Bound] | None:
    """Returns the bounds of a direct `desc_load` in `loop`, or None if they vary otherwise."""
    desc, *offsets = load.operands
    if _inside(desc, loop):
        return None
    iv = loop.regions[0].block.args[0]
    out = []
    for d, o in enumerate(offsets):
        if o is iv:
            out.append(Bound(desc, d, None, True))
        elif not _inside(o, loop):
            out.append(Bound(desc, d, o, False))
        else:
            op = o.defining_op
            if op is None or op.name != "binary" or op.attrs["op"] != "add" or o.type != iv.type:
                return None
            x, y = op.operands
            inv = y if x is iv else x if y is iv else None
            if inv is None or _inside(inv, loop):
                return None
            out.append(Bound(desc, d, inv, True))
    return out


def plan_edge_versioning(module: ir.Module, direct: set[int],
                         skip: set[int] = frozenset()) -> dict[int, LoopPlan]:  # fmt: skip
    """Returns a `LoopPlan` for each `for` loop whose dots can use edge versioning.

    Args:
        module: The kernel, after `simplify` has hoisted loop-invariant scalars.
        direct: The ids of the values that dots read from device memory
            (`codegen.dot.find_direct_operands`).
        skip: The ids of ops that another lowering handles, such as Metal 4 `matmul2d`.

    Returns:
        A dict from the id of each loop to version to its plan. A `dot` is covered only
        by its innermost enclosing loop.
    """
    plans: dict[int, LoopPlan] = {}
    refused: set[int] = set()
    for op in module.walk():
        if op.name != "dot" or id(op) in skip:
            continue
        loop = _innermost_loop(op)
        if loop is None or id(loop) in refused or id(loop) in skip:
            continue
        lb, ub, step = loop.operands[:3]
        step_op = step.defining_op
        if lb.type != ir.i32 or step_op is None or step_op.name != "const" or \
                step_op.attrs["value"] <= 0 or \
                any(o.name in ("print", "assert") for o in loop.walk()):  # fmt: skip
            refused.add(id(loop))
            continue
        bounds: list[Bound] = []
        loads = []
        for v in op.operands[:2]:
            if id(v) not in direct:
                continue
            src = v.defining_op
            if src.name == "trans":
                src = src.operands[0].defining_op
            loads.append(src)
        for load in loads:
            b = _bounds(load, loop)
            if b is None:
                break
            bounds += b
        else:
            if loads:
                plan = plans.setdefault(id(loop), LoopPlan())
                plan.dots.add(id(op))
                plan.bounds += bounds
    return plans
