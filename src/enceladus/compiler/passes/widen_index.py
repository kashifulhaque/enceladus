"""Index widening for kernels that address arrays past 2^31 elements (`idx64`).

When an array argument spans more than 2^31 - 1 elements, the launcher sets the `idx64`
specialization fact on it. For such a kernel, this pass recomputes in 64 bits the
signed 32-bit (or narrower) integer math that feeds pointer offsets, together with the
comparisons that read it, such as the mask `offs < n`. The address and the mask then
agree on the exact offset. For example, `pid * BLOCK + tl.arange(0, BLOCK)` doesn't wrap
around at 2^31. The results differ from 32-bit math only where that math overflows,
which is undefined behavior for signed integers in Metal.

The pass works on the *index slice*: the values that pointer offsets, and the
comparisons that read the slice, compute from. Within it:

- A signed 32-bit value computed by integer arithmetic or a shape op gets a 64-bit
  clone of its op, placed right after the original.
- Any other signed 32-bit value, such as `tl.program_id`, a load, a loop counter, or a
  value carried by a loop, gets a cast to 64 bits right after its definition. Index math
  that a loop carries in 32 bits across iterations therefore still wraps; carry a pointer
  or a `tl.int64` offset instead.
- A cast from a signed 32-bit value to 64 bits, including the one that type promotion
  inserts for `offs < n` with a 64-bit `n`, reads the 64-bit version instead.

Unsigned and 64-bit values keep their semantics. Uses of 32-bit values outside the
slice, such as a store of `offs` as data, keep their 32-bit results. DCE removes
originals that nothing uses anymore.
"""

from __future__ import annotations

from enceladus.compiler import ir
from enceladus.compiler.errors import CompilationError

NARROW_SIGNED = frozenset(["i8", "i16", "i32"])
WIDE_SIGNED = frozenset(["i64"])
# Ops whose clone computes the exact value of the original from 64-bit operands.
CLONED = frozenset(
    ["binary", "unary", "fma", "select", "splat", "broadcast", "expand_dims", "reshape",
     "trans", "hint", "const", "full"]
)  # fmt: skip
_POINTER_VIEWS = frozenset(["addptr", "splat", "broadcast", "expand_dims", "reshape", "trans",
                            "hint"])  # fmt: skip


def needs_idx64(module: ir.Module) -> bool:
    """Returns whether some argument of `module` has the `idx64` specialization fact."""
    return any(a.get("idx64") for a in module.func.attrs.get("arg_attrs", []))


def _kind(v: ir.Value) -> str | None:
    """Returns "narrow" or "wide" for a signed integer value, and None otherwise."""
    e = ir.elem_of(v.type)
    if isinstance(e, ir.ScalarType):
        if e.name in NARROW_SIGNED:
            return "narrow"
        if e.name in WIDE_SIGNED:
            return "wide"
    return None


def _widening_cast(v: ir.Value) -> ir.Value | None:
    """Returns `x` when `v` is a cast of a signed 32-bit-or-narrower `x` to 64 bits."""
    op = v.defining_op
    if op is not None and op.name == "cast" and _kind(v) == "wide" and \
            _kind(op.operands[0]) == "narrow":  # fmt: skip
        return op.operands[0]
    return None


def _root_arg(v: ir.Value, body: ir.Block) -> int | None:
    """Returns the index of the kernel argument that pointer `v` derives from, if any."""
    while True:
        if v.owner is body:
            return v.index
        op = v.defining_op
        if op is None or op.name not in _POINTER_VIEWS:
            return None
        v = op.operands[0]


def _check_descriptors(module: ir.Module) -> None:
    attrs = module.func.attrs.get("arg_attrs", [])
    names = module.func.attrs.get("arg_names", [])
    for op in module.walk():
        if op.name != "make_desc":
            continue
        i = _root_arg(op.operands[0], module.body)
        if i is not None and i < len(attrs) and attrs[i].get("idx64"):
            raise CompilationError(
                f"argument `{names[i]}` spans more than 2^31 elements, and tensor descriptors "
                "address at most 2^31 elements. Split the array, or load it through a "
                "pointer tile instead.",
                op.loc,
            )


def widen_index_math(module: ir.Module) -> int:
    """Widens 32-bit index math in an `idx64` module. Returns the number of values widened.

    Raises:
        CompilationError: A tensor descriptor covers an `idx64` argument. Descriptor
            addressing computes in 32 bits.
    """
    _check_descriptors(module)
    need: dict[int, ir.Value] = {}  # narrow values that get a 64-bit version
    seen: set[int] = set()

    def visit(v: ir.Value) -> None:
        k = _kind(v)
        if k is None or id(v) in seen:
            return
        seen.add(id(v))
        if k == "narrow":
            need[id(v)] = v
        x = _widening_cast(v)
        if x is not None:
            visit(x)
            return
        op = v.defining_op
        if op is not None and op.name in CLONED:
            for x in op.operands[1:] if op.name == "select" else op.operands:
                visit(x)

    def in_slice(v: ir.Value) -> bool:
        x = _widening_cast(v)
        return id(v) in seen or (x is not None and id(x) in seen)

    cmps: list[ir.Op] = []
    for op in module.walk():
        if op.name == "addptr":
            visit(op.operands[1])
        elif op.name == "cmp":
            cmps.append(op)
    marked: set[int] = set()
    grew = True
    while grew:
        grew = False
        for op in cmps:
            if id(op) not in marked and any(in_slice(v) for v in op.operands):
                marked.add(id(op))
                for v in op.operands:
                    visit(v)
                grew = True
    if not need:
        return 0

    wide: dict[int, ir.Value] = {}  # narrow value -> its 64-bit version
    alias: dict[int, ir.Value] = {}  # widening cast in the slice -> the 64-bit version

    def cast(v: ir.Value, loc) -> ir.Op:
        op = ir.Op("cast", [v], [ir.with_elem(v.type, ir.i64)], loc=loc)
        op.result.name_hint = v.name_hint
        wide[id(v)] = op.result
        return op

    def block(b: ir.Block, owner: ir.Op) -> None:
        out: list[ir.Op] = [cast(a, owner.loc) for a in b.args if id(a) in need]
        for op in b.ops:
            if op.name == "addptr" or id(op) in marked:
                op.operands = [wide.get(id(v), v) for v in op.operands]
            op.operands = [alias.get(id(v), v) for v in op.operands]
            for r in op.regions:
                block(r.block, op)
            out.append(op)
            for r in op.results:
                x = _widening_cast(r)
                if x is not None and id(x) in wide:
                    alias[id(r)] = wide[id(x)]
                if id(r) not in need:
                    continue
                if op.name in CLONED:
                    ops = [wide.get(id(v), v) for v in op.operands]
                    clone = ir.Op(op.name, ops, [ir.with_elem(r.type, ir.i64)], dict(op.attrs),
                                  loc=op.loc)  # fmt: skip
                    clone.result.name_hint = r.name_hint
                    wide[id(r)] = clone.result
                    out.append(clone)
                else:
                    out.append(cast(r, op.loc))
        for op in out:
            op.parent = b
        b.ops = out

    block(module.body, module.func)
    return len(need)
