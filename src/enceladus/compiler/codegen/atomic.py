"""Atomic lowering: `atomic_rmw` and `atomic_cas` on pointers and pointer tiles.

Metal's native device atomics cover these cases:

- `int` and `uint`: add, max, min, exchange, and, or, and xor.
- `float`: add and exchange, through `atomic_float`.
- `ulong`: max and min only, with no old value, on Apple9 and later. These are Metal's
  only 64-bit atomics: there's no `atomic_long` and no 64-bit exchange, fetch, or
  compare-and-swap, so the frontend refuses every other 64-bit atomic.

Everything else (float max and min, compare-and-swap, and 8-bit and 16-bit elements)
runs as a compare-and-swap loop on the aligned 32-bit word that holds the element. The
frontend refuses what neither path covers, such as `float16` addition.

Each element runs its atomic exactly once. A tile layout whose lane or SIMD-group bits
broadcast (several threads hold the same element) runs the atomic only in the owning
thread, then sends the old value to the other threads. A scalar atomic runs in thread 0
of the threadgroup, which then shares the old value through threadgroup memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from enceladus.compiler import ir
from enceladus.compiler import layout as L
from enceladus.compiler.codegen.msl import BARRIER, CTYPES, Tile, _add, literal
from enceladus.compiler.codegen.scan import shfl
from enceladus.compiler.errors import internal_error

if TYPE_CHECKING:
    from enceladus.compiler.codegen.msl import _Codegen

MO = "memory_order_relaxed"
_NATIVE_INT = ("add", "max", "min", "and", "or", "xor", "xchg")
_BITWISE = {"and": "&", "or": "|", "xor": "^"}
_WORD = {8: "uchar", 16: "ushort", 32: "uint"}


def _native_fn(kind: str) -> str:
    return "atomic_exchange_explicit" if kind == "xchg" else f"atomic_fetch_{kind}_explicit"


def _cas_loop_helper(cg: _Codegen, kind: str, elem: ir.ScalarType) -> str:
    """Emits a compare-and-swap loop helper for `kind` on `elem` and returns its name.

    The loop works on the aligned 32-bit word that holds the element, so it also covers
    8-bit and 16-bit elements. It compares bit patterns, so it ends even for NaNs.
    """
    ct = CTYPES[elem.name]
    bits = elem.dtype.primitive_bitwidth
    ut = _WORD[bits]
    name = f"tg_atom_{kind}_{ct}"
    fp = elem.dtype.is_floating()
    if kind == "add":
        comb = f"{ct}(o + v)"
    elif kind in ("max", "min"):
        comb = f"{ct}(f{kind}(float(o), float(v)))" if fp else f"{ct}({kind}(o, v))"
    elif kind == "xchg":
        comb = "v"
    elif kind in _BITWISE:
        comb = f"{ct}(o {_BITWISE[kind]} v)"
    else:  # cas
        comb = f"(as_type<{ut}>(o) == as_type<{ut}>(c) ? v : o)"
    params = f"device {ct}* p, {ct} c, {ct} v" if kind == "cas" else f"device {ct}* p, {ct} v"
    cas = f"atomic_compare_exchange_weak_explicit(w, &old, nw, {MO}, {MO})"
    if bits == 32:
        body = [
            "  device atomic_uint* w = (device atomic_uint*)p;",
            f"  uint old = atomic_load_explicit(w, {MO});",
            "  while (true) {",
            f"    const {ct} o = as_type<{ct}>(old);",
            f"    const uint nw = as_type<uint>({ct}({comb}));",
            f"    if (nw == old || {cas}) return o;",
            "  }",
        ]
    else:
        mask = (1 << bits) - 1
        body = [
            "  const ulong a = ulong(p);",
            "  device atomic_uint* w = (device atomic_uint*)((device uchar*)p - (a & 3ul));",
            "  const uint sh = uint(a & 3ul) * 8u;",
            f"  uint old = atomic_load_explicit(w, {MO});",
            "  while (true) {",
            f"    const {ct} o = as_type<{ct}>({ut}(old >> sh));",
            f"    const {ct} n = {comb};",
            f"    const uint nw = (old & ~({mask}u << sh)) | (uint(as_type<{ut}>(n)) << sh);",
            f"    if (nw == old || {cas}) return o;",
            "  }",
        ]
    code = "\n".join([f"static inline {ct} {name}({params}) {{", *body, "}"])
    cg.helper(name, code)
    return name


def _apple_family(cg: _Codegen) -> int:
    fam = cg.m.attrs.get("apple_family")
    if fam is None:
        try:
            from enceladus.runtime.device import get_device

            fam = get_device().caps.apple_family
        except Exception:  # noqa: BLE001 - no device means no 64-bit atomics
            fam = 0
    return int(fam)


def atomic_call(cg: _Codegen, kind: str, elem: ir.ScalarType, addr: str, val: str,
                cmp: str | None, used: bool) -> str:  # fmt: skip
    """Returns the MSL expression that runs one atomic and yields the old value."""
    n = elem.name
    if n in ("i32", "u32") and kind in _NATIVE_INT:
        at = "atomic_int" if n == "i32" else "atomic_uint"
        return f"{_native_fn(kind)}((device {at}*)({addr}), {val}, {MO})"
    if n == "f32" and kind in ("add", "xchg"):
        return f"{_native_fn(kind)}((device atomic_float*)({addr}), {val}, {MO})"
    if n == "u64":
        # The frontend allows only max and min here.
        if used:
            raise cg.err(
                f"tl.atomic_{kind} on uint64 can't return the old value, because Metal's "
                f"64-bit atomic_{kind}_explicit returns nothing and Metal has no 64-bit "
                "compare-and-swap to build a returning version from. Don't use the result, "
                "or use uint32 elements."
            )
        if _apple_family(cg) < 9:
            raise cg.err(
                f"tl.atomic_{kind} on uint64 needs an Apple9 GPU (M3 or later). Use uint32 "
                "elements on this GPU."
            )
        return f"atomic_{kind}_explicit((device atomic_ulong*)({addr}), {val}, {MO})"
    if n in ("i64", "u64", "i1"):
        raise internal_error(f"atomic `{kind}` on {n} passed the frontend", cg.loc)
    helper = _cas_loop_helper(cg, kind, elem)
    return f"{helper}({addr}, {cmp}, {val})" if kind == "cas" else f"{helper}({addr}, {val})"


def emit_atomic(cg: _Codegen, op: ir.Op) -> None:
    kind = "cas" if op.name == "atomic_cas" else op.attrs["op"]
    nvals = 2 if kind == "cas" else 1
    ptr, res = op.operands[0], op.result
    vals = op.operands[1 : 1 + nvals]
    mask = op.operands[1 + nvals] if len(op.operands) > 1 + nvals else None
    elem = ir.elem_of(ptr.type).elem
    ct = CTYPES[elem.name]
    used = cg.uses.get(id(res), 0) > 0
    zero = literal(0, elem)
    if not isinstance(res.type, ir.TileType):
        _emit_scalar(cg, kind, elem, ptr, vals, mask, res, used, ct, zero)
        return
    lay = cg.plan.layout_of(res)
    p = cg.mat(ptr, lay)
    vt = [cg.mat(v, lay) for v in vals]
    mt = cg.mat(mask, lay) if mask is not None else None
    if p.root:
        cg.written.add(p.root)
    own = cg.owner(lay)
    n = lay.num_regs
    arr = cg.declare(res.type, lay, res.name_hint or "old") if used else None
    masked = mt is not None and mt.uniform != "true"
    if arr is not None and (masked or own != "true"):
        cg.loop(n, f"{arr}[{{r}}] = {zero};")
    call = atomic_call(cg, kind, elem, f"{p.base} + {p.get('{r}')}", vt[-1].get("{r}"),
                       vt[0].get("{r}") if kind == "cas" else None, used)  # fmt: skip
    stmt = f"{arr}[{{r}}] = {call};" if arr is not None else f"{call};"
    if masked:
        stmt = f"if ({mt.get('{r}')}) {stmt}"
    if own == "true":
        cg.loop(n, stmt)
    else:
        with cg.e.block(f"if ({own})"):
            cg.loop(n, stmt)
    if arr is None:
        return
    if own != "true":
        _share_from_owner(cg, arr, lay, ct, own)
    cg.tiles[id(res)] = Tile(lay, arr)


def _emit_scalar(cg: _Codegen, kind, elem, ptr, vals, mask, res, used, ct, zero) -> None:
    root = cg.roots.get(id(ptr))
    if root:
        cg.written.add(root)
    args = [cg.s(v) for v in vals]
    call = atomic_call(cg, kind, elem, cg.s(ptr), args[-1], args[0] if kind == "cas" else None,
                       used)  # fmt: skip
    cond = "lane == 0 && warp == 0"
    m = cg.s(mask) if mask is not None else None
    if not used:
        if m is not None:
            cond += f" && {m}"
        cg.e.line(f"if ({cond}) {call};")
        return
    name = cg.fresh(res.name_hint or "old")
    cg.e.line(f"{ct} {name};")
    cg.use_tg(max(4, elem.dtype.itemsize))
    cg.e.line(BARRIER)
    with cg.e.block(""):
        bc = cg.fresh("bc")
        cg.e.line(f"threadgroup {ct}* {bc} = (threadgroup {ct}*)tg_mem;")
        rhs = call if m is None else f"{m} ? {call} : {zero}"
        cg.e.line(f"if ({cond}) {bc}[0] = {rhs};")
        cg.e.line(BARRIER)
        cg.e.line(f"{name} = {bc}[0];")
    cg.sv[id(res)] = name


def _share_from_owner(cg: _Codegen, arr: str, lay: L.BitLayout, ct: str, own: str) -> None:
    """Copies each element's old value from its owning thread to the threads that share it."""
    lm, wm = L.owner_mask(lay)
    n = lay.num_regs
    if not wm:
        src = cg.pro(f"ownlane|{lm}", "ol", "uint", f"(lane & ~{lm}u)")
        cg.loop(n, f"{arr}[{{r}}] = {shfl(cg, f'{arr}[{{r}}]', src)};")
        return
    strides, acc = [], 1
    for d in reversed(range(len(lay.shape))):
        strides.append(acc)
        acc *= lay.shape[d]
    flat, consts = cg.flat(lay, tuple(reversed(strides)))
    cg.use_tg(acc * max(1, _size(ct)))
    cg.e.line(BARRIER)
    with cg.e.block(""):
        buf = cg.fresh("ob")
        cg.e.line(f"threadgroup {ct}* {buf} = (threadgroup {ct}*)tg_mem;")
        with cg.e.block(f"if ({own})"):
            for r in range(n):
                cg.e.line(f"{buf}[{_add(flat, consts[r])}] = {arr}[{r}];")
        cg.e.line(BARRIER)
        for r in range(n):
            cg.e.line(f"{arr}[{r}] = {buf}[{_add(flat, consts[r])}];")


def _size(ct: str) -> int:
    return {"char": 1, "uchar": 1, "short": 2, "ushort": 2, "half": 2, "bfloat": 2,
            "long": 8, "ulong": 8}.get(ct, 4)  # fmt: skip
