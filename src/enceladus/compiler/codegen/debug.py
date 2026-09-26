"""MSL lowering of `tl.device_print` and `tl.device_assert`, and their shared formats.

`print` lowers to `os_log_default.log(...)` from `<metal_logging>`. The kernel then
needs `MTLCompileOptions.enableLogging` and language version 3.2 or later, which the
codegen reports through `GeneratedKernel.enable_logging` and `language_version`. Each
line starts with the program ID; tile lines also carry the element's index. Each thread
prints the elements it owns, and threads that hold broadcast copies stay silent, as
they do for stores.

`assert` lowers to a check that records the first failure in an error buffer, an extra
kernel argument after the runtime arguments. The runtime reads the buffer at the next
sync. The compiler emits `assert` ops only when `ENCELADUS_DEBUG=1`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from enceladus.compiler import ir
from enceladus.compiler import layout as L

if TYPE_CHECKING:
    from enceladus.compiler.codegen.msl import _Codegen

# IR scalar name -> (os_log conversion, C cast applied to the value).
PRINT_FORMATS = {
    "i1": ("%d", "int"), "i8": ("%d", "int"), "i16": ("%d", "int"), "i32": ("%d", "int"),
    "u8": ("%u", "uint"), "u16": ("%u", "uint"), "u32": ("%u", "uint"),
    "i64": ("%ld", "long"), "u64": ("%lu", "ulong"),
    "f16": ("%f", "float"), "bf16": ("%f", "float"), "f32": ("%f", "float"),
}  # fmt: skip

LOGGING_LANGUAGE_VERSION = (3, 2)
"""The lowest MSL version that supports `<metal_logging>`."""

ASSERT_BUFFER = "tg_assert_buf"
ASSERT_WORDS = 5
"""Words in the error buffer: failed flag, assert index, and the program ID (x, y, z)."""

_ASSERT_HELPER = """\
static void tg_assert_fail(device atomic_uint* e, uint index, uint3 p) {
  uint expected = 0;
  while (!atomic_compare_exchange_weak_explicit(&e[0], &expected, 1u, memory_order_relaxed,
                                                memory_order_relaxed)) {
    if (expected != 0) return;  // another failure came first
  }
  atomic_store_explicit(&e[1], index, memory_order_relaxed);
  atomic_store_explicit(&e[2], p.x, memory_order_relaxed);
  atomic_store_explicit(&e[3], p.y, memory_order_relaxed);
  atomic_store_explicit(&e[4], p.z, memory_order_relaxed);
}"""


def print_prefix(prefix: str) -> str:
    """Returns the text that precedes the values on each line: `"x: "` for `"x"`."""
    p = prefix.rstrip()
    if not p:
        return ""
    return p + " " if p.endswith(":") else p + ": "


def format_value(value: Any, dtype_name: str) -> str:
    """Formats one element the way the GPU's `os_log` conversion does."""
    conv = PRINT_FORMATS[dtype_name][0]
    if conv == "%f":
        return f"{float(value):f}"
    return str(int(value))


def format_line(pid: tuple[int, int, int], idx: tuple[int, ...] | None, prefix: str,
                values: list[str]) -> str:  # fmt: skip
    """Returns one printed line. `idx` is None for a print of scalars."""
    head = f"pid ({pid[0]}, {pid[1]}, {pid[2]})"
    if idx is not None and idx:
        head += f" idx ({', '.join(map(str, idx))})"
    return f"{head} {print_prefix(prefix)}{' '.join(values)}".rstrip()


def _c_string(text: str) -> str:
    """Returns `text` as the body of an MSL format string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")


def emit_print(cg: _Codegen, op: ir.Op) -> None:
    """Emits one `os_log` call per element that the thread owns."""
    cg.require_language_version(LOGGING_LANGUAGE_VERSION)
    cg.enable_logging = True
    cg.helper("metal_logging", "#include <metal_logging>")
    prefix = _c_string(print_prefix(op.attrs["prefix"]))
    convs = [PRINT_FORMATS[ir.elem_of(v.type).name] for v in op.operands]
    vals_fmt = " ".join(c for c, _ in convs)
    pid_args = "int(pid.x), int(pid.y), int(pid.z)"
    tiles = [v for v in op.operands if isinstance(v.type, ir.TileType)]
    if not tiles or not ir.shape_of(tiles[0].type):
        exprs = []
        for v, (_, cast) in zip(op.operands, convs, strict=True):
            e = cg.mat(v, _layout(cg, op, v)).get(0) if isinstance(v.type, ir.TileType) \
                else cg.s(v)  # fmt: skip
            exprs.append(f"{cast}({e})")
        fmt = f"pid (%d, %d, %d) {prefix}{vals_fmt}".rstrip()
        args = ", ".join([pid_args, *exprs])
        cg.e.line(f'if (lane == 0 && warp == 0) os_log_default.log("{fmt}", {args});')
        return
    lay = _layout(cg, op, tiles[0])
    rank = len(lay.shape)
    ins = [cg.mat(v, lay) if isinstance(v.type, ir.TileType) else cg.s(v) for v in op.operands]
    idx_fmt = ", ".join(["%d"] * rank)
    fmt = f"pid (%d, %d, %d) idx ({idx_fmt}) {prefix}{vals_fmt}".rstrip()
    own = cg.owner(lay)
    with cg.e.block(f"if ({own})" if own != "true" else ""):
        for r in range(lay.num_regs):
            coords = [cg.coord(lay, r, d) for d in range(rank)]
            vals = [f"{cast}({x if isinstance(x, str) else x.get(r)})"
                    for x, (_, cast) in zip(ins, convs, strict=True)]  # fmt: skip
            args = ", ".join([pid_args, *coords, *vals])
            cg.e.line(f'os_log_default.log("{fmt}", {args});')


def emit_assert(cg: _Codegen, op: ir.Op) -> None:
    """Emits a check that records the first failing program in the error buffer."""
    index = len(cg.asserts)
    loc = op.loc
    cg.asserts.append({"message": op.attrs["msg"], "file": loc.file if loc else "",
                       "line": loc.line if loc else 0, "col": loc.col if loc else 0})  # fmt: skip
    cg.helper("assert", _ASSERT_HELPER)
    call = f"tg_assert_fail({ASSERT_BUFFER}, {index}u, pid);"
    cond = op.operands[0]
    mask = op.operands[1] if len(op.operands) > 1 else None
    if not isinstance(cond.type, ir.TileType):
        c = cg.s(cond)
        if mask is not None:
            c = f"{c} || !{cg.s(mask)}"
        cg.e.line(f"if (!({c})) {call}")
        return
    lay = cg.plan.natural(cond) or cg.plan.default(cond.type)
    ct = cg.mat(cond, lay)
    mt = cg.mat(mask, lay) if mask is not None else None
    if ct.uniform is not None and (mt is None or mt.uniform is not None):
        c = ct.uniform if mt is None else f"{ct.uniform} || !{mt.uniform}"
        cg.e.line(f"if (!({c})) {call}")
        return
    ok = cg.fresh("assert_ok")
    cg.e.line(f"bool {ok} = true;")
    if mt is None:
        body = f"{ok} &= bool({ct.get('{r}')});"
    else:
        body = f"{ok} &= bool({ct.get('{r}')}) || !{mt.get('{r}')};"
    cg.loop(lay.num_regs, body)
    cg.e.line(f"if (!{ok}) {call}")


def _layout(cg: _Codegen, op: ir.Op, v: ir.Value) -> L.BitLayout:
    """Returns the layout a print reads its tiles in: the first natural one, or blocked."""
    for x in op.operands:
        if isinstance(x.type, ir.TileType):
            lay = cg.plan.natural(x)
            if lay is not None:
                return lay
    return cg.plan.default(v.type)
