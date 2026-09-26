"""MSL code generation: turn layout-assigned Tegula IR into one Metal kernel.

Each tile value becomes a per-thread register array (`T v[R]`, where `R` is the number
of registers in its layout). Elementwise ops loop over registers with compile-time trip
counts. Scalars (program IDs, arguments, loop counters, full reductions) are plain
locals and are uniform across the threadgroup, so control flow on them never diverges.

Cheap and view values (see `passes.layouts`) are emitted lazily, in the layout each use
needs. A use that needs an anchored value in a different layout gets a register remap
when the data is already in the same thread, and a threadgroup-memory exchange
otherwise.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from tegula.compiler import ir
from tegula.compiler import layout as L
from tegula.compiler.codegen.emitter import Emitter, NameGen
from tegula.compiler.errors import CompilationError, Loc
from tegula.compiler.passes.layouts import (
    ANCHORED,
    CHEAP,
    VIEWS,
    LayoutPlan,
    store_layout,
    view_source_layout,
)

PRELUDE = (Path(__file__).parent / "prelude.metal").read_text()

CTYPES = {
    "i1": "bool", "i8": "char", "i16": "short", "i32": "int", "i64": "long",
    "u8": "uchar", "u16": "ushort", "u32": "uint", "u64": "ulong",
    "f16": "half", "bf16": "bfloat", "f32": "float",
}  # fmt: skip
STRUCT_FORMATS = {
    "i1": "?", "i8": "b", "i16": "h", "i32": "i", "i64": "q",
    "u8": "B", "u16": "H", "u32": "I", "u64": "Q", "f16": "e", "f32": "f",
}  # fmt: skip
MATH_FUNCS = {
    "exp": "exp", "exp2": "exp2", "log": "log", "log2": "log2", "sqrt": "sqrt",
    "rsqrt": "rsqrt", "sin": "sin", "cos": "cos", "tanh": "tanh", "sigmoid": "tg_sigmoid",
    "erf": "tg_erf", "floor": "floor", "ceil": "ceil",
}  # fmt: skip
BIN_SYMBOLS = {
    "add": "+", "sub": "-", "mul": "*", "div": "/", "floordiv": "/", "mod": "%",
    "and": "&", "or": "|", "xor": "^", "shl": "<<", "shr": ">>",
}  # fmt: skip
CMP_SYMBOLS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
BARRIER = "threadgroup_barrier(mem_flags::mem_threadgroup);"
MAX_REGS_WARN, MAX_REGS_ERROR = 128, 256


def ctype(t: ir.Type) -> str:
    e = ir.elem_of(t)
    if isinstance(e, ir.PointerType):
        return "int"  # pointer tiles hold offsets
    return CTYPES[e.name]


def is_half(t: ir.ScalarType) -> bool:
    return t.name in ("f16", "bf16")


def is_float(t: ir.ScalarType) -> bool:
    return t.name in ("f16", "bf16", "f32")


def literal(value, t: ir.ScalarType) -> str:
    """Formats a Python constant as an MSL literal of type `t`."""
    n = t.name
    if n == "i1":
        return "true" if value else "false"
    if is_float(t):
        v = float(value)
        if math.isnan(v):
            s = "NAN"
        elif math.isinf(v):
            s = "INFINITY" if v > 0 else "(-INFINITY)"
        else:
            s = repr(v) + "f"
        return s if n == "f32" else f"{CTYPES[n]}({s})"
    v = int(value)
    if n == "i32":
        return "(-2147483647 - 1)" if v == -(1 << 31) else str(v)
    if n == "i64":
        return "(-9223372036854775807L - 1L)" if v == -(1 << 63) else f"{v}L"
    if n == "u32":
        return f"{v}u"
    if n == "u64":
        return f"{v}ul"
    return f"{CTYPES[n]}({v})"


@dataclass
class Tile:
    """A tile materialized in a layout: a register array or a uniform expression.

    Pointer tiles also carry `base`, a scalar pointer expression; their registers hold
    element offsets from it.
    """

    layout: L.BitLayout
    name: str | None = None
    uniform: str | None = None
    base: str | None = None
    root: str | None = None  # the kernel argument a pointer tile derives from
    frag: tuple[int, int] | None = None  # (TM, TN) for a simdgroup_matrix array

    def get(self, r: int | str) -> str:
        if self.uniform is not None:
            return self.uniform
        if self.frag is None:
            return f"{self.name}[{r}]"
        ltn = self.frag[1].bit_length() - 1
        if isinstance(r, int):
            i, j, e = r >> (1 + ltn), (r >> 1) & (self.frag[1] - 1), r & 1
            return f"{self.name}[{i}][{j}].thread_elements()[{e}]"
        return (f"{self.name}[({r}) >> {1 + ltn}][(({r}) >> 1) & {self.frag[1] - 1}]"
                f".thread_elements()[({r}) & 1]")  # fmt: skip


@dataclass
class KernelArg:
    """One runtime argument of a generated kernel, in binding order."""

    name: str
    index: int
    is_pointer: bool
    dtype: str  # IR scalar name of the element or scalar type
    written: bool = False

    @property
    def struct_format(self) -> str:
        return STRUCT_FORMATS[self.dtype]


@dataclass
class GeneratedKernel:
    name: str
    source: str
    args: list[KernelArg]
    num_warps: int
    threadgroup_memory: int
    warnings: list[str] = field(default_factory=list)


class _Codegen:
    def __init__(self, module: ir.Module, plan: LayoutPlan, max_tg_memory: int) -> None:
        self.m = module
        self.plan = plan
        self.nw = plan.num_warps
        self.max_tg = max_tg_memory
        self.names = NameGen()
        self.e = Emitter()
        self.prologue: dict[str, tuple[str, str]] = {}  # key -> (name, code)
        self.sv: dict[int, str] = {}  # scalar value -> expression
        self.tiles: dict[int, Tile] = {}  # anchored tile value -> tile
        self.roots: dict[int, str] = {}  # scalar pointer value -> argument name
        self.memo: dict[tuple, tuple[tuple[int, ...], Tile]] = {}
        self.scope: list[int] = [0]
        self._scope_counter = 0
        self.tg_bytes = 0
        self.tg_loc: Loc | None = None
        self.written: set[str] = set()
        self.warnings: list[str] = []
        self.loc: Loc | None = None
        self.descs: dict[int, object] = {}  # descriptor value -> DescInfo
        self._frag_elem: dict[str, str] = {}  # simdgroup_matrix array -> element type
        from tegula.compiler.codegen.dot import find_direct_operands, use_counts

        self.direct: set[int] = find_direct_operands(module)
        self.uses: dict[int, int] = use_counts(module)

    # ---- helpers ----

    def err(self, msg: str) -> CompilationError:
        return CompilationError(msg, self.loc)

    def fresh(self, hint: str | None) -> str:
        return self.names.fresh(hint)

    def push_scope(self) -> None:
        self._scope_counter += 1
        self.scope.append(self._scope_counter)

    def pop_scope(self) -> None:
        self.scope.pop()

    def memo_get(self, key: tuple) -> Tile | None:
        hit = self.memo.get(key)
        if hit is None:
            return None
        path, tile = hit
        cur = tuple(self.scope)
        return tile if cur[: len(path)] == path else None

    def memo_put(self, key: tuple, tile: Tile) -> Tile:
        self.memo[key] = (tuple(self.scope), tile)
        return tile

    def pro(self, key: str, hint: str, ctype_: str, expr: str) -> str:
        """Declares a kernel-level constant once and returns its name."""
        if expr in ("0", "true", "false"):
            return expr
        hit = self.prologue.get(key)
        if hit is None:
            name = self.fresh(hint)
            hit = self.prologue[key] = (name, f"const {ctype_} {name} = {expr};")
        return hit[0]

    def declare(self, t: ir.Type, lay: L.BitLayout, hint: str | None) -> str:
        name = self.fresh(hint)
        e = ir.elem_of(t)
        nbytes = 4 if isinstance(e, ir.PointerType) else max(1, e.dtype.itemsize)
        regs32 = lay.num_regs * max(1, nbytes // 4) if nbytes >= 4 else lay.num_regs
        if regs32 > MAX_REGS_ERROR:
            raise self.err(
                f"a {'x'.join(map(str, lay.shape))} tile needs {regs32} registers per thread, "
                f"more than the limit of {MAX_REGS_ERROR}; use smaller blocks or more num_warps"
            )
        if regs32 > MAX_REGS_WARN:
            self.warnings.append(f"{self.loc}: a tile uses {regs32} registers per thread")
        self.e.line(f"{ctype(t)} {name}[{lay.num_regs}];")
        return name

    def declare_frag(self, t: ir.Type, lay: L.BitLayout, hint: str | None) -> Tile | None:
        """Declares a simdgroup_matrix array for a float tile in an accumulator layout."""
        g = L.frag_grid(lay)
        e = ir.elem_of(t)
        if g is None or not isinstance(e, ir.ScalarType) or not is_float(e):
            return None
        tm, tn = g[0], g[1]
        name = self.fresh(hint)
        self._frag_elem[name] = CTYPES[e.name]
        self.e.line(f"simdgroup_matrix<{CTYPES[e.name]}, 8, 8> {name}[{tm}][{tn}];")
        return Tile(lay, name, frag=(tm, tn))

    def loop(self, n: int, body: str) -> None:
        """Emits `body` (with `{r}` for the register index) for registers 0..n-1."""
        if n == 1 or "thread_elements" in body:
            # simdgroup_matrix arrays must be indexed with constants.
            for r in range(n):
                self.e.line(body.format(r=r))
            return
        self.e.line("#pragma unroll")
        self.e.line(f"for (int r = 0; r < {n}; ++r) {body.format(r='r')}")

    # ---- thread coordinates ----

    def thread_coord(self, lay: L.BitLayout, dim: int) -> str:
        parts = []
        for t in L.thread_terms(lay):
            if t.dim != dim:
                continue
            idx = "lane" if t.source == "lane" else "warp"
            m = (1 << t.width) - 1
            s = f"(({idx} >> {t.src_shift}) & {m}u)" if t.src_shift else f"({idx} & {m}u)"
            parts.append(f"({s} << {t.dst_shift})" if t.dst_shift else s)
        if not parts:
            return "0"
        return self.pro(f"coord|{lay}|{dim}", "tc", "int", "int(" + " | ".join(parts) + ")")

    def coord(self, lay: L.BitLayout, r: int, dim: int) -> str:
        tt = self.thread_coord(lay, dim)
        c = L.coords_of(lay, r)[dim]
        if tt == "0":
            return str(c)
        return tt if c == 0 else f"({tt} + {c})"

    def owner(self, lay: L.BitLayout) -> str:
        lm, wm = L.owner_mask(lay)
        conds = []
        if lm:
            conds.append(f"(lane & {lm}u) == 0")
        if wm:
            conds.append(f"(warp & {wm}u) == 0")
        if not conds:
            return "true"
        return self.pro(f"owner|{lm}|{wm}", "own", "bool", " && ".join(conds))

    def flat(self, lay: L.BitLayout, strides: tuple[int, ...]) -> tuple[str, list[int]]:
        """Returns (thread expression, per-register constants) of the flat index sum(c*stride)."""
        parts = []
        for d, s in enumerate(strides):
            tt = self.thread_coord(lay, d)
            if tt != "0" and s:
                parts.append(tt if s == 1 else f"{tt} * {s}")
        expr = " + ".join(parts) if parts else "0"
        name = self.pro(f"flat|{lay}|{strides}", "tf", "int", expr) if parts else "0"
        consts = [
            sum(c * s for c, s in zip(L.coords_of(lay, r), strides, strict=True))
            for r in range(lay.num_regs)
        ]
        return name, consts

    def use_tg(self, nbytes: int) -> None:
        if nbytes > self.tg_bytes:
            self.tg_bytes = nbytes
            self.tg_loc = self.loc
        if nbytes > self.max_tg:
            raise self.err(
                f"this operation needs {nbytes} bytes of threadgroup memory, more than the "
                f"device's {self.max_tg}; use smaller blocks"
            )

    # ---- scalars ----

    def s(self, v: ir.Value) -> str:
        try:
            return self.sv[id(v)]
        except KeyError:
            raise self.err(f"internal error: scalar {v!r} used before its definition") from None

    def bind_scalar(self, v: ir.Value, expr: str, hint: str | None = None) -> str:
        """Binds a scalar result to a fresh local initialized with `expr`."""
        name = self.fresh(hint or v.name_hint)
        t = v.type
        if isinstance(t, ir.PointerType):
            self.e.line(f"const auto {name} = {expr};")  # keeps the argument's constness
        else:
            self.e.line(f"const {CTYPES[t.name]} {name} = {expr};")
        self.sv[id(v)] = name
        return name

    # ---- elementwise expressions ----

    def expr(self, op: ir.Op, args: list[str], types: list[ir.Type]) -> str:
        """Returns the MSL expression for one element of an elementwise op."""
        out = ir.elem_of(op.result.type)
        name = op.name
        if name == "binary":
            kind = op.attrs["op"]
            if is_float(out):
                a, b = (f"float({x})" if is_half(out) else x for x in args)
                if kind in ("min", "max"):
                    e = f"f{kind}({a}, {b})"
                elif kind == "mod":
                    e = f"fmod({a}, {b})"
                else:
                    e = f"({a} {BIN_SYMBOLS[kind]} {b})"
                return f"{CTYPES[out.name]}({e})" if is_half(out) else e
            a, b = args
            if kind in ("min", "max"):
                e = f"{kind}({a}, {b})"
            else:
                e = f"({a} {BIN_SYMBOLS[kind]} {b})"
            if out.name == "i1" or out.dtype.primitive_bitwidth < 32:
                e = f"{CTYPES[out.name]}{e}" if e.startswith("(") else f"{CTYPES[out.name]}({e})"
            return e
        if name == "cmp":
            src = ir.elem_of(types[0])
            a, b = (f"float({x})" if is_half(src) else x for x in args)
            return f"({a} {CMP_SYMBOLS[op.attrs['pred']]} {b})"
        if name == "unary":
            kind = op.attrs["op"]
            a = args[0]
            if kind == "neg":
                return f"{CTYPES[out.name]}(-{a})" if not is_float(out) or is_half(out) \
                    else f"(-{a})"
            if kind == "not":
                return f"(!{a})" if out.name == "i1" else f"{CTYPES[out.name]}(~{a})"
            if kind == "abs":
                if is_float(out):
                    return f"{CTYPES[out.name]}(fabs(float({a})))" if is_half(out) \
                        else f"fabs({a})"
                return f"{CTYPES[out.name]}(abs({a}))" if out.dtype.is_signed() else a
            fn = MATH_FUNCS[kind]
            if is_half(out):
                return f"{CTYPES[out.name]}({fn}(float({a})))"
            return f"{fn}({a})"
        if name == "fma":
            if is_float(out):
                if is_half(out):
                    a, b, c = (f"float({x})" for x in args)
                    return f"{CTYPES[out.name]}(fma({a}, {b}, {c}))"
                return f"fma({args[0]}, {args[1]}, {args[2]})"
            return f"{CTYPES[out.name]}({args[0]} * {args[1]} + {args[2]})"
        if name == "select":
            return f"({args[0]} ? {args[1]} : {args[2]})"
        if name == "cast":
            src = ir.elem_of(types[0])
            a = args[0]
            if out.name == "i1":
                return f"({a} != 0)"
            if src.name == "i1" or src == out:
                return f"{CTYPES[out.name]}({a})"
            if is_half(src) and is_half(out) or (is_half(src) and not is_float(out)):
                return f"{CTYPES[out.name]}(float({a}))"
            return f"{CTYPES[out.name]}({a})"
        if name == "bitcast":
            return f"as_type<{CTYPES[out.name]}>({args[0]})"
        raise self.err(f"internal error: `{name}` isn't elementwise")

    # ---- materialization ----

    def mat(self, v: ir.Value, lay: L.BitLayout) -> Tile:
        """Returns `v` as a tile in layout `lay`, emitting code if needed."""
        key = (id(v), lay)
        hit = self.memo_get(key)
        if hit is not None:
            return hit
        k = self.plan.classify(v)
        if k == ANCHORED:
            t = self.tiles.get(id(v))
            if t is None:
                raise self.err(f"internal error: tile {v!r} used before its definition")
            if t.layout == lay:
                return t
            return self.memo_put(key, self.convert(t, lay, v))
        op = self.plan.defining[id(v)]
        saved, self.loc = self.loc, op.loc or self.loc
        try:
            if op.name in VIEWS:
                tile = self.mat_view(op, lay)
            else:
                assert k == CHEAP
                tile = self.mat_cheap(op, lay)
        finally:
            self.loc = saved
        return self.memo_put(key, tile)

    def mat_view(self, op: ir.Op, lay: L.BitLayout) -> Tile:
        src_lay = view_source_layout(op, lay)
        src = self.mat(op.operands[0], src_lay)
        if op.name != "broadcast" or src.uniform is not None:
            return Tile(lay, src.name, src.uniform, src.base, src.root, src.frag)
        m = L.reg_map(src_lay, lay)
        assert m is not None
        if m == tuple(range(lay.num_regs)) and src.frag is None:
            return Tile(lay, src.name, None, src.base, src.root)
        name = self.declare(op.result.type, lay, "bc")
        for r, sr in enumerate(m):
            self.e.line(f"{name}[{r}] = {src.get(sr)};")
        return Tile(lay, name, None, src.base, src.root)

    def mat_cheap(self, op: ir.Op, lay: L.BitLayout) -> Tile:
        t = op.result.type
        name = op.name
        if name == "splat":
            x = op.operands[0]
            if isinstance(x.type, ir.PointerType):
                return Tile(lay, uniform="0", base=self.s(x), root=self.roots.get(id(x)))
            return Tile(lay, uniform=self.s(x))
        if name in ("full", "const"):
            return Tile(lay, uniform=literal(op.attrs["value"], t.elem))
        if name == "arange":
            start = op.attrs["start"]
            arr = self.declare(t, lay, "ar")
            for r in range(lay.num_regs):
                c = self.coord(lay, r, 0)
                self.e.line(f"{arr}[{r}] = {c if not start else f'{start} + {c}'};")
            return Tile(lay, arr)
        return self.elementwise(op, lay, lazy=True)

    def elementwise(self, op: ir.Op, lay: L.BitLayout, lazy: bool) -> Tile:
        """Emits an elementwise op (including `addptr`) over tiles in `lay`."""
        ins: list[Tile | str] = []
        types = [v.type for v in op.operands]
        for v in op.operands:
            ins.append(self.mat(v, lay) if isinstance(v.type, ir.TileType) else self.s(v))
        hint = op.result.name_hint or ("t" if lazy else None)
        if op.name == "addptr":
            p, off = ins
            if not isinstance(p, Tile):
                raise self.err("internal error: scalar pointer in a tile addptr")
            o = off if isinstance(off, str) else None
            if isinstance(off, Tile) and off.uniform is not None:
                o = off.uniform
            if p.uniform is not None and o is not None:
                u = o if p.uniform == "0" else f"({p.uniform} + {o})"
                return Tile(lay, uniform=u, base=p.base, root=p.root)
            if p.uniform == "0" and isinstance(off, Tile):
                return Tile(lay, off.name, base=p.base, root=p.root)
            arr = self.declare(op.result.type, lay, hint)
            og = (lambda r: o) if o is not None else off.get  # type: ignore[union-attr]
            self.loop(lay.num_regs, f"{arr}[{{r}}] = {p.get('{r}')} + {og('{r}')};")
            return Tile(lay, arr, base=p.base, root=p.root)
        ptr_tiles = [x for x in ins if isinstance(x, Tile) and x.base is not None]
        base = root = None
        if ptr_tiles:
            if op.name != "select":
                raise self.err(f"pointers don't support `{op.name}`")
            bases = {x.base for x in ptr_tiles}
            if len(bases) != 1 or len(ptr_tiles) != 2:
                raise self.err(
                    "tl.where on pointers needs both pointers to derive from the same base "
                    "pointer; select integer offsets instead"
                )
            base, root = ptr_tiles[0].base, ptr_tiles[0].root
        if all(isinstance(x, str) or x.uniform is not None for x in ins):
            args = [x if isinstance(x, str) else x.uniform for x in ins]
            e = self.expr(op, args, types)
            name = self.fresh(hint)
            self.e.line(f"const {ctype(op.result.type)} {name} = {e};")
            return Tile(lay, uniform=name, base=base, root=root)
        arr = self.declare(op.result.type, lay, hint)
        args = [x if isinstance(x, str) else x.get("{r}") for x in ins]
        self.loop(lay.num_regs, f"{arr}[{{r}}] = {self.expr(op, args, types)};")
        return Tile(lay, arr, base=base, root=root)

    def convert(self, t: Tile, lay: L.BitLayout, v: ir.Value) -> Tile:
        """Moves a tile to another layout: a register remap or a threadgroup exchange."""
        if t.uniform is not None:
            return Tile(lay, uniform=t.uniform, base=t.base, root=t.root)
        m = L.reg_map(t.layout, lay)
        name = self.declare(v.type, lay, "cv")
        if m is not None:
            for r, sr in enumerate(m):
                self.e.line(f"{name}[{r}] = {t.get(sr)};")
            return Tile(lay, name, base=t.base, root=t.root)
        cty = ctype(v.type)
        eb = 4 if cty == "int" else max(1, ir.elem_of(v.type).dtype.itemsize)
        shape = lay.shape
        inner = shape[-1] + (16 // eb if shape[-1] >= 16 else 0)
        strides, acc = [], 1
        for d in reversed(range(len(shape))):
            strides.append(acc)
            acc *= inner if d == len(shape) - 1 else shape[d]
        strides = tuple(reversed(strides))
        self.use_tg(acc * eb)
        src_flat, src_c = self.flat(t.layout, strides)
        dst_flat, dst_c = self.flat(lay, strides)
        own = self.owner(t.layout)
        self.e.line(BARRIER)
        with self.e.block(""):
            self.e.line(f"threadgroup {cty}* buf = (threadgroup {cty}*)tg_mem;")
            with self.e.block(f"if ({own})"):
                for r in range(t.layout.num_regs):
                    self.e.line(f"buf[{_add(src_flat, src_c[r])}] = {t.get(r)};")
            self.e.line(BARRIER)
            for r in range(lay.num_regs):
                self.e.line(f"{name}[{r}] = buf[{_add(dst_flat, dst_c[r])}];")
        return Tile(lay, name, base=t.base, root=t.root)

    # ---- ops ----

    def block(self, block: ir.Block) -> None:
        for op in block.ops:
            self.loc = op.loc or self.loc
            self.op(op)

    def op(self, op: ir.Op) -> None:
        name = op.name
        res = op.results[0] if len(op.results) == 1 else None
        if res is not None and isinstance(res.type, ir.TileType) and name not in (
            "reduce",
            "if",
            "for",
        ):
            if self.plan.classify(res) != ANCHORED:
                return  # emitted lazily at each use
            lay = self.plan.layout_of(res)
            if name == "load":
                self.tiles[id(res)] = self.load(op, lay)
            elif name in ("binary", "cmp", "unary", "fma", "select", "cast", "bitcast",
                          "addptr"):  # fmt: skip
                self.tiles[id(res)] = self.elementwise(op, lay, lazy=False)
            elif name == "dot":
                from tegula.compiler.codegen.dot import emit_dot

                self.tiles[id(res)] = emit_dot(self, op, lay)
            elif name == "desc_load":
                if id(res) in self.direct:
                    return  # read by its dot straight from device memory
                from tegula.compiler.codegen.dot import emit_desc_load

                self.tiles[id(res)] = emit_desc_load(self, op, lay)
            else:
                raise self.err(f"`{name}` on tiles isn't supported by the MSL backend yet")
            return
        handler = getattr(self, "op_" + name, None)
        if handler is None:
            raise self.err(f"`{name}` isn't supported by the MSL backend yet")
        handler(op)

    def op_const(self, op: ir.Op) -> None:
        self.sv[id(op.result)] = literal(op.attrs["value"], op.result.type)

    def op_program_id(self, op: ir.Op) -> None:
        self.sv[id(op.result)] = f"int(pid.{'xyz'[op.attrs['axis']]})"

    def op_num_programs(self, op: ir.Op) -> None:
        self.sv[id(op.result)] = f"int(npid.{'xyz'[op.attrs['axis']]})"

    def _scalar_elementwise(self, op: ir.Op) -> None:
        args = [self.s(v) for v in op.operands]
        self.bind_scalar(op.result, self.expr(op, args, [v.type for v in op.operands]))

    op_binary = op_cmp = op_unary = op_fma = op_select = op_cast = op_bitcast = (
        _scalar_elementwise
    )

    def op_addptr(self, op: ir.Op) -> None:
        p, off = op.operands
        self.bind_scalar(op.result, f"({self.s(p)} + {self.s(off)})")
        if id(p) in self.roots:
            self.roots[id(op.result)] = self.roots[id(p)]

    def op_splat(self, op: ir.Op) -> None:
        raise self.err("internal error: scalar splat")

    def load(self, op: ir.Op, lay: L.BitLayout) -> Tile:
        res = op.result
        p = self.mat(op.operands[0], lay)
        arr = self.declare(res.type, lay, res.name_hint)
        if len(op.operands) == 3:
            m = self.mat(op.operands[1], lay)
            o = self.mat(op.operands[2], lay)
            if m.uniform == "true":
                body = f"{arr}[{{r}}] = {p.base}[{p.get('{r}')}];"
            else:
                body = f"{arr}[{{r}}] = {m.get('{r}')} ? {p.base}[{p.get('{r}')}] : {o.get('{r}')};"
        else:
            body = f"{arr}[{{r}}] = {p.base}[{p.get('{r}')}];"
        self.loop(lay.num_regs, body)
        return Tile(lay, arr)

    def op_load(self, op: ir.Op) -> None:
        p = self.s(op.operands[0])
        if len(op.operands) == 3:
            e = f"({self.s(op.operands[1])} ? *{p} : {self.s(op.operands[2])})"
        else:
            e = f"*{p}"
        self.bind_scalar(op.result, e)

    def op_store(self, op: ir.Op) -> None:
        ptr, val = op.operands[0], op.operands[1]
        mask = op.operands[2] if len(op.operands) > 2 else None
        if not isinstance(ptr.type, ir.TileType):
            if id(ptr) in self.roots:
                self.written.add(self.roots[id(ptr)])
            cond = "lane == 0 && warp == 0"
            if mask is not None:
                cond += f" && {self.s(mask)}"
            self.e.line(f"if ({cond}) *{self.s(ptr)} = {self.s(val)};")
            return
        lay = store_layout(self.plan, op)
        p = self.mat(ptr, lay)
        v = self.mat(val, lay)
        if p.root:
            self.written.add(p.root)
        elem = CTYPES[ptr.type.elem.elem.name]
        store = f"{p.base}[{p.get('{r}')}] = {elem}({v.get('{r}')});"
        if mask is not None:
            mt = self.mat(mask, lay)
            if mt.uniform != "true":
                store = f"if ({mt.get('{r}')}) {store}"
        own = self.owner(lay)
        if own == "true":
            self.loop(lay.num_regs, store)
        else:
            with self.e.block(f"if ({own})"):
                self.loop(lay.num_regs, store)

    def op_return(self, op: ir.Op) -> None:
        pass

    def op_make_desc(self, op: ir.Op) -> None:
        from tegula.compiler.codegen.dot import DescInfo

        t = op.result.type
        r = t.shape_rank
        vals = [self.s(v) for v in op.operands]
        self.descs[id(op.result)] = DescInfo(
            vals[0], vals[1 : 1 + r], vals[1 + r :], t.block_shape, t.elem,
            self.roots.get(id(op.operands[0])),
        )  # fmt: skip

    def op_desc_store(self, op: ir.Op) -> None:
        from tegula.compiler.codegen.dot import emit_desc_store

        emit_desc_store(self, op)

    # ---- control flow ----

    def _assign(self, dst: Tile, src: Tile, n: int) -> None:
        if dst.frag is not None and (src.frag == dst.frag or src.uniform is not None):
            tm, tn = dst.frag
            et = self._frag_elem.get(dst.name, "float")
            for i in range(tm):
                for j in range(tn):
                    rhs = f"{src.name}[{i}][{j}]" if src.uniform is None else \
                        f"make_filled_simdgroup_matrix<{et}, 8, 8>({et}({src.uniform}))"
                    self.e.line(f"{dst.name}[{i}][{j}] = {rhs};")
            return
        self.loop(n, f"{dst.get('{r}')} = {src.get('{r}')};")

    def _carried(self, v: ir.Value, init_tile: Tile | None, init_expr: str | None,
                 hint: str | None) -> Tile | str:  # fmt: skip
        """Declares a mutable local for a loop-carried or if-result value."""
        t = v.type
        if isinstance(t, ir.TileType):
            lay = self.plan.layout_of(v)
            ft = self.declare_frag(t, lay, hint)
            if ft is not None:
                if init_tile is not None:
                    self._assign(ft, init_tile, lay.num_regs)
                return ft
            name = self.declare(t, lay, hint)
            base = root = None
            if init_tile is not None:
                self._assign(Tile(lay, name), init_tile, lay.num_regs)
                base, root = init_tile.base, init_tile.root
            return Tile(lay, name, base=base, root=root)
        name = self.fresh(hint)
        if isinstance(t, ir.PointerType):
            if init_expr is None:
                raise self.err("a pointer can't be the result of a runtime `if`; select an offset")
            decl = f"auto {name}"
        else:
            decl = f"{CTYPES[t.name]} {name}"
        self.e.line(f"{decl} = {init_expr};" if init_expr is not None else f"{decl};")
        return name

    def _yield_into(self, targets: list[Tile | str], values: list[ir.Value]) -> None:
        srcs = []
        for tgt, y in zip(targets, values, strict=True):
            if isinstance(tgt, Tile):
                src = self.mat(y, tgt.layout)
                if src.base is not None and tgt.base is not None and src.base != tgt.base:
                    raise self.err(
                        "a pointer tile carried through a loop or `if` must keep the same "
                        "base pointer; carry an integer offset tile instead"
                    )
                if tgt.base is None and src.base is not None:
                    tgt.base, tgt.root = src.base, src.root
                srcs.append(src)
            else:
                srcs.append(self.s(y))
        # Copy through temporaries when a target is read by a later assignment.
        names = {t.name if isinstance(t, Tile) else t for t in targets}
        staged = []
        for tgt, src in zip(targets, srcs, strict=True):
            if isinstance(tgt, Tile):
                if src.name in names and src.name != tgt.name:
                    tmp = self.fresh("tmp")
                    n = tgt.layout.num_regs
                    self.e.line(f"{ctype_of_tile(tgt, targets, values)} {tmp}[{n}];")
                    self._assign(Tile(tgt.layout, tmp), src, n)
                    src = Tile(tgt.layout, tmp)
                staged.append((tgt, src))
            else:
                if src in names and src != tgt:
                    tmp = self.fresh("tmp")
                    self.e.line(f"const auto {tmp} = {src};")
                    src = tmp
                staged.append((tgt, src))
        for tgt, src in staged:
            if isinstance(tgt, Tile):
                if src.name != tgt.name:
                    self._assign(tgt, src, tgt.layout.num_regs)
            elif src != tgt:
                self.e.line(f"{tgt} = {src};")

    def _bind(self, v: ir.Value, carried: Tile | str) -> None:
        if isinstance(carried, Tile):
            self.tiles[id(v)] = carried
        else:
            self.sv[id(v)] = carried
            if isinstance(v.type, ir.PointerType):
                pass

    def op_for(self, op: ir.Op) -> None:
        lb, ub, step = (self.s(v) for v in op.operands[:3])
        inits = op.operands[3:]
        body = op.regions[0].block
        carried: list[Tile | str] = []
        for arg, init in zip(body.args[1:], inits, strict=True):
            if isinstance(arg.type, ir.TileType):
                t = self.mat(init, self.plan.layout_of(arg))
                carried.append(self._carried(arg, t, None, arg.name_hint))
            else:
                carried.append(self._carried(arg, None, self.s(init), arg.name_hint))
                if id(init) in self.roots:
                    self.roots[id(arg)] = self.roots[id(init)]
        for arg, c in zip(body.args[1:], carried, strict=True):
            self._bind(arg, c)
        iv = body.args[0]
        ivn = self.fresh(iv.name_hint or "i")
        self.sv[id(iv)] = ivn
        ity = CTYPES[iv.type.name]
        step_op = op.operands[2].defining_op
        if step_op is not None and step_op.name == "const":
            cond = f"{ivn} < {ub}" if step_op.attrs["value"] > 0 else f"{ivn} > {ub}"
        else:
            cond = f"({step} > 0 ? {ivn} < {ub} : {ivn} > {ub})"
        with self.e.block(f"for ({ity} {ivn} = {lb}; {cond}; {ivn} += {step})"):
            self.push_scope()
            self.block(_body_ops(body))
            self._yield_into(carried, body.ops[-1].operands)
            self.pop_scope()
        for r, c in zip(op.results, carried, strict=True):
            self._bind(r, c)

    def op_if(self, op: ir.Op) -> None:
        cond = self.s(op.operands[0])
        carried = [self._carried(r, None, None, r.name_hint) for r in op.results]
        for i, rg in enumerate(op.regions):
            header = f"if ({cond})" if i == 0 else "else"
            with self.e.block(header):
                self.push_scope()
                self.block(_body_ops(rg.block))
                self._yield_into(carried, rg.block.ops[-1].operands)
                self.pop_scope()
        for r, c in zip(op.results, carried, strict=True):
            self._bind(r, c)

    # ---- reductions ----

    def op_reduce(self, op: ir.Op) -> None:
        from tegula.compiler.codegen.reduce import emit_reduce

        emit_reduce(self, op)

    # ---- kernel ----

    def run(self) -> GeneratedKernel:
        func = self.m.func
        body = self.m.body
        arg_names = func.attrs.get("arg_names") or [a.name_hint for a in body.args]
        args = []
        for i, (a, n) in enumerate(zip(body.args, arg_names, strict=True)):
            name = self.names.reserve(n)
            self.sv[id(a)] = name
            t = a.type
            if isinstance(t, ir.PointerType):
                self.roots[id(a)] = name
                args.append(KernelArg(name, i, True, t.elem.name))
            else:
                args.append(KernelArg(name, i, False, t.name))
        if len(args) > 31:
            raise CompilationError(
                f"a kernel can take at most 31 runtime arguments, but this one takes {len(args)}",
                func.loc,
            )
        kname = self.names.reserve(self.m.name)
        self.block(body)
        for a in args:
            a.written = a.name in self.written
        out = Emitter()
        out.line("#include <metal_stdlib>")
        out.line("using namespace metal;")
        out.lines(PRELUDE)
        params = []
        for a in args:
            ct = CTYPES[a.dtype]
            if a.is_pointer:
                q = "" if a.written else "const "
                params.append(f"device {q}{ct}* {a.name} [[buffer({a.index})]]")
            else:
                params.append(f"constant {ct}& {a.name} [[buffer({a.index})]]")
        params += [
            "uint3 pid [[threadgroup_position_in_grid]]",
            "uint3 npid [[threadgroups_per_grid]]",
            "uint lane [[thread_index_in_simdgroup]]",
            "uint warp [[simdgroup_index_in_threadgroup]]",
        ]
        out.line(f"[[kernel]] void {kname}(")
        with out.indented(), out.indented():
            for i, p in enumerate(params):
                out.line(p + ("," if i < len(params) - 1 else ") {"))
        with out.indented():
            if self.tg_bytes:
                n16 = -(-self.tg_bytes // 16)
                out.line(f"threadgroup float4 tg_mem4[{n16}];")
                out.line("threadgroup uchar* tg_mem = (threadgroup uchar*)tg_mem4;")
            for _, code in self.prologue.values():
                out.line(code)
            out.lines(self.e.text())
        out.line("}")
        return GeneratedKernel(kname, out.text(), args, self.nw, self.tg_bytes, self.warnings)


def _add(expr: str, c: int) -> str:
    if expr == "0":
        return str(c)
    return expr if c == 0 else f"{expr} + {c}"


def _body_ops(block: ir.Block) -> ir.Block:
    """Returns a view of `block` without its terminator."""
    view = ir.Block()
    view.ops = block.ops[:-1]
    return view


def ctype_of_tile(tgt: Tile, targets: list, values: list[ir.Value]) -> str:
    i = targets.index(tgt)
    return ctype(values[i].type)


def generate(module: ir.Module, plan: LayoutPlan, max_tg_memory: int = 32768) -> GeneratedKernel:
    """Generates MSL for a layout-assigned module."""
    return _Codegen(module, plan, max_tg_memory).run()


def pack_format(args: list[KernelArg]) -> str:
    """Returns the `struct` format that packs all scalar arguments back to back."""
    return "<" + "".join(a.struct_format for a in args if not a.is_pointer)


def scalar_slots(args: list[KernelArg]) -> list[tuple[int, int, int]]:
    """Returns (buffer index, byte offset, size) for each scalar argument."""
    out, off = [], 0
    for a in args:
        if not a.is_pointer:
            size = struct.calcsize("<" + a.struct_format)
            out.append((a.index, off, size))
            off += size
    return out
