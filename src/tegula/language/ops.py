"""Builtin definitions: one frontend handler and one interpreter handler per `tl` function.

Each builtin is declared with `@builtin(interp=_i_name)`. The decorated function is the
frontend handler: it takes the code generator context `ctx` and the user's arguments
(IR values or compile-time Python values) and emits IR through `ctx.b`. The interpreter
handler takes the same arguments as interpreter tiles, pointers, or Python values and
computes on NumPy.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from tegula.compiler import ir, semantic
from tegula.compiler.errors import CompilationError
from tegula.compiler.semantic import describe
from tegula.interpreter import interp as I  # noqa: N812
from tegula.interpreter.interp import IDesc, IPointer, ITile
from tegula.language import core
from tegula.language.core import DESC_METHODS, INTERP, TILE_METHODS, builtin

# ---------------------------------------------------------------------------
# Argument helpers shared by both handlers
# ---------------------------------------------------------------------------


def _cint(v: Any, what: str) -> int:
    """Returns `v` as a compile-time int, or raises an error that names `what`."""
    v = core.unwrap(v)
    if isinstance(v, np.integer):
        v = int(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    hint = ""
    if isinstance(v, (ir.Value, ITile)):
        name = getattr(v, "name_hint", None)
        p = f"`{name}`" if name else "it"
        hint = (
            f" If {p} comes from a kernel parameter, annotate the parameter as tl.constexpr, "
            f"for example `{name or 'BLOCK'}: tl.constexpr`."
        )
        v = describe(v) if isinstance(v, ir.Value) else f"a runtime value of type {v.dtype}"
    else:
        v = describe(v)
    raise CompilationError(f"{what} must be a compile-time integer, but got {v}.{hint}")


def _dtype(v: Any, what: str = "dtype") -> core.dtype:
    v = core.unwrap(v)
    if isinstance(v, core.dtype):
        return v
    raise CompilationError(f"`{what}` must be a tl dtype such as tl.float32, but got {v!r}")


def _axis(axis: Any, rank: int) -> int:
    a = _cint(axis, "`axis`")
    if not -rank <= a < rank:
        raise CompilationError(f"axis {a} is out of range for a tile of rank {rank}")
    return a % rank


def _shape_args(shape: Sequence[Any]) -> tuple[Any, ...]:
    """Accepts `f(x, 4, 8)` and `f(x, (4, 8))`."""
    if len(shape) == 1 and isinstance(core.unwrap(shape[0]), (tuple, list)):
        return tuple(core.unwrap(shape[0]))
    return tuple(shape)


def _pow2_count(n: int, what: str) -> None:
    if not ir.is_pow2_dim(n):
        raise CompilationError(
            f"{what} has {n} elements, but the number of elements must be a power of two from "
            "1 to 65536. Round the block size up to a power of two and mask the extra "
            "elements."
        )


def _value(ctx, x: Any) -> ir.Value:
    return semantic.to_value(ctx.b, x)


def _tile_value(ctx, x: Any, fname: str) -> ir.Value:
    v = _value(ctx, x)
    if not isinstance(v.type, ir.TileType):
        raise CompilationError(f"tl.{fname} needs a tile, but got {describe(v)}")
    return v


def _itile(x: Any) -> ITile:
    return I.to_tile(x)


def _imap(x: Any, f: Callable[[np.ndarray], np.ndarray]) -> Any:
    """Applies a shape function to an interpreter tile or pointer tile."""
    if isinstance(x, IPointer):
        return x.with_offsets(f(x.offsets))
    t = _itile(x)
    return ITile(f(t.data), t.dtype)


def _ishape(x: Any) -> tuple[int, ...]:
    return x.shape if isinstance(x, (ITile, IPointer)) else ()


def _method(b: core.Builtin, *names: str) -> core.Builtin:
    for n in names or (b.name,):
        TILE_METHODS[n] = b
    return b


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------


def _check_axis3(axis: Any) -> int:
    a = _cint(axis, "`axis`")
    if a not in (0, 1, 2):
        raise CompilationError(f"`axis` must be 0, 1, or 2, but got {a}")
    return a


def _i_program_id(axis):
    return ITile(np.int32(INTERP.program_id[_check_axis3(axis)]), core.int32)


@builtin(interp=_i_program_id)
def program_id(ctx, axis):
    """Returns the index of the current program along `axis` (0, 1, or 2)."""
    return ctx.b.create("program_id", [], [ir.i32], {"axis": _check_axis3(axis)}).result


def _i_num_programs(axis):
    return ITile(np.int32(INTERP.grid[_check_axis3(axis)]), core.int32)


@builtin(interp=_i_num_programs)
def num_programs(ctx, axis):
    """Returns the number of programs along `axis` (0, 1, or 2)."""
    return ctx.b.create("num_programs", [], [ir.i32], {"axis": _check_axis3(axis)}).result


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def _arange_bounds(start, end) -> tuple[int, int]:
    s = _cint(start, "the start of tl.arange")
    e = _cint(end, "the end of tl.arange")
    if e <= s:
        raise CompilationError(f"tl.arange({s}, {e}) needs end > start")
    _pow2_count(e - s, f"tl.arange({s}, {e})")
    return s, e


def _i_arange(start, end):
    s, e = _arange_bounds(start, end)
    return ITile(np.arange(s, e, dtype=np.int32), core.int32)


@builtin(interp=_i_arange)
def arange(ctx, start, end):
    """Returns the `int32` tile `[start, start + 1, ..., end - 1]`.

    `start` and `end` must be compile-time integers, and `end - start` must be a power of
    two.
    """
    s, e = _arange_bounds(start, end)
    t = ir.TileType((e - s,), ir.i32)
    return ctx.b.create("arange", [], [t], {"start": s, "end": e}).result


def _i_full(shape, value, dtype):
    shape = semantic.check_shape(shape, "tl.full")
    dt = _dtype(dtype)
    v = core.unwrap(value)
    if isinstance(v, ITile) and v.shape != ():
        raise CompilationError("tl.full needs a scalar value")
    return ITile(np.broadcast_to(I.to_tile(v, dt).data, shape).copy(), dt)


@builtin(interp=_i_full)
def full(ctx, shape, value, dtype):
    """Returns a tile of `shape` and `dtype` filled with the scalar `value`."""
    shape = semantic.check_shape(shape, "tl.full")
    dt = _dtype(dtype)
    v = core.unwrap(value)
    if isinstance(v, ir.Value):
        if isinstance(v.type, ir.TileType):
            raise CompilationError(f"tl.full needs a scalar value, but got {describe(v)}")
        return semantic.splat(ctx.b, semantic.cast(ctx.b, v, dt), shape)
    if not semantic.is_literal(v):
        raise CompilationError(f"tl.full needs a numeric value, but got {describe(v)}")
    t = ir.TileType(shape, ir.scalar(dt))
    return ctx.b.create("full", [], [t], {"value": semantic.coerce_literal(v, dt)}).result


def _i_zeros(shape, dtype):
    return _i_full(semantic.check_shape(shape, "tl.zeros"), 0, dtype)


@builtin(interp=_i_zeros)
def zeros(ctx, shape, dtype):
    """Returns a tile of `shape` and `dtype` filled with zeros."""
    return full.frontend(ctx, semantic.check_shape(shape, "tl.zeros"), 0, dtype)


def _i_zeros_like(input):
    t = _itile(input)
    return _i_full(t.shape, 0, t.dtype)


@builtin(interp=_i_zeros_like)
def zeros_like(ctx, input):
    """Returns a tile of zeros with the shape and dtype of `input`."""
    v = _tile_value(ctx, input, "zeros_like")
    return full.frontend(ctx, v.type.shape, 0, semantic.dtype_of(v))


def _i_full_like(input, value, dtype=None):
    t = _itile(input)
    return _i_full(t.shape, value, dtype or t.dtype)


@builtin(interp=_i_full_like)
def full_like(ctx, input, value, dtype=None):
    """Returns a tile with the shape of `input`, filled with `value`."""
    v = _tile_value(ctx, input, "full_like")
    return full.frontend(ctx, v.type.shape, value, dtype or semantic.dtype_of(v))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

_BLOCK_PTR_MSG = (
    "`boundary_check` and `padding_option` apply to block pointers, which Tegula doesn't "
    "support. Use tl.make_tensor_descriptor for bounds-checked block loads."
)


def _check_mask_dtype(dt: core.dtype) -> None:
    if not dt.is_bool():
        raise CompilationError(
            f"a mask must be a boolean (int1) tile, such as `offs < n`, but got {dt}"
        )


def _i_load(pointer, mask=None, other=None, boundary_check=(), padding_option="",
            cache_modifier="", eviction_policy="", volatile=False):  # fmt: skip
    if not isinstance(pointer, IPointer):
        raise CompilationError(f"tl.load needs a pointer, but got {pointer!r}")
    if boundary_check or padding_option:
        raise CompilationError(_BLOCK_PTR_MSG)
    mask = core.unwrap(mask)
    elem = pointer.elem
    if mask is None or mask is True:
        pointer.check_bounds(pointer.offsets, None, "load")
        return ITile(np.asarray(pointer.flat[pointer.offsets]), elem)
    m = _itile(mask)
    _check_mask_dtype(m.dtype)
    offs, mb = np.broadcast_arrays(pointer.offsets, m.data)
    pointer.check_bounds(offs, mb, "load")
    if pointer.flat.shape[0] == 0:
        data = np.zeros(offs.shape, dtype=elem.to_numpy())
    else:
        data = np.array(pointer.flat[np.where(mb, offs, 0)])
    fill = I.to_tile(other if other is not None else 0, elem).data
    np.copyto(data, np.broadcast_to(fill, data.shape), where=~mb)
    return ITile(data, elem)


@builtin(interp=_i_load)
def load(ctx, pointer, mask=None, other=None, boundary_check=(), padding_option="",
         cache_modifier="", eviction_policy="", volatile=False):  # fmt: skip
    """Loads from a pointer or a tile of pointers.

    Where `mask` is false, the result is `other`, or 0 when `other` isn't given, and no
    memory is read. `cache_modifier`, `eviction_policy`, and `volatile` are accepted for
    Triton compatibility and have no effect.
    """
    b = ctx.b
    p = core.unwrap(pointer)
    if not semantic.is_pointer(p):
        raise CompilationError(f"tl.load needs a pointer or pointer tile, but got {describe(p)}")
    if boundary_check or padding_option:
        raise CompilationError(_BLOCK_PTR_MSG)
    elem = ir.elem_of(p.type).elem.dtype
    mask = core.unwrap(mask)
    if mask is None or mask is True:
        return b.create("load", [p], [ir.with_elem(p.type, ir.scalar(elem))]).result
    if not isinstance(mask, ir.Value):
        raise CompilationError(f"`mask` must be a boolean tile, but got {describe(mask)}")
    _check_mask_dtype(semantic.dtype_of(mask))
    shape = semantic.broadcast_shapes(ir.shape_of(p.type), ir.shape_of(mask.type))
    p = semantic.broadcast_to(b, p, shape)
    mask = semantic.broadcast_to(b, mask, shape)
    other = core.unwrap(other)
    ov = semantic.to_value(b, 0 if other is None else other, elem)
    ov = semantic.broadcast_to(b, ov, shape)
    return b.create("load", [p, mask, ov], [ir.with_elem(p.type, ir.scalar(elem))]).result


def _i_store(pointer, value, mask=None, boundary_check=(), cache_modifier="",
             eviction_policy=""):  # fmt: skip
    if not isinstance(pointer, IPointer):
        raise CompilationError(f"tl.store needs a pointer, but got {pointer!r}")
    if boundary_check:
        raise CompilationError(_BLOCK_PTR_MSG)
    val = I.to_tile(core.unwrap(value), pointer.elem)
    mask = core.unwrap(mask)
    arrays = [pointer.offsets, val.data]
    if mask is not None and mask is not True:
        m = _itile(mask)
        _check_mask_dtype(m.dtype)
        arrays.append(m.data)
    semantic.broadcast_shapes(*[a.shape for a in arrays[:2]])
    bc = np.broadcast_arrays(*arrays)
    offs, vals = bc[0], bc[1]
    mb = bc[2] if len(bc) == 3 else None
    pointer.check_bounds(offs, mb, "store")
    if mb is None:
        pointer.flat[offs] = vals
    else:
        pointer.flat[offs[mb]] = vals[mb]


@builtin(interp=_i_store)
def store(ctx, pointer, value, mask=None, boundary_check=(), cache_modifier="",
          eviction_policy=""):  # fmt: skip
    """Stores `value` through a pointer or a tile of pointers where `mask` is true.

    `value` is converted to the pointee dtype and broadcast to the pointer's shape.
    """
    b = ctx.b
    p = core.unwrap(pointer)
    if not semantic.is_pointer(p):
        raise CompilationError(f"tl.store needs a pointer or pointer tile, but got {describe(p)}")
    if boundary_check:
        raise CompilationError(_BLOCK_PTR_MSG)
    elem = ir.elem_of(p.type).elem.dtype
    v = core.unwrap(value)
    if semantic.is_pointer(v):
        raise CompilationError("tl.store can't store pointers")
    vv = semantic.to_value(b, v, elem)
    mask = core.unwrap(mask)
    if mask is True:
        mask = None
    if mask is not None:
        if not isinstance(mask, ir.Value):
            raise CompilationError(f"`mask` must be a boolean tile, but got {describe(mask)}")
        _check_mask_dtype(semantic.dtype_of(mask))
    shapes = [ir.shape_of(p.type), ir.shape_of(vv.type)]
    if mask is not None:
        shapes.append(ir.shape_of(mask.type))
    shape = shapes[0]
    for s in shapes[1:]:
        shape = semantic.broadcast_shapes(shape, s)
    ops = [semantic.broadcast_to(b, p, shape), semantic.broadcast_to(b, vv, shape)]
    if mask is not None:
        ops.append(semantic.broadcast_to(b, mask, shape))
    b.create("store", ops)


# ---------------------------------------------------------------------------
# Tensor descriptors
# ---------------------------------------------------------------------------


def _desc_parts(base, shape, strides, block_shape):
    shape, strides = list(core.unwrap(shape)), list(core.unwrap(strides))
    block = semantic.check_shape(block_shape, "tensor descriptor block")
    if not (len(shape) == len(strides) == len(block)):
        raise CompilationError(
            f"tl.make_tensor_descriptor needs `shape`, `strides`, and `block_shape` of the "
            f"same rank, but got ranks {len(shape)}, {len(strides)}, and {len(block)}"
        )
    return shape, strides, block


_LAST_STRIDE_MSG = (
    "the last stride of a tensor descriptor must be 1, because the innermost dimension must "
    "be contiguous. Pass the literal 1, or transpose the data so that it's contiguous."
)


def _i_make_tensor_descriptor(base, shape, strides, block_shape, padding_option="zero"):
    shape, strides, block = _desc_parts(base, shape, strides, block_shape)
    if not isinstance(base, IPointer) or base.shape != ():
        raise CompilationError("tl.make_tensor_descriptor needs a scalar base pointer")
    if int(strides[-1]) != 1:
        raise CompilationError(_LAST_STRIDE_MSG)
    return IDesc(base, shape, strides, block)


@builtin(interp=_i_make_tensor_descriptor)
def make_tensor_descriptor(ctx, base, shape, strides, block_shape, padding_option="zero"):
    """Creates a tensor descriptor over `base` with the given shape and strides.

    `desc.load(offsets)` returns the `block_shape` block at element offsets `offsets` and
    fills out-of-bounds elements with zero. `desc.store(offsets, value)` skips
    out-of-bounds elements. The last stride must be 1.
    """
    b = ctx.b
    shape, strides, block = _desc_parts(base, shape, strides, block_shape)
    base = core.unwrap(base)
    if not (semantic.is_pointer(base) and isinstance(base.type, ir.PointerType)):
        raise CompilationError(
            f"tl.make_tensor_descriptor needs a scalar base pointer, but got {describe(base)}"
        )
    if padding_option not in ("zero", "", None):
        raise CompilationError("tensor descriptors support only zero padding")
    if not ctx.is_known_one(strides[-1]):
        raise CompilationError(_LAST_STRIDE_MSG)
    vals = []
    for what, v in [*(("shape", s) for s in shape), *(("strides", s) for s in strides)]:
        v = core.unwrap(v)
        ok = (isinstance(v, int) and not isinstance(v, bool)) or (
            isinstance(v, ir.Value) and isinstance(v.type, ir.ScalarType) and v.type.dtype.is_int()
        )
        if not ok:
            raise CompilationError(
                f"descriptor `{what}` entries must be integer scalars, but got {describe(v)}"
            )
        vals.append(semantic.to_value(b, v))
    t = ir.DescType(base.type.elem, len(block), block)
    return b.create("make_desc", [base, *vals], [t]).result


def _offset_values(ctx, desc: ir.Value, offsets) -> list[ir.Value]:
    offs = list(core.unwrap(offsets))
    if len(offs) != desc.type.shape_rank:
        raise CompilationError(
            f"the descriptor has rank {desc.type.shape_rank}, but got {len(offs)} offsets"
        )
    out = []
    for o in offs:
        o = core.unwrap(o)
        if isinstance(o, ir.Value) and not (
            isinstance(o.type, ir.ScalarType) and o.type.dtype.is_int()
        ):
            raise CompilationError(f"descriptor offsets must be integer scalars, got {describe(o)}")
        out.append(semantic.to_value(ctx.b, o))
    return out


def _i_desc_load(desc: IDesc, offsets):
    off, mask = desc.block_offsets([int(o) for o in offsets])
    return _i_load(desc.base.with_offsets(off), ITile(mask, core.int1))


def _desc_load(ctx, desc, offsets):
    """Loads the block at element `offsets`, filling out-of-bounds elements with zero."""
    offs = _offset_values(ctx, desc, offsets)
    t = ir.TileType(desc.type.block_shape, desc.type.elem)
    return ctx.b.create("desc_load", [desc, *offs], [t]).result


def _i_desc_store(desc: IDesc, offsets, value):
    off, mask = desc.block_offsets([int(o) for o in offsets])
    val = I.to_tile(core.unwrap(value), desc.dtype)
    if val.shape != desc.block_shape:
        raise CompilationError(
            f"desc.store needs a value of the block shape {desc.block_shape}, got {val.shape}"
        )
    _i_store(desc.base.with_offsets(off), val, ITile(mask, core.int1))


def _desc_store(ctx, desc, offsets, value):
    """Stores `value` at element `offsets`, skipping out-of-bounds elements."""
    offs = _offset_values(ctx, desc, offsets)
    v = semantic.to_value(ctx.b, value, desc.type.elem.dtype)
    if ir.shape_of(v.type) != desc.type.block_shape:
        raise CompilationError(
            f"desc.store needs a value of the block shape {desc.type.block_shape}, but got "
            f"{describe(v)}"
        )
    ctx.b.create("desc_store", [desc, *offs, v])


DESC_METHODS["load"] = core.Builtin("tensor_descriptor.load", _desc_load, _i_desc_load)
DESC_METHODS["store"] = core.Builtin("tensor_descriptor.store", _desc_store, _i_desc_store)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def _i_where(condition, x, y):
    c = _itile(condition)
    if not c.dtype.is_bool():
        c = I.binary("ne", c, 0)
    if isinstance(x, IPointer) or isinstance(y, IPointer):
        raise CompilationError("the interpreter doesn't support tl.where on pointers")
    dt = semantic.computation_dtype("select", I._operand(x), I._operand(y))
    semantic.broadcast_shapes(semantic.broadcast_shapes(c.shape, _ishape(x)), _ishape(y))
    return I.wrap(np.where(c.data, I.as_compute(x, dt), I.as_compute(y, dt)), dt)


@builtin(interp=_i_where)
def where(ctx, condition, x, y):
    """Returns `x` where `condition` is true and `y` elsewhere, elementwise."""
    c = core.unwrap(condition)
    if not isinstance(c, ir.Value):
        return x if c else y
    return semantic.where(ctx.b, c, x, y)


@builtin(interp=lambda x, y: I.binary("max", x, y))
def maximum(ctx, x, y):
    """Returns the elementwise maximum. For floats, a NaN operand yields the other operand."""
    return semantic.binary(ctx.b, "max", x, y)


@builtin(interp=lambda x, y: I.binary("min", x, y))
def minimum(ctx, x, y):
    """Returns the elementwise minimum. For floats, a NaN operand yields the other operand."""
    return semantic.binary(ctx.b, "min", x, y)


def _fma_dtype(x, y, z, op_of) -> core.dtype:
    dt = semantic.computation_dtype("mul", op_of(x), op_of(y))
    dt = semantic.computation_dtype("add", dt, op_of(z))
    if not dt.is_floating():
        raise CompilationError(f"tl.fma needs floating-point operands, but they promote to {dt}")
    return dt


def _i_fma(x, y, z):
    dt = _fma_dtype(x, y, z, I._operand)
    a, b, c = (I.as_compute(v, dt).astype(np.float64) for v in (x, y, z))
    return I.wrap(a * b + c, dt)


@builtin(interp=_i_fma)
def fma(ctx, x, y, z):
    """Returns `x * y + z`, computed with a single rounding."""
    b = ctx.b
    dt = _fma_dtype(x, y, z, semantic.operand_dtype)
    vals = [semantic.to_value(b, v, dt) for v in (x, y, z)]
    shape = ()
    for v in vals:
        shape = semantic.broadcast_shapes(shape, ir.shape_of(v.type))
    vals = [semantic.broadcast_to(b, v, shape) for v in vals]
    return b.create("fma", vals, [vals[0].type]).result


@builtin(interp=lambda x: I.unary("abs", x))
def abs_(ctx, x):
    """Returns the elementwise absolute value."""
    return semantic.unary(ctx.b, "abs", x)


def _i_cdiv(x, div):
    x, div = core.unwrap(x), core.unwrap(div)
    if isinstance(x, int) and isinstance(div, int):
        return -(-x // div)
    return (x + div - 1) // div


@builtin(interp=_i_cdiv)
def cdiv(ctx, x, div):
    """Returns `(x + div - 1) // div`, the number of `div`-sized blocks that cover `x`."""
    x, div = core.unwrap(x), core.unwrap(div)
    if not isinstance(x, ir.Value) and not isinstance(div, ir.Value):
        return -(-x // div)
    return semantic.binary(ctx.b, "floordiv", semantic.binary(ctx.b, "sub", semantic.binary(
        ctx.b, "add", x, div), 1), div)  # fmt: skip


def _i_clamp(x, min, max):
    return I.binary("min", I.binary("max", x, min), max)


@builtin(interp=_i_clamp)
def clamp(ctx, x, min, max):
    """Returns `minimum(maximum(x, min), max)`."""
    return semantic.binary(ctx.b, "min", semantic.binary(ctx.b, "max", x, min), max)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def _i_to(x, dtype, bitcast=False, fp_downcast_rounding=None):
    dt = _dtype(dtype)
    if isinstance(x, IPointer):
        raise CompilationError("pointers can't be converted with .to()")
    t = _itile(x)
    if not bitcast:
        return I.convert(t, dt)
    src = t.dtype
    if src.primitive_bitwidth != dt.primitive_bitwidth or src.is_bool() or dt.is_bool():
        raise CompilationError(
            f"a bitcast needs types of the same width, but {src} has {src.primitive_bitwidth} "
            f"bits and {dt} has {dt.primitive_bitwidth}"
        )
    return ITile(t.data.view(dt.to_numpy()), dt)


def _to(ctx, x, dtype, bitcast=False, fp_downcast_rounding=None):
    """Converts `x` to `dtype`, or reinterprets its bits when `bitcast=True`.

    Float-to-integer conversion truncates toward zero. Conversion to `int1` is `x != 0`.
    """
    if fp_downcast_rounding not in (None, "rtne"):
        raise CompilationError("only round-to-nearest-even float conversion is supported")
    return semantic.cast(ctx.b, _value(ctx, x), _dtype(dtype), bitcast=bool(core.unwrap(bitcast)))


cast = builtin(interp=_i_to, name="cast")(_to)
TILE_METHODS["to"] = core.Builtin("to", _to, _i_to)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def _i_broadcast_to(input, *shape):
    shape = semantic.check_shape(_shape_args(shape), "tl.broadcast_to")
    s = _ishape(input)
    if len(s) > len(shape):
        raise CompilationError(f"can't broadcast shape {s} to the smaller rank of {shape}")
    if semantic.broadcast_shapes(s, shape) != shape:
        raise CompilationError(f"can't broadcast shape {s} to {shape}")
    return _imap(input, lambda a: np.broadcast_to(a, shape))


@builtin(interp=_i_broadcast_to)
def broadcast_to(ctx, input, *shape):
    """Broadcasts `input` to `shape`."""
    shape = semantic.check_shape(_shape_args(shape), "tl.broadcast_to")
    return semantic.broadcast_to(ctx.b, _value(ctx, input), shape)


def _norm_axes(axis: Any, rank_out: int) -> list[int]:
    axes = core.unwrap(axis)
    axes = list(axes) if isinstance(axes, (tuple, list)) else [axes]
    out = sorted(_axis(a, rank_out) for a in axes)
    if len(set(out)) != len(out):
        raise CompilationError(f"repeated axis in {axes}")
    return out


def _i_expand_dims(input, axis):
    s = _ishape(input)
    n = len(core.unwrap(axis)) if isinstance(core.unwrap(axis), (tuple, list)) else 1
    axes = _norm_axes(axis, len(s) + n)
    return _imap(input, lambda a: np.expand_dims(a, tuple(axes)))


@builtin(interp=_i_expand_dims)
def expand_dims(ctx, input, axis):
    """Inserts size-1 dimensions at `axis`, an int or a tuple of ints."""
    v = _value(ctx, input)
    ax = core.unwrap(axis)
    n = len(ax) if isinstance(ax, (tuple, list)) else 1
    for a in _norm_axes(axis, len(ir.shape_of(v.type)) + n):
        v = semantic.expand_dims(ctx.b, v, a)
    return v


def _reshape_shape(src: tuple[int, ...], shape) -> tuple[int, ...]:
    shape = semantic.check_shape(_shape_args(shape), "tl.reshape")
    if int(np.prod(src, dtype=np.int64)) != int(np.prod(shape, dtype=np.int64)):
        raise CompilationError(f"can't reshape a tile of shape {src} to {shape}")
    return shape


def _i_reshape(input, *shape, can_reorder=False):
    shape = _reshape_shape(_ishape(input), shape)
    return _imap(input, lambda a: a.reshape(shape))


@builtin(interp=_i_reshape)
def reshape(ctx, input, *shape, can_reorder=False):
    """Reshapes `input` to `shape`, keeping the row-major element order."""
    v = _value(ctx, input)
    shape = _reshape_shape(ir.shape_of(v.type), shape)
    if ir.shape_of(v.type) == shape:
        return v
    if not isinstance(v.type, ir.TileType):
        return semantic.splat(ctx.b, v, shape)
    return ctx.b.create("reshape", [v], [ir.TileType(shape, v.type.elem)]).result


def _perm(rank: int, dims: tuple[Any, ...], fname: str) -> tuple[int, ...]:
    dims = _shape_args(dims)
    if not dims:
        if fname == "permute":
            raise CompilationError("tl.permute needs the new dimension order")
        return tuple(reversed(range(rank)))
    perm = tuple(_axis(d, rank) for d in dims)
    if sorted(perm) != list(range(rank)):
        raise CompilationError(f"{dims} isn't a permutation of the {rank} dimensions")
    return perm


def _i_trans(input, *dims):
    perm = _perm(len(_ishape(input)), dims, "trans")
    return _imap(input, lambda a: np.transpose(a, perm))


def _trans(ctx, input, *dims, fname="trans"):
    v = _tile_value(ctx, input, fname)
    perm = _perm(len(v.type.shape), dims, fname)
    if perm == tuple(range(len(perm))):
        return v
    t = ir.TileType(tuple(v.type.shape[p] for p in perm), v.type.elem)
    return ctx.b.create("trans", [v], [t], {"perm": list(perm)}).result


@builtin(interp=_i_trans)
def trans(ctx, input, *dims):
    """Permutes the dimensions of `input`. With no `dims`, reverses them."""
    return _trans(ctx, input, *dims)


@builtin(interp=_i_trans)
def permute(ctx, input, *dims):
    """Permutes the dimensions of `input` into the order `dims`."""
    return _trans(ctx, input, *dims, fname="permute")


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _sum_input_dtype(dt: core.dtype) -> core.dtype:
    """Integers narrower than 32 bits sum in 32 bits, as in Triton."""
    if dt.is_int() and dt.primitive_bitwidth < 32:
        return core.uint32 if dt.kind == "uint" else core.int32
    return dt


def _reduce_frontend(ctx, input, axis, kind: str, keep_dims, fname: str, dtype=None):
    b = ctx.b
    v = _tile_value(ctx, input, fname)
    if semantic.is_pointer(v):
        raise CompilationError(f"tl.{fname} can't reduce pointers")
    if dtype is not None:
        v = semantic.cast(b, v, _dtype(dtype))
    if kind == "sum":
        v = semantic.cast(b, v, _sum_input_dtype(semantic.dtype_of(v)))
    rank = len(v.type.shape)
    axis = core.unwrap(axis)
    if axis is None:
        if rank > 1:
            v = b.create("reshape", [v], [ir.TileType((v.type.numel,), v.type.elem)]).result
        ax = 0
    else:
        ax = _axis(axis, rank)
    s = list(v.type.shape)
    del s[ax]
    elem = ir.i32 if kind in ("argmax", "argmin") else v.type.elem
    r = b.create("reduce", [v], [ir.with_shape(s, elem)], {"axis": ax, "kind": kind}).result
    if core.unwrap(keep_dims):
        if axis is None:
            return semantic.splat(b, r, (1,) * rank)
        r = semantic.expand_dims(b, r, ax)
    return r


def _reduce_interp(input, axis, kind: str, keep_dims, dtype=None):
    t = _itile(input)
    if dtype is not None:
        t = I.convert(t, _dtype(dtype))
    if kind == "sum":
        t = I.convert(t, _sum_input_dtype(t.dtype))
    axis = core.unwrap(axis)
    ax = None if axis is None else _axis(axis, len(t.shape))
    keep = bool(core.unwrap(keep_dims))
    data = I.as_compute(t, t.dtype)
    if kind == "sum":
        out = np.sum(data, axis=ax, keepdims=keep, dtype=data.dtype)
    elif kind in ("max", "min"):
        uf = {"max": np.fmax, "min": np.fmin} if t.dtype.is_floating() else {
            "max": np.maximum, "min": np.minimum}  # fmt: skip
        flat = data.reshape(-1) if ax is None else data
        out = uf[kind].reduce(flat, axis=0 if ax is None else ax, keepdims=keep)
        if ax is None and keep:
            out = out.reshape((1,) * len(t.shape))
    else:
        f = np.argmax if kind == "argmax" else np.argmin
        out = f(data, axis=ax, keepdims=keep)
        return ITile(np.asarray(out, dtype=np.int32), core.int32)
    return I.wrap(np.asarray(out), t.dtype)


def _i_sum(input, axis=None, keep_dims=False, dtype=None):
    return _reduce_interp(input, axis, "sum", keep_dims, dtype)


@builtin(interp=_i_sum, name="sum")
def sum_(ctx, input, axis=None, keep_dims=False, dtype=None):
    """Returns the sum along `axis`, or over all elements when `axis` is `None`.

    `float16` and `bfloat16` sums accumulate in `float32` and round the result back.
    Integers narrower than 32 bits sum in 32 bits.
    """
    return _reduce_frontend(ctx, input, axis, "sum", keep_dims, "sum", dtype)


def _minmax_interp(kind):
    def f(input, axis=None, return_indices=False, return_indices_tie_break_left=True,
          keep_dims=False):  # fmt: skip
        v = _reduce_interp(input, axis, kind, keep_dims)
        if return_indices:
            return v, _reduce_interp(input, axis, "arg" + kind, keep_dims)
        return v

    return f


def _minmax_frontend(ctx, kind, input, axis, return_indices, tie_left, keep_dims):
    if not tie_left:
        raise CompilationError("only `return_indices_tie_break_left=True` is supported")
    v = _reduce_frontend(ctx, input, axis, kind, keep_dims, kind)
    if core.unwrap(return_indices):
        return v, _reduce_frontend(ctx, input, axis, "arg" + kind, keep_dims, kind)
    return v


@builtin(interp=_minmax_interp("max"), name="max")
def max_(ctx, input, axis=None, return_indices=False, return_indices_tie_break_left=True,
         keep_dims=False):  # fmt: skip
    """Returns the maximum along `axis`, and the index of the first maximum with
    `return_indices=True`. NaNs are ignored unless every element is NaN."""
    return _minmax_frontend(ctx, "max", input, axis, return_indices,
                            return_indices_tie_break_left, keep_dims)  # fmt: skip


@builtin(interp=_minmax_interp("min"), name="min")
def min_(ctx, input, axis=None, return_indices=False, return_indices_tie_break_left=True,
         keep_dims=False):  # fmt: skip
    """Returns the minimum along `axis`, and the index of the first minimum with
    `return_indices=True`. NaNs are ignored unless every element is NaN."""
    return _minmax_frontend(ctx, "min", input, axis, return_indices,
                            return_indices_tie_break_left, keep_dims)  # fmt: skip


def _argminmax(kind):
    def interp(input, axis, tie_break_left=True, keep_dims=False):
        if not tie_break_left:
            raise CompilationError("only `tie_break_left=True` is supported")
        return _reduce_interp(input, axis, kind, keep_dims)

    def frontend(ctx, input, axis, tie_break_left=True, keep_dims=False):
        if not tie_break_left:
            raise CompilationError("only `tie_break_left=True` is supported")
        return _reduce_frontend(ctx, input, axis, kind, keep_dims, kind)

    frontend.__doc__ = f"Returns the `int32` index of the first {kind[3:]}imum along `axis`."
    return builtin(interp=interp, name=kind)(frontend)


argmax = _argminmax("argmax")
argmin = _argminmax("argmin")


def _i_reduce(input, axis, combine_fn, keep_dims=False):
    single = not isinstance(input, tuple)
    xs = [_itile(x) for x in ((input,) if single else input)]
    shapes = {x.shape for x in xs}
    if len(shapes) != 1:
        raise CompilationError(f"tl.reduce needs inputs of one shape, but got {sorted(shapes)}")
    axis = core.unwrap(axis)
    if axis is None:
        rank = len(xs[0].shape)
        xs = [ITile(x.data.reshape(-1), x.dtype) for x in xs]
        ax = 0
    else:
        ax = _axis(axis, len(xs[0].shape))
    n = xs[0].shape[ax]
    while n > 1:
        h = n // 2
        lo = [slice(None)] * len(xs[0].shape)
        hi = list(lo)
        lo[ax], hi[ax] = slice(0, h), slice(h, n)
        res = combine_fn(*[ITile(x.data[tuple(lo)], x.dtype) for x in xs],
                         *[ITile(x.data[tuple(hi)], x.dtype) for x in xs])  # fmt: skip
        res = res if isinstance(res, tuple) else (res,)
        if len(res) != len(xs):
            raise CompilationError(f"combine_fn returns {len(res)} values for {len(xs)} inputs")
        xs = [I.to_tile(r, x.dtype) for r, x in zip(res, xs, strict=True)]
        n = h
    outs = []
    for x in xs:
        d = x.data if core.unwrap(keep_dims) else np.squeeze(x.data, axis=ax)
        if core.unwrap(keep_dims) and axis is None:
            d = d.reshape((1,) * rank)
        outs.append(ITile(d, x.dtype))
    return outs[0] if single else tuple(outs)


@builtin(interp=_i_reduce)
def reduce(ctx, input, axis, combine_fn, keep_dims=False):
    """Reduces `input` along `axis` with `combine_fn`, a @tegula.jit function.

    `input` can be a tile or a tuple of tiles of the same shape. `combine_fn` takes two
    sets of values, `(a0, ..., b0, ...)`, and returns the combined values. It must be
    associative and commutative.
    """
    from tegula.compiler.frontend import is_jit_function

    b = ctx.b
    fn = core.unwrap(combine_fn)
    if not is_jit_function(fn):
        raise CompilationError("tl.reduce needs a @tegula.jit function as `combine_fn`")
    single = not isinstance(core.unwrap(input), tuple)
    vals = [_tile_value(ctx, x, "reduce") for x in ((input,) if single else core.unwrap(input))]
    shapes = {v.type.shape for v in vals}
    if len(shapes) != 1:
        raise CompilationError(f"tl.reduce needs inputs of one shape, but got {sorted(shapes)}")
    rank = len(vals[0].type.shape)
    axis = core.unwrap(axis)
    if axis is None:
        vals = [b.create("reshape", [v], [ir.TileType((v.type.numel,), v.type.elem)]).result
                for v in vals]  # fmt: skip
        ax = 0
    else:
        ax = _axis(axis, rank)
    elems = [v.type.elem for v in vals]
    if any(not isinstance(e, ir.ScalarType) for e in elems):
        raise CompilationError("tl.reduce can't reduce pointers")
    names = [f"a{i}" for i in range(len(elems))] + [f"b{i}" for i in range(len(elems))]
    block = ir.Block(elems + elems, names)
    with b.at(block):
        res = ctx.call_function(fn, list(block.args), {})
        res = res if isinstance(res, tuple) else (res,)
        if len(res) != len(vals):
            raise CompilationError(f"combine_fn returns {len(res)} values for {len(vals)} inputs")
        b.create("yield", [ctx.materialize(r, e, f"result {i} of combine_fn")
                           for i, (r, e) in enumerate(zip(res, elems, strict=True))])  # fmt: skip
    s = list(vals[0].type.shape)
    del s[ax]
    op = b.create("reduce", vals, [ir.with_shape(s, e) for e in elems], {"axis": ax},
                  regions=[ir.Region(block)])  # fmt: skip
    outs = list(op.results)
    if core.unwrap(keep_dims):
        if axis is None:
            outs = [semantic.splat(b, r, (1,) * rank) for r in outs]
        else:
            outs = [semantic.expand_dims(b, r, ax) for r in outs]
    return outs[0] if single else tuple(outs)


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------

_DOT_DTYPES = (core.float16, core.bfloat16, core.float32)


def _dot_check(a_shape, b_shape, a_dt, b_dt, out_dt, input_precision):
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise CompilationError(f"tl.dot needs 2D tiles, but got shapes {a_shape} and {b_shape}")
    if a_shape[1] != b_shape[0]:
        raise CompilationError(
            f"tl.dot inner dimensions differ: {a_shape} @ {b_shape}. The first operand's "
            "columns must match the second operand's rows."
        )
    if a_dt not in _DOT_DTYPES or b_dt not in _DOT_DTYPES:
        raise CompilationError(
            f"tl.dot supports float16, bfloat16, and float32 operands, but got {a_dt} and {b_dt}"
        )
    if a_dt is not b_dt:
        raise CompilationError(
            f"tl.dot needs operands of the same dtype, but got {a_dt} and {b_dt}. Convert one "
            "with `.to(...)`."
        )
    if out_dt not in (core.float32, core.float16):
        raise CompilationError(f"tl.dot accumulates in float32 or float16, not {out_dt}")
    if input_precision not in (None, "ieee", "tf32", "tf32x3"):
        raise CompilationError(f"unknown input_precision {input_precision!r}")


def _i_dot(input, other, acc=None, input_precision=None, allow_tf32=None,
           max_num_imprecise_acc=None, out_dtype=core.float32):  # fmt: skip
    a, bt = _itile(input), _itile(other)
    out_dt = _dtype(out_dtype, "out_dtype") if acc is None else _itile(acc).dtype
    _dot_check(a.shape, bt.shape, a.dtype, bt.dtype, out_dt, input_precision)
    m, n = a.shape[0], bt.shape[1]
    c = np.zeros((m, n), np.float32) if acc is None else I.as_compute(acc, out_dt)
    if c.shape != (m, n):
        raise CompilationError(f"the tl.dot accumulator must be {m}x{n}, but got {c.shape}")
    r = a.data.astype(np.float32) @ bt.data.astype(np.float32) + c
    return I.wrap(r, out_dt)


@builtin(interp=_i_dot)
def dot(ctx, input, other, acc=None, input_precision=None, allow_tf32=None,
        max_num_imprecise_acc=None, out_dtype=core.float32):  # fmt: skip
    """Returns `input @ other + acc`.

    Operands are `float16`, `bfloat16`, or `float32` tiles of the same dtype. The result
    accumulates in `out_dtype` (`float32` by default) or in the dtype of `acc`. `float32`
    operands compute in full `float32`; Apple GPUs have no TF32, so `input_precision` has
    no effect.
    """
    b = ctx.b
    av, bv = _value(ctx, input), _value(ctx, other)
    acc = core.unwrap(acc)
    out_dt = _dtype(out_dtype, "out_dtype")
    if acc is not None:
        if not isinstance(acc, ir.Value):
            raise CompilationError(f"the tl.dot accumulator must be a tile, got {describe(acc)}")
        out_dt = semantic.dtype_of(acc)
    _dot_check(ir.shape_of(av.type), ir.shape_of(bv.type), semantic.dtype_of(av),
               semantic.dtype_of(bv), out_dt, input_precision)  # fmt: skip
    m, n = av.type.shape[0], bv.type.shape[1]
    if acc is None:
        acc = b.create("full", [], [ir.TileType((m, n), ir.scalar(out_dt))], {"value": 0.0}).result
    elif ir.shape_of(acc.type) != (m, n):
        raise CompilationError(f"the tl.dot accumulator must be {m}x{n}, but got {acc.type}")
    return b.create("dot", [av, bv, acc], [acc.type]).result


# ---------------------------------------------------------------------------
# Compile time
# ---------------------------------------------------------------------------


def range_bounds(arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None,
                 disallow_acc_multi_buffer=False, flatten=False,
                 warp_specialize=False):  # fmt: skip
    """Normalizes `tl.range` arguments to a list of 1 to 3 range bounds.

    The pipelining and unrolling hints are accepted for Triton compatibility and have no
    effect.
    """
    return [a for a in (arg1, arg2, step) if a is not None]


def _loop_only(name):
    def frontend(ctx, *args, **kwargs):
        raise CompilationError(f"tl.{name} is supported only as the iterable of a `for` loop")

    return frontend


def _i_static_range(arg1, arg2=None, step=None):
    args = [core.unwrap(a) for a in (arg1, arg2, step) if a is not None]
    if any(isinstance(a, ITile) for a in args):
        raise CompilationError("tl.static_range needs compile-time integer bounds")
    return builtins.range(*args)


static_range = builtin(interp=_i_static_range, name="static_range")(_loop_only("static_range"))
static_range.__doc__ = "Like `range`, but the frontend unrolls the loop at compile time."

range_ = builtin(
    interp=lambda *a, **k: builtins.range(*[int(x) for x in range_bounds(*a, **k)]),
    name="range",
)(_loop_only("range"))
range_.__doc__ = "Like `range`. Pipelining hints such as `num_stages` have no effect."


def _static_assert_msg(msg: str) -> str:
    return f"static assertion failed: {msg}" if msg else "static assertion failed"


def _i_static_assert(cond, msg=""):
    if isinstance(cond, ITile):
        raise CompilationError("tl.static_assert needs a compile-time condition")
    if not cond:
        raise CompilationError(_static_assert_msg(msg))


@builtin(interp=_i_static_assert)
def static_assert(ctx, cond, msg=""):
    """Raises a `CompilationError` with `msg` if the compile-time `cond` is false."""
    cond = core.unwrap(cond)
    if isinstance(cond, ir.Value):
        raise CompilationError("tl.static_assert needs a compile-time condition")
    if not cond:
        raise CompilationError(_static_assert_msg(msg))


def _fmt_static(v: Any) -> str:
    v = core.unwrap(v)
    if isinstance(v, ir.Value):
        return f"<{v.type}>"
    if isinstance(v, ITile):
        return f"<{v.dtype}{list(v.shape)}>"
    return str(v)


def _i_static_print(*values, sep=" "):
    if INTERP.program_id == (0, 0, 0):
        print(*[_fmt_static(v) for v in values], sep=sep)


@builtin(interp=_i_static_print)
def static_print(ctx, *values, sep=" "):
    """Prints compile-time values while the kernel compiles. Runtime values print their type."""
    print(*[_fmt_static(v) for v in values], sep=sep)


# ---------------------------------------------------------------------------
# Tile methods
# ---------------------------------------------------------------------------

for _b in (reshape, trans, permute, broadcast_to, expand_dims, argmax, argmin, reduce, cast):
    _method(_b)
TILE_METHODS["sum"] = sum_
TILE_METHODS["max"] = max_
TILE_METHODS["min"] = min_
TILE_METHODS["abs"] = abs_
