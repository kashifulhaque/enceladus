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

from enceladus.compiler import ir, semantic
from enceladus.compiler.errors import CompilationError
from enceladus.compiler.semantic import describe
from enceladus.interpreter import interp as I  # noqa: N812
from enceladus.interpreter.interp import IDesc, IPointer, ITile
from enceladus.language import core
from enceladus.language.core import DESC_METHODS, INTERP, TILE_METHODS, builtin

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
        hint = " Pass a Python int, or a tl.constexpr parameter that holds one."
        if isinstance(v, float) and v.is_integer():
            hint = f" Pass the int {int(v)} instead of the float {v!r}."
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
        raise CompilationError(
            f"axis {a} is out of range for a tile of rank {rank}. Use an axis from {-rank} to "
            f"{rank - 1}."
        )
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
        raise CompilationError(
            f"`axis` must be 0, 1, or 2, because the grid has at most three dimensions, but "
            f"got {a}"
        )
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
        raise CompilationError(
            f"tl.arange({s}, {e}) needs `end` greater than `start`. Swap the bounds, or "
            "subtract the tile from a scalar to count down."
        )
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
        raise CompilationError(
            f"tl.full needs a scalar value, but got {describe(v)}. To give a tile a new "
            "shape, use tl.broadcast_to."
        )
    return ITile(np.broadcast_to(I.to_tile(v, dt).data, shape).copy(), dt)


@builtin(interp=_i_full)
def full(ctx, shape, value, dtype):
    """Returns a tile of `shape` and `dtype` filled with the scalar `value`.

    `shape` holds compile-time powers of two. `value` can be a literal or a runtime
    scalar.
    """
    shape = semantic.check_shape(shape, "tl.full")
    dt = _dtype(dtype)
    v = core.unwrap(value)
    if isinstance(v, ir.Value):
        if isinstance(v.type, ir.TileType):
            raise CompilationError(
                f"tl.full needs a scalar value, but got {describe(v)}. To give a tile a new "
                "shape, use tl.broadcast_to."
            )
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

_PTR_HINT = " Pass an array as the kernel argument, and add offsets to it, as in `x_ptr + offs`."

_BLOCK_PTR_MSG = (
    "`boundary_check` and `padding_option` apply to block pointers, which Enceladus doesn't "
    "support. Use tl.make_tensor_descriptor for bounds-checked block loads."
)


def _check_mask_dtype(dt: core.dtype) -> None:
    if not dt.is_bool():
        raise CompilationError(
            f"a mask must be a boolean (int1) tile, such as `offs < n`, but got {dt}. Build "
            "the mask with a comparison, for example `x != 0`."
        )


def _i_load(pointer, mask=None, other=None, boundary_check=(), padding_option="",
            cache_modifier="", eviction_policy="", volatile=False):  # fmt: skip
    if not isinstance(pointer, IPointer):
        raise CompilationError(f"tl.load needs a pointer or pointer tile, but got "
                               f"{describe(pointer)}.{_PTR_HINT}")
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

    `mask` and `other` broadcast to the pointer's shape. Where `mask` is false, the result
    is `other`, or 0 when `other` isn't given, and no memory is read. `cache_modifier`,
    `eviction_policy`, and `volatile` are accepted for Triton compatibility and have no
    effect. `boundary_check` and `padding_option` apply to block pointers, which Enceladus
    doesn't support; use `tl.make_tensor_descriptor` instead.
    """
    b = ctx.b
    p = core.unwrap(pointer)
    if not semantic.is_pointer(p):
        raise CompilationError(
            f"tl.load needs a pointer or pointer tile, but got {describe(p)}.{_PTR_HINT}"
        )
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
        raise CompilationError(f"tl.store needs a pointer or pointer tile, but got "
                               f"{describe(pointer)}.{_PTR_HINT}")
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
        raise CompilationError(
            f"tl.store needs a pointer or pointer tile, but got {describe(p)}.{_PTR_HINT}"
        )
    if boundary_check:
        raise CompilationError(_BLOCK_PTR_MSG)
    elem = ir.elem_of(p.type).elem.dtype
    v = core.unwrap(value)
    if semantic.is_pointer(v):
        raise CompilationError("tl.store can't store pointers. Store integer offsets instead.")
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
        raise CompilationError(
            f"tl.make_tensor_descriptor needs a scalar base pointer, but got {describe(base)}"
        )
    if int(strides[-1]) != 1:
        raise CompilationError(_LAST_STRIDE_MSG)
    return IDesc(base, shape, strides, block)


@builtin(interp=_i_make_tensor_descriptor)
def make_tensor_descriptor(ctx, base, shape, strides, block_shape, padding_option="zero"):
    """Creates a tensor descriptor over `base` with the given shape and strides.

    `base` is a scalar pointer. `shape` and `strides` are integer scalars, in elements,
    and the last stride must be 1, so the innermost dimension is contiguous.
    `block_shape` holds compile-time powers of two.

    `desc.load(offsets)` returns the `block_shape` block at element offsets `offsets` and
    fills out-of-bounds elements with zero. `desc.store(offsets, value)` skips
    out-of-bounds elements. `desc.dtype` is the element type, and `desc.block_shape` is
    the block shape. Only zero padding (`padding_option="zero"`) is supported.
    """
    b = ctx.b
    shape, strides, block = _desc_parts(base, shape, strides, block_shape)
    base = core.unwrap(base)
    if not (semantic.is_pointer(base) and isinstance(base.type, ir.PointerType)):
        raise CompilationError(
            f"tl.make_tensor_descriptor needs a scalar base pointer, but got {describe(base)}"
        )
    if padding_option not in ("zero", "", None):
        raise CompilationError(
            'tensor descriptors support only zero padding. Omit `padding_option`, or pass '
            '"zero".'
        )
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
        if not (isinstance(x, IPointer) and isinstance(y, IPointer) and x.flat is y.flat):
            raise CompilationError(
                "tl.where selects between pointers only when both derive from the same base "
                "pointer, as in `tl.where(c, x_ptr + i, x_ptr + j)`. Otherwise, select integer "
                "offsets, and add the result to one base pointer."
            )
        semantic.broadcast_shapes(semantic.broadcast_shapes(c.shape, x.shape), y.shape)
        return x.with_offsets(np.where(c.data, x.offsets, y.offsets))
    dt = semantic.computation_dtype("select", I._operand(x), I._operand(y))
    semantic.broadcast_shapes(semantic.broadcast_shapes(c.shape, _ishape(x)), _ishape(y))
    return I.wrap(np.where(c.data, I.as_compute(x, dt), I.as_compute(y, dt)), dt)


@builtin(interp=_i_where)
def where(ctx, condition, x, y):
    """Returns `x` where `condition` is true and `y` elsewhere, elementwise.

    The three arguments broadcast to one shape. With a compile-time `condition`, the
    result is `x` or `y` itself.
    """
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
        raise CompilationError(
            f"tl.fma needs floating-point operands, but they promote to {dt}. Convert them "
            "with `.to(tl.float32)` first."
        )
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
        raise CompilationError(
            "pointers can't be converted with .to(). Load the values with tl.load, and then "
            "convert them."
        )
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

    The tile method `x.to(dtype)` does the same. Float-to-integer conversion truncates
    toward zero. For a value out of the integer type's range, both execution modes give
    what Metal gives on Apple GPUs: NaN becomes 0; `int64` wraps the truncated value
    modulo 2**64 and turns infinities into 0; and every other integer type saturates to
    its minimum or maximum. For example, -2.5 converts to 0 as `uint8` and to -2 as
    `int8`, and 300.0 converts to 255 as `uint8`. Conversion to `int1` is `x != 0`. A
    bitcast needs types of the same width. `fp_downcast_rounding` accepts only `None`
    and `"rtne"`.
    """
    if fp_downcast_rounding not in (None, "rtne"):
        raise CompilationError(
            "only round-to-nearest-even float conversion is supported. Omit "
            '`fp_downcast_rounding`, or pass "rtne".'
        )
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
        raise CompilationError(f"the axes {axes} name an axis more than once. List each axis once.")
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
        raise CompilationError(
            f"can't reshape a tile of shape {src} to {shape}. The new shape must have the same "
            "number of elements."
        )
    return shape


def _i_reshape(input, *shape, can_reorder=False):
    shape = _reshape_shape(_ishape(input), shape)
    return _imap(input, lambda a: a.reshape(shape))


@builtin(interp=_i_reshape)
def reshape(ctx, input, *shape, can_reorder=False):
    """Reshapes `input` to `shape`, keeping the row-major element order.

    Every dimension of `shape` must be a power of two, and the element count must not
    change. `can_reorder` is accepted for Triton compatibility; the order is always kept.
    """
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
            raise CompilationError(
                "tl.permute needs the new dimension order, for example `tl.permute(x, (1, 0))`"
            )
        return tuple(reversed(range(rank)))
    perm = tuple(_axis(d, rank) for d in dims)
    if sorted(perm) != list(range(rank)):
        raise CompilationError(
            f"{dims} isn't a permutation of the {rank} dimensions. Name each dimension from 0 "
            f"to {rank - 1} exactly once."
        )
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
        raise CompilationError(f"tl.{fname} can't reduce pointers. Reduce integer offsets instead.")
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
        return I.wrap(np.sum(data, axis=ax, keepdims=keep, dtype=data.dtype), t.dtype)
    flat = data.reshape(-1) if ax is None else data
    a = 0 if ax is None else ax
    uf = {"max": np.fmax, "min": np.fmin} if t.dtype.is_floating() else {
        "max": np.maximum, "min": np.minimum}  # fmt: skip
    out = uf[kind.removeprefix("arg")].reduce(flat, axis=a, keepdims=True)
    if kind in ("argmax", "argmin"):
        # The index of the first element equal to the NaN-ignoring max or min, so the index
        # matches the value that tl.max and tl.min return. All-NaN slices give index 0.
        hit = flat == out
        if t.dtype.is_floating():
            hit |= np.isnan(flat) & np.isnan(out)
        out = np.argmax(hit, axis=a, keepdims=True).astype(np.int32)
    if not keep:
        out = np.squeeze(out, axis=a)
    elif ax is None:
        out = out.reshape((1,) * len(t.shape))
    if kind in ("argmax", "argmin"):
        return ITile(out, core.int32)
    return I.wrap(out, t.dtype)


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
        raise CompilationError(
            "only `return_indices_tie_break_left=True` is supported. Omit the argument."
        )
    v = _reduce_frontend(ctx, input, axis, kind, keep_dims, kind)
    if core.unwrap(return_indices):
        return v, _reduce_frontend(ctx, input, axis, "arg" + kind, keep_dims, kind)
    return v


@builtin(interp=_minmax_interp("max"), name="max")
def max_(ctx, input, axis=None, return_indices=False, return_indices_tie_break_left=True,
         keep_dims=False):  # fmt: skip
    """Returns the maximum along `axis`, or over all elements when `axis` is `None`.

    With `return_indices=True`, also returns the `int32` index of the first maximum. NaNs
    are ignored unless every element is NaN.
    """
    return _minmax_frontend(ctx, "max", input, axis, return_indices,
                            return_indices_tie_break_left, keep_dims)  # fmt: skip


@builtin(interp=_minmax_interp("min"), name="min")
def min_(ctx, input, axis=None, return_indices=False, return_indices_tie_break_left=True,
         keep_dims=False):  # fmt: skip
    """Returns the minimum along `axis`, or over all elements when `axis` is `None`.

    With `return_indices=True`, also returns the `int32` index of the first minimum. NaNs
    are ignored unless every element is NaN.
    """
    return _minmax_frontend(ctx, "min", input, axis, return_indices,
                            return_indices_tie_break_left, keep_dims)  # fmt: skip


def _argminmax(kind):
    def interp(input, axis, tie_break_left=True, keep_dims=False):
        if not tie_break_left:
            raise CompilationError("only `tie_break_left=True` is supported. Omit the argument.")
        return _reduce_interp(input, axis, kind, keep_dims)

    def frontend(ctx, input, axis, tie_break_left=True, keep_dims=False):
        if not tie_break_left:
            raise CompilationError("only `tie_break_left=True` is supported. Omit the argument.")
        return _reduce_frontend(ctx, input, axis, kind, keep_dims, kind)

    frontend.__doc__ = (
        f"Returns the `int32` index of the first {kind[3:]}imum along `axis`.\n\n"
        "Only `tie_break_left=True` is supported. With `keep_dims=True`, the reduced axis "
        "stays as a dimension of size 1.\n"
    )
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
            raise CompilationError(
                f"combine_fn returns {len(res)} values for {len(xs)} inputs. Return one value "
                "per input."
            )
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
    """Reduces `input` along `axis` with `combine_fn`, a @enceladus.jit function.

    `input` can be a tile or a tuple of tiles of the same shape. `combine_fn` takes two
    sets of values, `(a0, ..., b0, ...)`, and returns the combined values. It must be
    associative and commutative.
    """
    from enceladus.compiler.frontend import is_jit_function

    b = ctx.b
    fn = core.unwrap(combine_fn)
    if not is_jit_function(fn):
        raise CompilationError(
            "tl.reduce needs a @enceladus.jit function as `combine_fn`. Decorate the combine "
            "function with @enceladus.jit."
        )
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
        raise CompilationError("tl.reduce can't reduce pointers. Reduce integer offsets instead.")
    names = [f"a{i}" for i in range(len(elems))] + [f"b{i}" for i in range(len(elems))]
    block = ir.Block(elems + elems, names)
    with b.at(block):
        res = ctx.call_function(fn, list(block.args), {})
        res = res if isinstance(res, tuple) else (res,)
        if len(res) != len(vals):
            raise CompilationError(
                f"combine_fn returns {len(res)} values for {len(vals)} inputs. Return one value "
                "per input."
            )
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

_DOT_FLOAT = (core.float16, core.bfloat16, core.float32)
_DOT_INT = (core.int8, core.uint8, core.int16, core.uint16, core.int32, core.uint32)


def _dot_out_dtype(a_dt: core.dtype, out_dtype: Any) -> core.dtype:
    """Returns the accumulator dtype of a `tl.dot` without an accumulator.

    Float operands accumulate in `out_dtype`, `float32` by default. Integer operands
    accumulate in `int32`, as in Triton, so `out_dtype` must be `None` or `int32`.
    """
    if not a_dt.is_int():
        return core.float32 if out_dtype is None else _dtype(out_dtype, "out_dtype")
    out_dt = core.int32 if out_dtype is None else _dtype(out_dtype, "out_dtype")
    if out_dt is not core.int32:
        raise CompilationError(
            f"tl.dot accumulates {a_dt} operands in int32, not {out_dt}. Omit `out_dtype`, "
            "or pass `out_dtype=tl.int32`, and convert the result afterward."
        )
    return out_dt


def _dot_check(a_shape, b_shape, a_dt, b_dt, out_dt, input_precision):
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise CompilationError(
            f"tl.dot needs 2D tiles, but got shapes {a_shape} and {b_shape}. Reshape the "
            "operands to 2D with tl.reshape."
        )
    if a_shape[1] != b_shape[0]:
        raise CompilationError(
            f"tl.dot inner dimensions differ: {a_shape} @ {b_shape}. The first operand's "
            "columns must match the second operand's rows."
        )
    supported = (*_DOT_FLOAT, *_DOT_INT)
    if a_dt not in supported or b_dt not in supported:
        raise CompilationError(
            f"tl.dot supports float16, bfloat16, float32, and 8-, 16-, and 32-bit integer "
            f"operands, but got {a_dt} and {b_dt}. Convert the operands with "
            "`.to(tl.float32)` or `.to(tl.int32)`."
        )
    if a_dt is not b_dt:
        raise CompilationError(
            f"tl.dot needs operands of the same dtype, but got {a_dt} and {b_dt}. Convert one "
            "with `.to(...)`."
        )
    if a_dt.is_int():
        if out_dt is not core.int32:
            raise CompilationError(
                f"tl.dot accumulates {a_dt} operands in int32, not {out_dt}. Create the "
                "accumulator with `tl.zeros(..., dtype=tl.int32)`, or omit it."
            )
    elif out_dt not in (core.float32, core.float16):
        raise CompilationError(
            f"tl.dot accumulates float operands in float32 or float16, not {out_dt}. Pass "
            "`out_dtype=tl.float32`, or an accumulator of one of those types."
        )
    if input_precision not in (None, "ieee", "tf32", "tf32x3"):
        raise CompilationError(
            f'unknown input_precision {input_precision!r}. Use "ieee", "tf32", or "tf32x3".'
        )


def _i_dot(input, other, acc=None, input_precision=None, allow_tf32=None,
           max_num_imprecise_acc=None, out_dtype=None):  # fmt: skip
    a, bt = _itile(input), _itile(other)
    out_dt = _dot_out_dtype(a.dtype, out_dtype) if acc is None else _itile(acc).dtype
    _dot_check(a.shape, bt.shape, a.dtype, bt.dtype, out_dt, input_precision)
    m, n = a.shape[0], bt.shape[1]
    if acc is not None and _itile(acc).shape != (m, n):
        raise CompilationError(
            f"the tl.dot accumulator must be {m}x{n}, but got {_itile(acc).shape}. Create it "
            f"with `tl.zeros(({m}, {n}), dtype={out_dt!r})`."
        )
    if out_dt.is_int():
        # Exact products and sums, wrapped to int32 as the GPU wraps them.
        c = np.zeros((m, n), np.int64) if acc is None else _itile(acc).data.astype(np.int64)
        r = a.data.astype(np.int64) @ bt.data.astype(np.int64) + c
        return I.wrap(r, out_dt)
    c = np.zeros((m, n), np.float32) if acc is None else I.as_compute(acc, out_dt)
    r = a.data.astype(np.float32) @ bt.data.astype(np.float32) + c
    return I.wrap(r, out_dt)


@builtin(interp=_i_dot)
def dot(ctx, input, other, acc=None, input_precision=None, allow_tf32=None,
        max_num_imprecise_acc=None, out_dtype=None):  # fmt: skip
    """Returns `input @ other + acc` for 2D tiles.

    Operands are tiles of the same dtype: `float16`, `bfloat16`, `float32`, or an 8-, 16-,
    or 32-bit integer type.

    - Float operands accumulate in `out_dtype` (`float32` by default) or in the dtype of
      `acc`, which must be `float32` or `float16`. `float32` operands compute in full
      `float32`; Apple GPUs have no TF32, so `input_precision`, `allow_tf32`, and
      `max_num_imprecise_acc` have no effect.
    - Integer operands accumulate in `int32`, as in Triton, and the result is exact:
      products and sums wrap modulo 2^32. `int8` and `uint8` operands run on the
      `simdgroup_matrix` units, at about 85% of the `float16` rate on an M4 Pro.
      16-bit and 32-bit integer operands run as scalar multiply-adds, at about a third
      of the `int8` rate.

    The K dimension must be a multiple of 8. The launch option `dot_warps=(WM, WN)`
    splits the M x N result over the kernel's SIMD groups, and each SIMD group's strip
    must be a multiple of 8 in both dimensions. The launch option `dot_backend` selects
    the lowering: `simdgroup_matrix` code, or Metal 4 `matmul2d` for eligible float
    loops.
    """
    b = ctx.b
    av, bv = _value(ctx, input), _value(ctx, other)
    acc = core.unwrap(acc)
    if acc is not None:
        if not isinstance(acc, ir.Value):
            raise CompilationError(
                f"the tl.dot accumulator must be a tile, but got {describe(acc)}. Create it "
                "with tl.zeros."
            )
        out_dt = semantic.dtype_of(acc)
    else:
        out_dt = _dot_out_dtype(semantic.dtype_of(av), core.unwrap(out_dtype))
    _dot_check(ir.shape_of(av.type), ir.shape_of(bv.type), semantic.dtype_of(av),
               semantic.dtype_of(bv), out_dt, input_precision)  # fmt: skip
    m, n = av.type.shape[0], bv.type.shape[1]
    if acc is None:
        zero = 0 if out_dt.is_int() else 0.0
        tt = ir.TileType((m, n), ir.scalar(out_dt))
        acc = b.create("full", [], [tt], {"value": zero}).result
    elif ir.shape_of(acc.type) != (m, n):
        raise CompilationError(
            f"the tl.dot accumulator must be {m}x{n}, but got {describe(acc)}. Create it with "
            f"`tl.zeros(({m}, {n}), dtype={out_dt!r})`."
        )
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
        raise CompilationError(
            "tl.static_range needs compile-time integer bounds. Use literals or "
            "tl.constexpr parameters, or use range(...) for a runtime loop."
        )
    return builtins.range(*args)


static_range = builtin(interp=_i_static_range, name="static_range")(_loop_only("static_range"))
static_range.__doc__ = """Returns a loop range that the compiler unrolls, for use in a `for` loop.

The bounds and step must be compile-time integers. Each iteration compiles separately,
so the loop variable is a compile-time integer inside the body.
"""

range_ = builtin(
    interp=lambda *a, **k: I.typed_range(*range_bounds(*a, **k)),
    name="range",
)(_loop_only("range"))
range_.__doc__ = """Returns a runtime loop range, like Python's `range`, for use in a `for` loop.

`for i in tl.range(start, end, step)` compiles to the same loop as `range(...)`. The
Triton hints `num_stages`, `loop_unroll_factor`, `disallow_acc_multi_buffer`, `flatten`,
and `warp_specialize` are accepted and have no effect.
"""


def _static_assert_msg(msg: str) -> str:
    return f"static assertion failed: {msg}" if msg else "static assertion failed"


def _i_static_assert(cond, msg=""):
    if isinstance(cond, ITile):
        raise CompilationError(
            "tl.static_assert needs a compile-time condition. For a runtime check, use "
            "tl.device_assert."
        )
    if not cond:
        raise CompilationError(_static_assert_msg(msg))


@builtin(interp=_i_static_assert)
def static_assert(ctx, cond, msg=""):
    """Raises a `CompilationError` with `msg` if the compile-time `cond` is false."""
    cond = core.unwrap(cond)
    if isinstance(cond, ir.Value):
        raise CompilationError(
            "tl.static_assert needs a compile-time condition. For a runtime check, use "
            "tl.device_assert."
        )
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
# Device printing and asserts
# ---------------------------------------------------------------------------


def _print_prefix_arg(prefix: Any, hex: Any) -> str:
    prefix = core.unwrap(prefix)
    if not isinstance(prefix, str):
        raise CompilationError(
            f"tl.device_print needs a string prefix as its first argument, but got "
            f"{describe(prefix)}. Write, for example, `tl.device_print(\"x\", x)`."
        )
    if not all(" " <= ch <= "~" for ch in prefix):
        raise CompilationError(
            "the tl.device_print prefix must be printable ASCII text. Remove other characters, "
            "such as newlines, from the prefix."
        )
    if core.unwrap(hex):
        raise CompilationError("tl.device_print doesn't support hex=True. Omit `hex`.")
    return prefix


def _i_device_print(prefix, *args, hex=False):
    from enceladus.compiler.codegen.debug import format_line, format_value

    prefix = _print_prefix_arg(prefix, hex)
    tiles = []
    for a in args:
        if isinstance(a, IPointer):
            raise CompilationError("tl.device_print can't print pointers. Print offsets instead.")
        tiles.append(I.to_tile(core.unwrap(a)))
    shape: tuple[int, ...] = ()
    for t in tiles:
        shape = semantic.broadcast_shapes(shape, t.shape)
    pid = INTERP.program_id
    data = [np.broadcast_to(t.data, shape) for t in tiles]
    names = [t.dtype.ir_name for t in tiles]
    lines = []
    for idx in np.ndindex(*shape) if shape else [None]:
        k = idx if idx is not None else ()
        vals = [format_value(d[k], n) for d, n in zip(data, names, strict=True)]
        lines.append(format_line(pid, idx, prefix, vals) + "\n")
    import sys

    sys.stderr.write("".join(lines))


@builtin(interp=_i_device_print)
def device_print(ctx, prefix, *args, hex=False):
    """Prints runtime values from the GPU, one line per element, to `sys.stderr`.

    Each line starts with the program ID, then the element's index for tiles, then the
    prefix and the values, for example `pid (0, 0, 0) idx (3) x: 1.500000`. Tile
    arguments broadcast to one shape, and each element prints once. Lines arrive when
    the stream synchronizes, in no particular order. Printing is incompatible with
    `enceladus.capture`.
    """
    b = ctx.b
    prefix = _print_prefix_arg(prefix, hex)
    vals = []
    for a in args:
        a = core.unwrap(a)
        if semantic.is_pointer(a):
            raise CompilationError("tl.device_print can't print pointers. Print offsets instead.")
        vals.append(semantic.to_value(b, a))
    shape: tuple[int, ...] = ()
    for v in vals:
        shape = semantic.broadcast_shapes(shape, ir.shape_of(v.type))
    if shape:
        vals = [semantic.broadcast_to(b, v, shape) if isinstance(v.type, ir.TileType) else v
                for v in vals]  # fmt: skip
    b.create("print", vals, [], {"prefix": prefix})


def _assert_message(msg: Any) -> str:
    msg = core.unwrap(msg)
    if not isinstance(msg, str):
        raise CompilationError(
            f"the tl.device_assert message must be a string, not {describe(msg)}"
        )
    return msg


def _i_device_assert(cond, msg="", mask=None):
    msg = _assert_message(msg)
    if not core.debug_enabled():
        return
    c = I.to_tile(core.unwrap(cond))
    if not c.dtype.is_bool():
        raise CompilationError(f"tl.device_assert needs a boolean condition, but got {c.dtype}")
    failed = ~c.data.astype(bool)
    mask = core.unwrap(mask)
    if mask is not None:
        m = _itile(mask)
        _check_mask_dtype(m.dtype)
        failed, mb = np.broadcast_arrays(failed, m.data.astype(bool))
        failed = failed & mb
    if np.any(failed):
        from enceladus.compiler.errors import DeviceAssertionError

        raise DeviceAssertionError(msg, I.kernel_loc(), INTERP.program_id)


@builtin(interp=_i_device_assert)
def device_assert(ctx, cond, msg="", mask=None):
    """Checks `cond` on the GPU and raises `enceladus.DeviceAssertionError` if it fails.

    Asserts are active only when the environment sets `ENCELADUS_DEBUG=1`; otherwise the
    compiler removes them. A failing assert records the first failing program and the
    assert's source line, and the error is raised at the next synchronization. The
    kernel keeps running after a failure, so guard dangerous accesses with a mask too.
    Where `mask` is false, `cond` isn't checked. The interpreter raises immediately.
    """
    msg = _assert_message(msg)
    if not getattr(ctx, "debug", False):
        return
    b = ctx.b
    c = core.unwrap(cond)
    if isinstance(c, (bool, np.bool_)):
        c = semantic.const(b, bool(c), core.int1)
    if not isinstance(c, ir.Value) or semantic.dtype_of(c) is not core.int1:
        what = describe(c)
        raise CompilationError(
            f"tl.device_assert needs a boolean condition, such as `offs < n`, but got {what}"
        )
    ops = [c]
    mask = core.unwrap(mask)
    if mask is not None and mask is not True:
        if not isinstance(mask, ir.Value):
            raise CompilationError(f"`mask` must be a boolean tile, but got {describe(mask)}")
        _check_mask_dtype(semantic.dtype_of(mask))
        shape = semantic.broadcast_shapes(ir.shape_of(c.type), ir.shape_of(mask.type))
        ops = [semantic.broadcast_to(b, c, shape) if shape else c,
               semantic.broadcast_to(b, mask, shape) if shape else mask]  # fmt: skip
    b.create("assert", ops, [], {"msg": msg})


def _i_debug_barrier():
    # The interpreter runs each operation of a program on every element before the next
    # operation starts, so stores are already visible to later loads.
    return None


@builtin(interp=_i_debug_barrier)
def debug_barrier(ctx):
    """Waits for every thread of the program and orders their device memory accesses.

    After the barrier, every thread of the program sees the device memory stores that any
    thread of the same program made before it. Use it when a program loads addresses that
    it stored earlier through a different arrangement of elements, such as a loop that
    stores a tile and loads it back transposed. The barrier doesn't order memory between
    programs; use atomics for that. The interpreter doesn't need it and ignores it.
    """
    ctx.b.create("barrier", [], [])


# ---------------------------------------------------------------------------
# Tile methods
# ---------------------------------------------------------------------------

for _b in (reshape, trans, permute, broadcast_to, expand_dims, argmax, argmin, reduce, cast):
    _method(_b)
TILE_METHODS["sum"] = sum_
TILE_METHODS["max"] = max_
TILE_METHODS["min"] = min_
TILE_METHODS["abs"] = abs_


# ---------------------------------------------------------------------------
# Atomics
# ---------------------------------------------------------------------------

_ATOMIC_BITWISE = ("and", "or", "xor")


def _check_sem(sem: Any, scope: Any, fname: str) -> None:
    """Accepts only relaxed ordering at GPU or threadgroup scope."""
    sem, scope = core.unwrap(sem), core.unwrap(scope)
    if sem not in (None, "relaxed"):
        raise CompilationError(
            f"tl.{fname} supports only relaxed memory ordering, but got sem={sem!r}. Metal "
            'device atomics are relaxed; omit `sem` or pass sem="relaxed".'
        )
    if scope not in (None, "gpu", "cta"):
        raise CompilationError(
            f'tl.{fname} supports scope="gpu" and scope="cta", but got scope={scope!r}. '
            'Omit `scope`, or pass one of those values.'
        )


# What Metal offers for 64-bit atomics, as probed on macOS 27 (MSL 3.2 to 4.1, Apple9):
# `atomic_max_explicit` and `atomic_min_explicit` on `device atomic_ulong`, which return
# void. There's no `atomic_long`, and no 64-bit fetch, exchange, load, store, or
# compare-and-swap, even through the compiler's `__metal_atomic_*` builtins. Without a
# 64-bit compare-and-swap, no other 64-bit atomic can be built.
_ATOMIC64_LIMITS = (
    "Metal's only 64-bit atomics are uint64 tl.atomic_max and tl.atomic_min, which return "
    "no old value and need an Apple9 GPU (M3 or later). Metal has no 64-bit "
    "compare-and-swap, so Enceladus can't build other 64-bit atomics from one."
)
_ATOMIC64_HINT = {
    "add": "Use int32 or uint32 elements. For a 64-bit sum, keep the low and high halves "
           "in two uint32 buffers: `old = tl.atomic_add(lo_ptr, lo)`, then add "
           "`hi + (old + lo < old)` to the high half. The pair is exact after the kernel "
           "ends.",
    "max": "For int64, store each value as uint64 with its sign bit flipped "
           "(`x ^ (1 << 63)`), which keeps the order, use uint64 tl.atomic_max, and flip "
           "the bit back afterward.",
    "min": "For int64, store each value as uint64 with its sign bit flipped "
           "(`x ^ (1 << 63)`), which keeps the order, use uint64 tl.atomic_min, and flip "
           "the bit back afterward.",
    "and": "Use 32-bit elements, or apply the operation to each uint32 half through a "
           "uint32 view of the buffer.",
    "or": "Use 32-bit elements, or apply the operation to each uint32 half through a "
          "uint32 view of the buffer.",
    "xor": "Use 32-bit elements, or apply the operation to each uint32 half through a "
           "uint32 view of the buffer.",
    "xchg": "Use 32-bit elements.",
    "cas": "Use 32-bit elements.",
}


def check_atomic_dtype(kind: str, dt: core.dtype, fname: str) -> None:
    """Raises an error when no backend can run atomic `kind` on elements of `dt`.

    `kind` is an `atomic_rmw` op (`add`, `max`, `min`, `xchg`, `and`, `or`, `xor`) or `cas`.
    Both execution modes call this check, so a kernel that runs in the interpreter doesn't
    fail on the GPU because of its dtype.
    """
    if dt.is_bool():
        raise CompilationError(
            f"tl.{fname} doesn't support int1 (boolean) pointers. Use an int32 buffer instead."
        )
    if dt.is_floating() and kind in _ATOMIC_BITWISE:
        raise CompilationError(
            f"tl.{fname} needs integer elements, but the pointer points to {dt}. Use an "
            "integer buffer, or bitcast the pointer's buffer on the host."
        )
    if kind == "add" and dt in (core.float16, core.bfloat16):
        raise CompilationError(
            f"tl.{fname} doesn't support {dt} addition, because 16-bit float atomics lose "
            "precision and have no hardware support. Accumulate in a float32 buffer instead, "
            "and convert the result afterward."
        )
    if dt.primitive_bitwidth == 64 and not (dt is core.uint64 and kind in ("max", "min")):
        raise CompilationError(
            f"tl.{fname} doesn't support {dt}. {_ATOMIC64_LIMITS} {_ATOMIC64_HINT[kind]}"
        )


def _atomic_frontend(ctx, kind: str, pointer, operands: list[Any], mask, sem, scope,
                     fname: str):  # fmt: skip
    b = ctx.b
    p = core.unwrap(pointer)
    if not semantic.is_pointer(p):
        raise CompilationError(
            f"tl.{fname} needs a pointer or pointer tile, but got {describe(p)}.{_PTR_HINT}"
        )
    _check_sem(sem, scope, fname)
    elem = ir.elem_of(p.type).elem.dtype
    check_atomic_dtype(kind, elem, fname)
    vals = []
    for v in operands:
        v = core.unwrap(v)
        if semantic.is_pointer(v):
            raise CompilationError(
                f"tl.{fname} can't take pointers as values. Pass integer offsets instead."
            )
        vals.append(semantic.to_value(b, v, elem))
    mask = core.unwrap(mask)
    if mask is True:
        mask = None
    if mask is not None:
        if not isinstance(mask, ir.Value):
            raise CompilationError(f"`mask` must be a boolean tile, but got {describe(mask)}")
        _check_mask_dtype(semantic.dtype_of(mask))
    shape = ir.shape_of(p.type)
    for v in [*vals, *([mask] if mask is not None else [])]:
        shape = semantic.broadcast_shapes(shape, ir.shape_of(v.type))
    ops = [semantic.broadcast_to(b, p, shape)] + [semantic.broadcast_to(b, v, shape) for v in vals]
    if mask is not None:
        ops.append(semantic.broadcast_to(b, mask, shape))
    rtype = ir.with_elem(ops[0].type, ir.scalar(elem))
    if kind == "cas":
        return b.create("atomic_cas", ops, [rtype]).result
    return b.create("atomic_rmw", ops, [rtype], {"op": kind}).result


def _apply_atomic(kind: str, old: np.ndarray, val: np.ndarray, cmp: np.ndarray | None,
                  dt: core.dtype) -> np.ndarray:  # fmt: skip
    """Returns the values that an atomic writes, given the old values."""
    if kind == "xchg":
        return val
    if kind == "cas":
        bits = np.dtype(f"u{dt.itemsize}")
        return np.where(old.view(bits) == cmp.view(bits), val, old)
    if kind in _ATOMIC_BITWISE:
        return {"and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor}[kind](old, val)
    if dt.is_floating():
        a, v = old.astype(np.float32), val.astype(np.float32)
        out = {"add": np.add, "max": np.fmax, "min": np.fmin}[kind](a, v)
        return out.astype(dt.to_numpy())
    return {"add": np.add, "max": np.maximum, "min": np.minimum}[kind](old, val)


def _atomic_interp(kind: str, pointer, operands: list[Any], mask, sem, scope, fname: str):
    if not isinstance(pointer, IPointer):
        raise CompilationError(
            f"tl.{fname} needs a pointer or pointer tile, but got {describe(pointer)}."
            f"{_PTR_HINT}"
        )
    _check_sem(sem, scope, fname)
    dt = pointer.elem
    check_atomic_dtype(kind, dt, fname)
    vals = [I.to_tile(core.unwrap(v), dt).data for v in operands]
    arrays = [pointer.offsets, *vals]
    mask = core.unwrap(mask)
    if mask is not None and mask is not True:
        m = _itile(mask)
        _check_mask_dtype(m.dtype)
        arrays.append(m.data)
    shape: tuple[int, ...] = ()
    for a in arrays:
        shape = semantic.broadcast_shapes(shape, a.shape)
    bc = [np.broadcast_to(a, shape) for a in arrays]
    offs = bc[0]
    mb = bc[1 + len(vals)] if len(bc) > 1 + len(vals) else np.ones(shape, bool)
    pointer.check_bounds(offs, mb, fname)
    old = np.zeros(shape, dt.to_numpy())
    o = offs[mb]
    v = bc[len(vals)][mb]  # the stored value is the last operand
    c = bc[1][mb] if kind == "cas" else None
    flat = pointer.flat
    if np.unique(o).size == o.size:
        cur = np.array(flat[o])
        old[mb] = cur
        flat[o] = _apply_atomic(kind, cur, v, c, dt)
    else:
        # Colliding addresses apply one at a time, in row-major order.
        got = np.empty(o.shape, dt.to_numpy())
        for i in range(o.size):
            cur = np.array(flat[o[i : i + 1]])
            got[i] = cur[0]
            ci = None if c is None else c[i : i + 1]
            flat[o[i : i + 1]] = _apply_atomic(kind, cur, v[i : i + 1], ci, dt)
        old[mb] = got
    return ITile(old, dt)


def _make_atomic(kind: str) -> core.Builtin:
    fname = f"atomic_{kind}"

    def interp(pointer, val, mask=None, sem=None, scope=None):
        return _atomic_interp(kind, pointer, [val], mask, sem, scope, fname)

    def frontend(ctx, pointer, val, mask=None, sem=None, scope=None):
        return _atomic_frontend(ctx, kind, pointer, [val], mask, sem, scope, fname)

    what = {"add": "adds `val` to", "max": "stores the maximum of `val` and",
            "min": "stores the minimum of `val` and", "xchg": "stores `val` in",
            "and": "stores the bitwise AND of `val` and",
            "or": "stores the bitwise OR of `val` and",
            "xor": "stores the bitwise XOR of `val` and"}[kind]  # fmt: skip
    notes = {
        "add": "`float16` and `bfloat16` addition isn't supported; accumulate in a `float32` "
               "buffer instead.",
        "max": "For floats, a NaN operand is ignored, as in `tl.maximum`. `uint64` needs an "
               "Apple9 GPU (M3 or later), and you can't use its result.",
        "min": "For floats, a NaN operand is ignored, as in `tl.minimum`. `uint64` needs an "
               "Apple9 GPU (M3 or later), and you can't use its result.",
        "xchg": "",
        "and": "The elements must be integers.",
        "or": "The elements must be integers.",
        "xor": "The elements must be integers.",
    }[kind]  # fmt: skip
    frontend.__doc__ = (
        f"Atomically {what} each element that `pointer` points to, where `mask` is true.\n\n"
        "`pointer` is a pointer or a tile of pointers, and `val` broadcasts to its shape. "
        "Returns the values that memory held before the operation, or 0 where `mask` is "
        "false. Memory ordering is relaxed: `sem` accepts only `None` and `\"relaxed\"`, "
        "and `scope` accepts `None`, `\"gpu\"`, and `\"cta\"`. Metal has no 64-bit "
        "atomics other than `uint64` `max` and `min`, so other 64-bit elements aren't "
        f"supported. {notes}"
    ).rstrip() + "\n"
    return builtin(interp=interp, name=fname)(frontend)


atomic_add = _make_atomic("add")
atomic_max = _make_atomic("max")
atomic_min = _make_atomic("min")
atomic_xchg = _make_atomic("xchg")
atomic_and = _make_atomic("and")
atomic_or = _make_atomic("or")
atomic_xor = _make_atomic("xor")


def _i_atomic_cas(pointer, cmp, val, sem=None, scope=None):
    return _atomic_interp("cas", pointer, [cmp, val], None, sem, scope, "atomic_cas")


@builtin(interp=_i_atomic_cas)
def atomic_cas(ctx, pointer, cmp, val, sem=None, scope=None):
    """Atomically stores `val` where memory equals `cmp`, and returns the old values.

    The comparison is bitwise, so for floats `-0.0` doesn't match `0.0`, and a NaN matches
    a NaN with the same bits. Memory ordering is relaxed.
    """
    return _atomic_frontend(ctx, "cas", pointer, [cmp, val], None, sem, scope, "atomic_cas")


# ---------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------


def _scan_axis(axis: Any, rank: int, fname: str) -> int:
    if core.unwrap(axis) is None:
        raise CompilationError(f"tl.{fname} needs an `axis`, for example `axis=0`")
    return _axis(axis, rank)


def _check_combine_fn(fn: Any, fname: str) -> None:
    from enceladus.compiler.frontend import is_jit_function

    if not is_jit_function(fn):
        raise CompilationError(
            f"tl.{fname} needs a @enceladus.jit function as `combine_fn`. Plain Python "
            "functions aren't accepted, because the kernel cache doesn't track their source."
        )


def _scan_frontend(ctx, inputs: list[Any], axis, reverse, fname: str, kind: str | None = None,
                   combine_fn=None) -> list[ir.Value]:  # fmt: skip
    b = ctx.b
    vals = [_tile_value(ctx, x, fname) for x in inputs]
    if any(semantic.is_pointer(v) for v in vals):
        raise CompilationError(f"tl.{fname} can't scan pointers. Scan integer offsets instead.")
    if kind == "sum":
        vals = [semantic.cast(b, vals[0], _sum_input_dtype(semantic.dtype_of(vals[0])))]
    shapes = {v.type.shape for v in vals}
    if len(shapes) != 1:
        raise CompilationError(f"tl.{fname} needs inputs of one shape, but got {sorted(shapes)}")
    ax = _scan_axis(axis, len(vals[0].type.shape), fname)
    rev = core.unwrap(reverse)
    if not isinstance(rev, bool):
        raise CompilationError(f"`reverse` must be a compile-time bool, but got {describe(rev)}")
    elems = [v.type.elem for v in vals]
    attrs: dict[str, Any] = {"axis": ax, "reverse": rev}
    regions = []
    if kind is not None:
        attrs["kind"] = kind
    else:
        names = [f"a{i}" for i in range(len(elems))] + [f"b{i}" for i in range(len(elems))]
        block = ir.Block(elems + elems, names)
        with b.at(block):
            res = ctx.call_function(combine_fn, list(block.args), {})
            res = res if isinstance(res, tuple) else (res,)
            if len(res) != len(vals):
                raise CompilationError(
                    f"combine_fn returns {len(res)} values for {len(vals)} inputs. Return one "
                    "value per input."
                )
            pairs = enumerate(zip(res, elems, strict=True))
            b.create("yield", [ctx.materialize(r, e, f"result {i} of combine_fn")
                               for i, (r, e) in pairs])  # fmt: skip
        regions = [ir.Region(block)]
    op = b.create("scan", vals, [v.type for v in vals], attrs, regions=regions)
    return list(op.results)


def _hillis_steele(xs: list[ITile], ax: int, combine) -> list[ITile]:
    """Returns the inclusive scan of `xs` along `ax`, calling `combine(earlier, later)`."""
    n = xs[0].shape[ax]
    s = 1
    while s < n:
        lo, hi, head = ([slice(None)] * len(xs[0].shape) for _ in range(3))
        lo[ax], hi[ax], head[ax] = slice(0, n - s), slice(s, n), slice(0, s)
        res = combine(*[ITile(x.data[tuple(lo)], x.dtype) for x in xs],
                      *[ITile(x.data[tuple(hi)], x.dtype) for x in xs])  # fmt: skip
        res = res if isinstance(res, tuple) else (res,)
        if len(res) != len(xs):
            raise CompilationError(
                f"combine_fn returns {len(res)} values for {len(xs)} inputs. Return one value "
                "per input."
            )
        xs = [ITile(np.concatenate([x.data[tuple(head)], I.to_tile(r, x.dtype).data], axis=ax),
                    x.dtype) for x, r in zip(xs, res, strict=True)]  # fmt: skip
        s *= 2
    return xs


def _i_cumsum(input, axis=0, reverse=False, dtype=None):
    t = _itile(input)
    if dtype is not None:
        t = I.convert(t, _dtype(dtype))
    t = I.convert(t, _sum_input_dtype(t.dtype))
    ax = _scan_axis(axis, len(t.shape), "cumsum")
    data = I.as_compute(t, t.dtype)
    rev = bool(core.unwrap(reverse))
    if rev:
        data = np.flip(data, ax)
    out = np.cumsum(data, axis=ax, dtype=data.dtype)
    if rev:
        out = np.flip(out, ax)
    return I.wrap(out, t.dtype)


@builtin(interp=_i_cumsum)
def cumsum(ctx, input, axis=0, reverse=False, dtype=None):
    """Returns the inclusive cumulative sum along `axis`.

    With `reverse=True`, each element is the sum of itself and every later element.
    Integers narrower than 32 bits, including `int1`, sum in 32 bits. `float16` and
    `bfloat16` accumulate in `float32` and round each result back.
    """
    v = _tile_value(ctx, input, "cumsum")
    if dtype is not None:
        v = semantic.cast(ctx.b, v, _dtype(dtype))
    return _scan_frontend(ctx, [v], axis, reverse, "cumsum", kind="sum")[0]


def _i_associative_scan(input, axis, combine_fn, reverse=False):
    _check_combine_fn(combine_fn, "associative_scan")
    single = not isinstance(input, tuple)
    xs = [_itile(x) for x in ((input,) if single else input)]
    shapes = {x.shape for x in xs}
    if len(shapes) != 1:
        raise CompilationError(
            f"tl.associative_scan needs inputs of one shape, but got {sorted(shapes)}"
        )
    ax = _scan_axis(axis, len(xs[0].shape), "associative_scan")
    rev = bool(core.unwrap(reverse))
    if rev:
        xs = [ITile(np.flip(x.data, ax), x.dtype) for x in xs]
    xs = _hillis_steele(xs, ax, combine_fn)
    if rev:
        xs = [ITile(np.flip(x.data, ax), x.dtype) for x in xs]
    return xs[0] if single else tuple(xs)


@builtin(interp=_i_associative_scan)
def associative_scan(ctx, input, axis, combine_fn, reverse=False):
    """Returns the inclusive scan of `input` along `axis` with `combine_fn`.

    `input` can be a tile or a tuple of tiles of the same shape. `combine_fn` is a
    @enceladus.jit function that takes `(a0, ..., b0, ...)`, where the `a` values come
    from earlier elements, and returns the combined values. It must be associative, but
    it doesn't need to be commutative. With `reverse=True`, the scan runs from the last
    element to the first, and the `a` values come from later elements.
    """
    fn = core.unwrap(combine_fn)
    _check_combine_fn(fn, "associative_scan")
    single = not isinstance(core.unwrap(input), tuple)
    inputs = [input] if single else list(core.unwrap(input))
    outs = _scan_frontend(ctx, inputs, axis, reverse, "associative_scan", combine_fn=fn)
    return outs[0] if single else tuple(outs)


_method(cumsum)
_method(associative_scan)
