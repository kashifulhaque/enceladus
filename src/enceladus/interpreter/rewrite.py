"""Source rewrites that make interpreted kernels follow compiled semantics.

The interpreter runs a kernel's Python function, but a few Python constructs mean
something different in a kernel. Before the first launch, `interpretable` recompiles the
kernel from its source with these rewrites:

- `for i in range(...)` and `for i in tl.range(...)` make `i` a typed `int32` (or
  `int64`) scalar instead of a Python int, so `i.to(...)` works and arithmetic on `i`
  wraps as on the GPU. Numbers assigned before the loop and reassigned in its body
  become typed scalars too, as they do in compiled code.
- Chained comparisons such as `0 <= i < n` keep Python's short-circuit semantics for
  scalars, and a chain that compares tiles raises a `CompilationError`, as in compiled
  code. Without the rewrite, Python would return the last comparison of a chain such as
  `0 < n < x` unchecked.
- Calls to other `@enceladus.jit` functions run the rewritten helper.

A function whose source can't be read or rewritten runs unchanged.
"""

from __future__ import annotations

import ast
import builtins
import copy
import types
import weakref
from typing import Any

from enceladus.compiler.errors import CompilationError
from enceladus.interpreter import interp
from enceladus.language import core

_CACHE: weakref.WeakKeyDictionary[types.FunctionType, types.FunctionType | None] = (
    weakref.WeakKeyDictionary()
)
"""Rewritten functions by original function; `None` means the function runs unchanged."""

_RANGE = "__enceladus_range__"
_RUNTIME = "__enceladus_runtime_loop__"
_CARRY = "__enceladus_carry__"
_CHAIN = "__enceladus_chain__"
_JIT = "__enceladus_jit__"

_CMP_NAMES = {
    ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne",
    ast.Is: "is", ast.IsNot: "is_not", ast.In: "in", ast.NotIn: "not_in",
}  # fmt: skip


def _is_runtime_range(func: Any) -> bool:
    return func is builtins.range or func is core.BUILTINS["range"]


def _loop_range(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Returns the iterable of a `for` loop whose iterable is the call `func(*args)`."""
    if func is builtins.range and not kwargs:
        return interp.typed_range(*args)
    return func(*args, **kwargs)  # `tl.range` yields typed scalars itself.


def _carry(value: Any) -> Any:
    """Converts a number that a runtime loop reassigns to a typed scalar."""
    if type(value) in (bool, int, float):
        return interp.to_tile(value)
    return value


def _jit_callee(fn: Any) -> Any:
    """Returns the function to call for `fn(...)` when `fn` is a `@enceladus.jit` function."""
    inner = getattr(fn, "fn", None)
    if getattr(fn, "_is_enceladus_jit", False) is True and isinstance(inner, types.FunctionType):
        return interpretable(inner)
    return fn


_HELPERS = {_RANGE: _loop_range, _RUNTIME: _is_runtime_range, _CARRY: _carry,
            _CHAIN: interp.chain_compare, _JIT: _jit_callee}  # fmt: skip


def interpretable(fn: types.FunctionType) -> types.FunctionType:
    """Returns `fn` recompiled with the interpreter's rewrites, or `fn` if that fails."""
    try:
        out = _CACHE[fn]
    except KeyError:
        try:
            out = _rewrite(fn)
        except (CompilationError, OSError, SyntaxError, TypeError, ValueError):
            out = None  # For example, the source isn't available.
        _CACHE[fn] = out
    return fn if out is None else out


class _Rewriter(ast.NodeTransformer):
    def __init__(self, jit_names: set[str]) -> None:
        self.jit_names = jit_names
        self.changed = False

    def _name(self, name: str, like: ast.AST) -> ast.Name:
        return ast.copy_location(ast.Name(name, ast.Load()), like)

    def visit_For(self, node: ast.For) -> Any:
        from enceladus.compiler.frontend import assigned_names

        self.generic_visit(node)
        it = node.iter
        if not isinstance(it, ast.Call) or any(isinstance(a, ast.Starred) for a in it.args):
            return node
        self.changed = True
        node.iter = ast.copy_location(
            ast.Call(self._name(_RANGE, it), [it.func, *it.args], it.keywords), it
        )
        target = node.target.id if isinstance(node.target, ast.Name) else None
        names = [n for n in assigned_names(node.body) if n != target]
        if not names:
            return node
        # if __enceladus_runtime_loop__(<func>):
        #     try: name = __enceladus_carry__(name)
        #     except NameError: pass
        body = []
        for n in names:
            call = ast.Call(self._name(_CARRY, node), [ast.Name(n, ast.Load())], [])
            body.append(ast.Try(
                body=[ast.Assign([ast.Name(n, ast.Store())], call)],
                handlers=[ast.ExceptHandler(ast.Name("NameError", ast.Load()), None,
                                            [ast.Pass()])],
                orelse=[], finalbody=[],
            ))  # fmt: skip
        test = ast.Call(self._name(_RUNTIME, node), [copy.deepcopy(it.func)], [])
        guard = ast.If(test, body, [])
        for n in ast.walk(guard):
            ast.copy_location(n, node)
        return [guard, node]

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) < 2:
            return node
        self.changed = True
        args: list[ast.expr] = [node.left, ast.Constant(_CMP_NAMES[type(node.ops[0])]),
                                node.comparators[0]]  # fmt: skip
        empty = ast.arguments([], [], None, [], [], None, [])
        for op, comp in zip(node.ops[1:], node.comparators[1:], strict=True):
            args.append(ast.copy_location(ast.Constant(_CMP_NAMES[type(op)]), comp))
            args.append(ast.copy_location(ast.Lambda(empty, comp), comp))
        return ast.copy_location(ast.Call(self._name(_CHAIN, node), args, []), node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.jit_names:
            self.changed = True
            node.func = ast.copy_location(
                ast.Call(self._name(_JIT, node.func), [node.func], []), node.func
            )
        return node


def _jit_names(fn: types.FunctionType) -> set[str]:
    """Returns the global and closure names of `fn` that hold `@enceladus.jit` functions."""
    from enceladus.compiler.frontend import is_jit_function

    names = {n for n in fn.__code__.co_names if is_jit_function(fn.__globals__.get(n))}
    for n, cell in zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=True):
        try:
            if is_jit_function(cell.cell_contents):
                names.add(n)
        except ValueError:  # an empty cell
            pass
    return names


def _rewrite(fn: types.FunctionType) -> types.FunctionType | None:
    from enceladus.compiler.frontend import parse_function

    src = parse_function(fn)
    fdef = src.tree
    for node in ast.walk(fdef):
        if hasattr(node, "col_offset"):
            node.col_offset += src.indent
            if getattr(node, "end_col_offset", None) is not None:
                node.end_col_offset += src.indent
    ast.increment_lineno(fdef, src.first_line - 1)
    fdef.decorator_list = []
    rw = _Rewriter(_jit_names(fn))
    fdef = rw.visit(fdef)
    if not rw.changed:
        return None
    # Compile the kernel inside a factory whose parameters become the kernel's free
    # variables, so closure variables stay closure variables and helpers need no globals.
    code = fn.__code__
    params = [*code.co_freevars, *_HELPERS]
    factory = ast.FunctionDef(
        name="__enceladus_factory__",
        args=ast.arguments([], [ast.arg(p) for p in params], None, [], [], None, []),
        body=[fdef, ast.Return(ast.Name(fdef.name, ast.Load()))],
        decorator_list=[], returns=None, type_params=[],
    )  # fmt: skip
    ast.copy_location(factory, fdef)
    module = ast.fix_missing_locations(ast.Module([factory], []))
    top = compile(module, code.co_filename, "exec", dont_inherit=True)
    fcode = next(c for c in top.co_consts if isinstance(c, types.CodeType))
    new = next(c for c in fcode.co_consts
               if isinstance(c, types.CodeType) and c.co_name == fdef.name)  # fmt: skip
    cells = dict(zip(code.co_freevars, fn.__closure__ or (), strict=True))
    closure = []
    for n in new.co_freevars:
        if n in cells:
            closure.append(cells[n])
        elif n in _HELPERS:
            closure.append(types.CellType(_HELPERS[n]))
        else:
            return None
    out = types.FunctionType(new, fn.__globals__, fn.__name__, fn.__defaults__,
                             tuple(closure) or None)  # fmt: skip
    out.__kwdefaults__ = fn.__kwdefaults__
    out.__qualname__ = fn.__qualname__
    for c in _code_objects(new):
        interp.register_kernel_code(c)
    return out


def _code_objects(code: types.CodeType):
    yield code
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            yield from _code_objects(c)
