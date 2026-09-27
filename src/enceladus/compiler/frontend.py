"""The frontend: Python AST to Enceladus IR.

`CodeGenerator` walks a kernel's AST, in the style of Triton's `CodeGenerator`:

- Values derived only from constexprs and Python literals stay Python values, and the
  frontend folds them in Python. Runtime values are `ir.Value` objects.
- `if` on a compile-time condition emits only the taken branch. `if` on a runtime scalar
  emits an `if` op whose results are the variables that the branches assign. `if` on a
  tile is an error that suggests `tl.where`.
- `for ... in range(...)` emits a `for` op. Variables that the body reassigns and that
  exist before the loop become iteration arguments. `tl.static_range` unrolls.
- Calls to other `@enceladus.jit` functions are inlined with their own scope.
- Unsupported constructs raise `CompilationError` with the source location.

The entry point is `build_ir`.
"""

from __future__ import annotations

import ast
import builtins
import functools
import inspect
import operator
import textwrap
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from enceladus.compiler import ir, semantic
from enceladus.compiler.errors import CompilationError, Loc, internal_error
from enceladus.language import core

_MISSING = object()


@dataclass(frozen=True)
class SourceInfo:
    """The parsed source of a Python function, with the offsets that map AST positions back
    to the file."""

    tree: ast.FunctionDef
    file: str
    first_line: int
    indent: int

    def loc(self, node: ast.AST) -> Loc:
        return Loc(self.file, self.first_line + node.lineno - 1, node.col_offset + self.indent + 1)


def parse_function(fn: types.FunctionType) -> SourceInfo:
    """Gets, dedents, and parses the source of `fn`."""
    try:
        lines, first_line = inspect.getsourcelines(fn)
    except (OSError, TypeError) as e:
        raise CompilationError(
            f"can't read the source of `{fn.__name__}`. Define kernels in a file, not in an "
            "interactive prompt.",
            Loc(fn.__code__.co_filename, fn.__code__.co_firstlineno, 1),
        ) from e
    indent = len(lines[0]) - len(lines[0].lstrip())
    tree = ast.parse(textwrap.dedent("".join(lines)))
    fdef = tree.body[0]
    if not isinstance(fdef, ast.FunctionDef):
        raise CompilationError(
            f"`{fn.__name__}` must be a plain `def` function. Decorate a module-level `def` "
            "with @enceladus.jit.",
            Loc(fn.__code__.co_filename, first_line, 1),
        )
    file = inspect.getsourcefile(fn) or fn.__code__.co_filename
    return SourceInfo(fdef, file, first_line, indent)


def is_jit_function(obj: Any) -> bool:
    return getattr(obj, "_is_enceladus_jit", False) is True


def assigned_names(stmts: Sequence[ast.stmt]) -> list[str]:
    """Returns the names that `stmts` assign, in source order, without duplicates."""
    out: dict[str, None] = {}

    class V(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                out[node.id] = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            pass

        visit_Lambda = visit_FunctionDef

    for s in stmts:
        V().visit(s)
    return list(out)


class _BoundMethod:
    """A tile or descriptor method bound to its receiver, such as `x.to`."""

    def __init__(self, fn: core.Builtin, receiver: ir.Value) -> None:
        self.fn = fn
        self.receiver = receiver


_BINOPS = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.FloorDiv: "floordiv",
    ast.Mod: "mod", ast.BitAnd: "and", ast.BitOr: "or", ast.BitXor: "xor", ast.LShift: "shl",
    ast.RShift: "shr",
}  # fmt: skip
_PY_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
    ast.LShift: operator.lshift, ast.RShift: operator.rshift, ast.Pow: operator.pow,
    ast.MatMult: operator.matmul,
}  # fmt: skip
_CMPOPS = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}
_PY_CMPOPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}  # fmt: skip

# Scalar ops that are safe to run when their result isn't needed, so the right side of
# `and` or `or` made only of them becomes a `select` instead of an `if`.
_SPECULATABLE = frozenset(
    ["const", "binary", "cmp", "unary", "cast", "bitcast", "select", "fma", "program_id",
     "num_programs"]
)  # fmt: skip

_FORBIDDEN = {
    "While": "`while` loops aren't supported. Use `for i in range(...)` with a bound.",
    "Break": "`break` isn't supported. Use a mask, or restructure the loop bounds.",
    "Continue": "`continue` isn't supported. Guard the rest of the loop body with an `if`.",
    "Try": "`try` isn't supported in kernels. Check for errors on the host before the launch.",
    "TryStar": "`try` isn't supported in kernels. Check for errors on the host first.",
    "With": "`with` isn't supported in kernels. Use the context manager around the launch.",
    "AsyncWith": "`with` isn't supported in kernels. Use it around the launch instead.",
    "AsyncFor": "`async for` isn't supported in kernels. Use `for i in range(...)`.",
    "Global": "`global` isn't supported in kernels. Pass the value as a kernel argument.",
    "Nonlocal": "`nonlocal` isn't supported in kernels. Pass the value as an argument.",
    "Delete": "`del` isn't supported in kernels. Remove the statement.",
    "Import": "imports aren't supported inside kernels. Import at module level.",
    "ImportFrom": "imports aren't supported inside kernels. Import at module level.",
    "ClassDef": "class definitions aren't supported inside kernels. Define them at module level.",
    "FunctionDef": (
        "nested functions (closures) aren't supported. Define the helper at module level and "
        "decorate it with @enceladus.jit."
    ),
    "AsyncFunctionDef": (
        "nested functions aren't supported. Define the helper at module level and decorate it "
        "with @enceladus.jit."
    ),
    "Lambda": "lambdas aren't supported. Define a module-level @enceladus.jit function instead.",
    "Raise": "`raise` isn't supported. Use tl.static_assert for compile-time checks.",
    "ListComp": "comprehensions aren't supported. Use tl.static_range to unroll a loop.",
    "SetComp": "comprehensions aren't supported. Use tl.static_range to unroll a loop.",
    "DictComp": "comprehensions aren't supported. Use tl.static_range to unroll a loop.",
    "GeneratorExp": "generator expressions aren't supported. Use tl.static_range instead.",
    "Yield": "`yield` isn't supported in kernels. Write results with tl.store.",
    "YieldFrom": "`yield` isn't supported in kernels. Write results with tl.store.",
    "Await": "`await` isn't supported in kernels. Await on the host, around the launch.",
    "NamedExpr": "the `:=` operator isn't supported in kernels. Assign on a separate line.",
    "Match": "`match` isn't supported in kernels. Use `if` and `elif` instead.",
    "Starred": "star expressions are supported only for compile-time tuples in calls.",
    "Dict": (
        "dict literals aren't supported in kernels. Use a compile-time tuple, or pass each "
        "value as its own tl.constexpr parameter."
    ),
    "Set": "set literals aren't supported in kernels. Use a compile-time tuple instead.",
    "TypeAlias": "`type` statements aren't supported in kernels. Define the alias at module level.",
    "TemplateStr": "template strings (t-strings) aren't supported in kernels. Use an f-string.",
}

_ALLOWED_GLOBAL_TYPES = (
    int, float, bool, str, type(None), tuple, core.dtype, core.Builtin, types.ModuleType,
    types.FunctionType, types.BuiltinFunctionType, type,
)  # fmt: skip


def _check_tracked(v: Any, what: str) -> None:
    """Refuses a global value whose changes the kernel cache can't detect.

    Raises:
        CompilationError: The dependency hash can't track `v`.
    """
    from enceladus.runtime.jit import is_tracked

    if is_tracked(v):
        return
    fix = ("Use a plain function, a @enceladus.jit function, or a functools.partial instead."
           if callable(v) else
           "Use a module, a class, or a constant value such as a tuple instead, or pass the "
           "value as a tl.constexpr argument.")  # fmt: skip
    raise CompilationError(
        f"{what} is a {type(v).__name__} object, and the kernel cache can't detect a "
        f"change to it. {fix}"
    )


class CodeGenerator(ast.NodeVisitor):
    """Emits IR for a kernel and the functions it calls.

    Builtin frontend handlers receive this object as `ctx`. They use `ctx.b` (the IR
    builder), `ctx.call_function(fn, args, kwargs)` to inline a `@enceladus.jit` function, and
    `ctx.is_known_one(value)`.
    """

    def __init__(self, fn: Any, builder: ir.Builder, known_one: set[int]) -> None:
        self.b = builder
        self.fn = fn
        self.src: SourceInfo = fn.source_info()
        self.globals: dict[str, Any] = fn.fn.__globals__
        self.scope: dict[str, Any] = {}
        self.known_one = known_one
        self.call_stack: list[Any] = [fn]
        self.runtime_depth = 0
        self.returned = False
        self.ret_value: Any = None
        self.in_kernel = True
        self.scoped_out: dict[str, str] = {}
        self.truth_context = False
        self.debug = False  # keep tl.device_assert checks (ENCELADUS_DEBUG)
        # Python callables that `_check_helper` accepted, by id, holding a reference so
        # that the ids stay unique.
        self._tracked_helpers: dict[int, Any] = {}

    # ---- infrastructure ----

    def visit(self, node: ast.AST) -> Any:
        saved = self.b.loc
        if hasattr(node, "lineno"):
            self.b.loc = self.src.loc(node)
        try:
            return super().visit(node)
        except CompilationError as e:
            raise e.with_loc(self.b.loc) from None
        except (TypeError, ValueError, ArithmeticError, IndexError, KeyError, AttributeError) as e:
            raise CompilationError(
                f"evaluating this line at compile time raised {type(e).__name__}: {e}. Check "
                "the compile-time values that it uses, such as tl.constexpr parameters.",
                self.b.loc,
            ) from e
        finally:
            self.b.loc = saved

    def generic_visit(self, node: ast.AST) -> Any:
        name = type(node).__name__
        msg = _FORBIDDEN.get(name)
        if msg is None:
            try:
                what = f"`{ast.unparse(node)}`"
            except Exception:  # noqa: BLE001 - unparse can fail on nodes of new Python versions.
                what = "this syntax"
            if len(what) > 40:
                what = "this syntax"
            msg = (
                f"kernels don't support {what} (a Python `{name}` node). Rewrite it with "
                "assignments, arithmetic, `if`, `for`, and calls to tl functions or "
                "@enceladus.jit functions."
            )
        raise CompilationError(msg)

    def visit_body(self, stmts: Sequence[ast.stmt]) -> None:
        for s in stmts:
            if self.returned:
                return
            self.visit(s)

    def is_known_one(self, v: Any) -> bool:
        """Returns whether `v` is 1 at compile time or by specialization."""
        v = core.unwrap(v)
        if not isinstance(v, ir.Value):
            return v == 1 and not isinstance(v, bool)
        op = v.defining_op
        if op is not None and op.name == "const":
            return op.attrs["value"] == 1
        return id(v) in self.known_one

    # ---- kernel and function bodies ----

    def run_kernel(self, scope: dict[str, Any]) -> None:
        self.scope = scope
        self.visit_body(self.src.tree.body)
        self.b.loc = Loc(self.src.file, self.src.first_line + _last_line(self.src.tree) - 1, 1)
        self.b.create("return")

    def call_function(self, fn: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
        """Inlines a call to a `@enceladus.jit` function and returns its result."""
        if any(f is fn for f in self.call_stack):
            chain = " -> ".join(f.__name__ for f in [*self.call_stack, fn])
            raise CompilationError(
                f"recursion isn't supported: {chain}. Rewrite the recursion as a loop."
            )
        try:
            bound = fn.signature.bind(*args, **kwargs)
        except TypeError as e:
            raise CompilationError(
                f"the arguments don't match the parameters of `{fn.__name__}`: {e}"
            ) from None
        bound.apply_defaults()
        scope: dict[str, Any] = {}
        for p in fn.params:
            v = core.unwrap(bound.arguments[p.name])
            if p.is_constexpr and isinstance(v, ir.Value):
                raise CompilationError(
                    f"parameter `{p.name}` of `{fn.__name__}` is tl.constexpr, but the call "
                    f"passes {semantic.describe(v)}. Pass a compile-time value, such as a "
                    "literal or a tl.constexpr parameter of the caller."
                )
            scope[p.name] = v
        saved = (self.fn, self.src, self.globals, self.scope, self.runtime_depth,
                 self.returned, self.ret_value, self.in_kernel, self.scoped_out)  # fmt: skip
        self.fn, self.src, self.globals = fn, fn.source_info(), fn.fn.__globals__
        self.scope, self.runtime_depth, self.returned, self.ret_value = scope, 0, False, None
        self.in_kernel, self.scoped_out = False, {}
        self.call_stack.append(fn)
        try:
            self.visit_body(self.src.tree.body)
            return self.ret_value
        finally:
            self.call_stack.pop()
            (self.fn, self.src, self.globals, self.scope, self.runtime_depth,
             self.returned, self.ret_value, self.in_kernel, self.scoped_out) = saved  # fmt: skip

    # ---- statements ----

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return  # Docstring.
        self.visit(node.value)

    def visit_Pass(self, node: ast.Pass) -> None:
        pass

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self.visit(node.value)
        for t in node.targets:
            self.assign(t, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.assign(node.target, self.visit(node.value))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise CompilationError(
                "augmented assignment needs a plain variable name as its target, as in "
                "`acc += x`. Tiles are immutable; build a new tile instead."
            )
        cur = self.lookup(node.target.id)
        self.assign(node.target, self.binop(node.op, cur, self.visit(node.value)))

    def assign(self, target: ast.expr, value: Any) -> None:
        if isinstance(target, ast.Name):
            _check_no_tile_list(value)
            if isinstance(value, ir.Value) and value.name_hint is None:
                if value.defining_op is not None:
                    value.name_hint = target.id
            self.scope[target.id] = value
            self.scoped_out.pop(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (tuple, list)) or len(value) != len(target.elts):
                k = len(target.elts)
                if isinstance(value, (tuple, list)):
                    what = f"{len(value)} values"
                else:
                    what = f"{semantic.describe(value)}, which isn't a compile-time tuple,"
                raise CompilationError(
                    f"can't unpack {what} into {k} names. Unpack only compile-time tuples "
                    "with the same number of elements."
                )
            for t, v in zip(target.elts, value, strict=True):
                self.assign(t, v)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            raise CompilationError(
                "tiles are immutable, so you can't assign to an element or attribute. Build a "
                "new tile instead, for example with tl.where."
            )
        else:
            raise CompilationError(
                f"assignment to a `{type(target).__name__}` target isn't supported in kernels. "
                "Assign to a plain variable name."
            )

    def visit_Return(self, node: ast.Return) -> None:
        if self.runtime_depth > 0:
            raise CompilationError(
                "`return` inside a runtime `if` or `for` isn't supported. Assign the result to "
                "a variable and return it at the end of the function."
            )
        value = self.visit(node.value) if node.value is not None else None
        if self.in_kernel and value is not None:
            raise CompilationError(
                "kernels can't return values. Write results to memory with tl.store."
            )
        self.ret_value = value
        self.returned = True

    def visit_Assert(self, node: ast.Assert) -> None:
        cond = self.visit(node.test)
        if isinstance(cond, ir.Value):
            raise CompilationError(
                "`assert` on a runtime value isn't supported. Use tl.static_assert for "
                "compile-time checks."
            )
        if not cond:
            msg = self.visit(node.msg) if node.msg is not None else ""
            raise CompilationError(f"assertion failed: {msg}" if msg else "assertion failed")

    def visit_If(self, node: ast.If) -> None:
        cond = self.visit_truth(node.test)
        if not isinstance(cond, ir.Value):
            self.visit_body(node.body if cond else node.orelse)
            return
        c = self._runtime_cond(cond, "`if`")
        pre = self.scope
        then_blk, else_blk = ir.Block(), ir.Block()
        self.runtime_depth += 1
        try:
            self.scope = dict(pre)
            with self.b.at(then_blk):
                self.visit_body(node.body)
            then_scope = self.scope
            self.scope = dict(pre)
            with self.b.at(else_blk):
                self.visit_body(node.orelse)
            else_scope = self.scope
        finally:
            self.runtime_depth -= 1
        self.scope = dict(pre)
        merged: dict[str, Any] = {}
        pending: list[tuple[str, Any, Any]] = []
        for n in dict.fromkeys([*then_scope, *else_scope]):
            tv, ev = then_scope.get(n, _MISSING), else_scope.get(n, _MISSING)
            if tv is _MISSING or ev is _MISSING:
                self.scoped_out[n] = (
                    f"`{n}` is assigned in only one branch of the runtime `if` at line "
                    f"{self.b.loc.line}, so it's undefined after the `if`. Assign it before the "
                    "`if` or in both branches."
                )
                continue
            if _same_value(tv, ev):
                merged[n] = tv
            else:
                pending.append((n, tv, ev))
        results = self._emit_if(c, then_blk, else_blk, pending)
        for (n, _, _), r in zip(pending, results, strict=True):
            r.name_hint = n
            merged[n] = r
        self.scope = merged

    def _emit_if(self, c: ir.Value, then_blk: ir.Block, else_blk: ir.Block,
                 pending: list[tuple[str, Any, Any]]) -> list[ir.Value]:  # fmt: skip
        tys = [self._unify_type(n, tv, ev) for n, tv, ev in pending]
        with self.b.at(then_blk):
            vals = [self.materialize(tv, t, n) for (n, tv, _), t in zip(pending, tys, strict=True)]
            self.b.create("yield", vals)
        with self.b.at(else_blk):
            vals = [self.materialize(ev, t, n) for (n, _, ev), t in zip(pending, tys, strict=True)]
            self.b.create("yield", vals)
        op = self.b.create("if", [c], tys, regions=[ir.Region(then_blk), ir.Region(else_blk)])
        return op.results

    def _runtime_cond(self, cond: ir.Value, what: str) -> ir.Value:
        if isinstance(cond.type, ir.TileType):
            raise CompilationError(
                f"{what} on a tile isn't supported, because a tile of shape {cond.type.shape} "
                "has no single truth value. To choose values elementwise, use "
                "tl.where(cond, x, y)."
            )
        if isinstance(cond.type, ir.ScalarType):
            return semantic.to_bool(self.b, cond)
        raise CompilationError(
            f"{what} needs a scalar condition, but got {semantic.describe(cond)}"
        )

    def _unify_type(self, name: str, a: Any, b: Any) -> ir.Type:
        if isinstance(a, ir.Value) and isinstance(b, ir.Value):
            if a.type != b.type:
                raise CompilationError(
                    f"`{name}` has type {semantic.type_str(a.type)} in one branch and "
                    f"{semantic.type_str(b.type)} in the other. Convert one of them with "
                    "`.to(...)` so that both branches agree."
                )
            return a.type
        for x in (a, b):
            if not isinstance(x, ir.Value) and not semantic.is_literal(x):
                raise CompilationError(
                    f"`{name}` holds different compile-time values ({a!r} and {b!r}) depending on "
                    "a runtime condition. Only numbers can differ between branches."
                )
        if isinstance(a, ir.Value) or isinstance(b, ir.Value):
            return a.type if isinstance(a, ir.Value) else b.type
        return ir.scalar(semantic.computation_dtype("select", a, b))

    def materialize(self, v: Any, t: ir.Type, name: str = "value") -> ir.Value:
        """Returns `v` as a value of type `t`, emitting constants for Python literals."""
        v = core.unwrap(v)
        if isinstance(v, ir.Value):
            if v.type != t:
                raise CompilationError(
                    f"`{name}` has type {semantic.type_str(v.type)}, but "
                    f"{semantic.type_str(t)} is expected here. Convert it with `.to(...)`, "
                    "or broadcast it to the expected shape."
                )
            return v
        e = ir.elem_of(t)
        if not semantic.is_literal(v) or not isinstance(e, ir.ScalarType):
            raise CompilationError(
                f"can't use {semantic.describe(v)} as `{name}`, which has type "
                f"{semantic.type_str(t)}. Use a number or a value of that type."
            )
        if isinstance(v, float) and not e.dtype.is_floating():
            raise CompilationError(
                f"can't use the float {v!r} as `{name}`, which has type "
                f"{semantic.type_str(t)}. Use an integer, or make `{name}` a float from the "
                "start."
            )
        if not semantic.literal_fits(v, e.dtype):
            # Narrowing would change the value: 300 becomes 44 in uint8.
            wide = semantic.computation_dtype("select", e.dtype, v)
            if not semantic.literal_fits(v, wide):
                wide = core.int64 if v < 0 else semantic.literal_dtype(v)
            raise CompilationError(
                f"`{name}` is {v} on one path and a {semantic.type_str(t)} on another, but {v} "
                f"doesn't fit in {e.dtype!r}. Convert the {e.dtype!r} value with "
                f"`.to({wide!r})` so that both paths have a type that holds {v}."
            )
        if isinstance(t, ir.TileType):
            val = semantic.coerce_literal(v, e.dtype)
            return self.b.create("full", [], [t], {"value": val}).result
        return semantic.const(self.b, v, e.dtype)

    def visit_For(self, node: ast.For) -> None:
        if node.orelse:
            raise CompilationError(
                "`for ... else` isn't supported. Move the `else` body after the loop."
            )
        it = node.iter
        func = self.visit(it.func) if isinstance(it, ast.Call) else None
        static_range = core.BUILTINS["static_range"]
        if not any(func is f for f in (static_range, builtins.range, core.BUILTINS["range"])):
            raise CompilationError(
                "`for` loops must iterate over range(...), tl.range(...), or "
                "tl.static_range(...). To process the elements of a tile, use elementwise tl "
                "operations on the whole tile."
            )
        args = [core.unwrap(self.visit(a)) for a in it.args]
        kwargs = {k.arg: core.unwrap(self.visit(k.value)) for k in it.keywords}
        if func is static_range:
            if any(isinstance(a, ir.Value) for a in args) or kwargs:
                raise CompilationError(
                    "tl.static_range needs compile-time integer bounds. Use literals or "
                    "tl.constexpr parameters, or use range(...) for a runtime loop."
                )
            for v in range(*args):
                self.assign(node.target, v)
                self.visit_body(node.body)
                if self.returned:
                    return
            return
        if func is builtins.range and kwargs:
            raise CompilationError(
                "range() takes no keyword arguments. Pass the bounds positionally, as in "
                "`range(start, stop, step)`."
            )
        self._runtime_for(node, args)

    def _runtime_for(self, node: ast.For, args: list[Any]) -> None:
        if not 1 <= len(args) <= 3:
            raise CompilationError(f"range() takes 1 to 3 arguments, but got {len(args)}")
        lb, ub, step = (0, args[0], 1) if len(args) == 1 else (*args, 1)[:3]
        for v in (lb, ub, step):
            ok = (isinstance(v, int) and not isinstance(v, bool)) or (
                isinstance(v, ir.Value)
                and isinstance(v.type, ir.ScalarType)
                and v.type.dtype.is_int()
                and not v.type.dtype.is_bool()
            )
            if not ok:
                raise CompilationError(
                    f"range() bounds must be integer scalars, not {semantic.describe(v)}. "
                    "Convert a float bound with `.to(tl.int32)`."
                )
        if step == 0:
            raise CompilationError("the step of range() can't be zero. Use a nonzero step.")
        if not isinstance(node.target, ast.Name):
            raise CompilationError(
                "the loop variable must be a single name, as in `for i in range(n)`"
            )
        target = node.target.id
        iv_dt = _counter_dtype((lb, ub, step))
        iv_t = ir.scalar(iv_dt)
        lbv, ubv, stv = (semantic.to_value(self.b, v, iv_dt) for v in (lb, ub, step))

        pre = self.scope
        # As in Python, the loop variable keeps its last value after the loop, and an empty
        # loop leaves it unchanged. So when it shadows a variable of the counter's type, the
        # loop carries it like any other variable.
        shadowed = target in pre
        carry_target = shadowed and (
            (isinstance(pre[target], ir.Value) and pre[target].type == iv_t)
            or (type(pre[target]) is int and semantic.literal_fits(pre[target], iv_dt))
        )
        cands = [n for n in assigned_names(node.body) if n in pre and n != target]
        carried = [n for n in cands if isinstance(pre[n], ir.Value) or semantic.is_literal(pre[n])]
        fixed = [n for n in cands if n not in carried]
        types_ = {n: pre[n].type if isinstance(pre[n], ir.Value)
                  else ir.scalar(semantic.literal_dtype(pre[n])) for n in carried}  # fmt: skip
        if carry_target:
            carried.insert(0, target)
            types_[target] = iv_t
        for _ in range(8):
            block = ir.Block([iv_t, *(types_[n] for n in carried)], [target, *carried])
            self.scope = dict(pre)
            for n, a in zip(carried, block.args[1:], strict=True):
                self.scope[n] = a
            self.scope[target] = block.args[0]  # The body sees the counter, not the carry.
            self.runtime_depth += 1
            try:
                with self.b.at(block):
                    self.visit_body(node.body)
            finally:
                self.runtime_depth -= 1
            end = self.scope
            self.scope = pre
            changed, keep = False, []
            for n, a in zip(carried, block.args[1:], strict=True):
                e, p = end[n], pre[n]
                if e is a or (not isinstance(e, ir.Value) and _same_value(e, p)):
                    changed = True  # The body doesn't change it; don't carry it.
                    continue
                if isinstance(e, ir.Value) and e.type != types_[n]:
                    if isinstance(p, ir.Value):
                        raise CompilationError(
                            f"loop-carried variable `{n}` has type {semantic.type_str(p.type)} "
                            f"before the loop but {semantic.type_str(e.type)} at the end of the "
                            "body. Convert it with `.to(...)` so that the type stays the same, "
                            "or initialize it with the final type."
                        )
                    types_[n], changed = e.type, True
                elif not isinstance(e, ir.Value) and not semantic.is_literal(e):
                    raise CompilationError(
                        f"`{n}` is reassigned to {semantic.describe(e)} inside a runtime loop, "
                        "but only numbers and tiles can be loop-carried. Assign it outside the "
                        "loop, or use tl.static_range to unroll the loop."
                    )
                keep.append(n)
            for n in fixed:
                if not _same_value(end[n], pre[n]):
                    raise CompilationError(
                        f"`{n}` holds {semantic.describe(pre[n])}, which can't change inside a "
                        "runtime loop. Only numbers and tiles can be loop-carried."
                    )
            if not changed:
                break
            carried = keep
        else:  # pragma: no cover - the loop converges in at most one retype per variable.
            raise internal_error("couldn't infer the types of the loop-carried variables")

        inits = [self.materialize(pre[n], types_[n], n) for n in carried]
        with self.b.at(block):
            self.b.create("yield", [self.materialize(end[n], types_[n], n) for n in carried])
        op = self.b.create(
            "for", [lbv, ubv, stv, *inits], [types_[n] for n in carried], regions=[ir.Region(block)]
        )
        self.scope = dict(pre)
        for n, r in zip(carried, op.results, strict=True):
            r.name_hint = n
            self.scope[n] = r
        line = self.b.loc.line
        if shadowed and not carry_target:
            del self.scope[target]
            self.scoped_out[target] = (
                f"`{target}` is the loop variable of the runtime `for` at line {line}, but "
                f"before the loop it holds {semantic.describe(pre[target])}, not a "
                f"{iv_dt!r} value. Its value after the loop would depend on whether the loop "
                "runs. Give the loop variable a different name."
            )
        for n in end:
            if n in pre:
                continue
            if n == target:
                self.scoped_out[n] = (
                    f"`{n}` is the loop variable of the runtime `for` at line {line}, so it's "
                    "undefined after the loop when the loop doesn't run. To give it a value for "
                    f"that case, assign it before the loop, for example `{n} = 0`."
                )
            else:
                self.scoped_out[n] = (
                    f"`{n}` is assigned only inside the loop body at line {line}, so it's "
                    "undefined after the loop. Initialize it before the loop."
                )

    # ---- expressions ----

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        return self.lookup(node.id)

    def lookup(self, name: str) -> Any:
        if name in self.scope:
            return self.scope[name]
        if name in self.scoped_out:
            raise CompilationError(self.scoped_out[name])
        if name in self.globals:
            v = self.globals[name]
            if isinstance(v, core.constexpr):
                return v.value
            if is_jit_function(v) or isinstance(v, _ALLOWED_GLOBAL_TYPES) or callable(v):
                _check_tracked(v, f"global `{name}`")
                return v
            raise CompilationError(
                f"global `{name}` is a {type(v).__name__}. Kernels can use only constant globals: "
                "numbers, strings, dtypes, tuples, tl.constexpr values, modules, and functions."
            )
        if hasattr(builtins, name):
            return getattr(builtins, name)
        raise CompilationError(
            f"name `{name}` isn't defined. Assign it before this line, or define it at module "
            "level."
        )

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        obj = core.unwrap(self.visit(node.value))
        attr = node.attr
        if isinstance(obj, ir.Value):
            return self.value_attr(obj, attr)
        if isinstance(obj, (list, dict, set)):
            raise CompilationError(
                f"methods of {type(obj).__name__} aren't supported in kernels. Use a "
                "compile-time tuple instead."
            )
        from enceladus.runtime.jit import is_content_hashed, is_library_callable

        owner_tracked = isinstance(obj, (types.ModuleType, type))
        if not owner_tracked and self._global_rooted(node.value) and not is_content_hashed(obj):
            raise CompilationError(
                f"`{ast.unparse(node)}` reads an attribute of a {type(obj).__name__}, and "
                "the kernel cache can't detect a change to that attribute. Read attributes "
                "only of modules, classes, and constant values such as tuples, or pass the "
                "value as a tl.constexpr argument."
            )
        try:
            v = core.unwrap(getattr(obj, attr))
        except AttributeError:
            hint = ""
            if isinstance(obj, types.ModuleType) and obj.__name__ == "enceladus.language":
                hint = (
                    ". Check the spelling. Some Triton functions don't exist in Enceladus; "
                    "the language reference lists the supported ones."
                )
            raise CompilationError(
                f"{semantic.describe(obj)} has no attribute `{attr}`{hint}"
            ) from None
        if owner_tracked and not is_library_callable(obj, v):
            _check_tracked(v, f"`{ast.unparse(node)}`")
        return v

    def _global_rooted(self, node: ast.expr) -> bool:
        """Returns whether `node` is a chain of attributes and subscripts on a global name,
        which is what the dependency hash follows. A value that a call returns, or that
        a local variable or parameter holds, isn't."""
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return isinstance(node, ast.Name) and node.id not in self.scope

    def value_attr(self, v: ir.Value, attr: str) -> Any:
        t = v.type
        if isinstance(t, ir.DescType):
            if attr == "dtype":
                return t.elem.dtype
            if attr == "block_shape":
                return t.block_shape
            if attr in core.DESC_METHODS:
                return _BoundMethod(core.DESC_METHODS[attr], v)
            raise CompilationError(
                f"tensor descriptors have no attribute `{attr}`. They have `load`, `store`, "
                "`dtype`, and `block_shape`."
            )
        e = ir.elem_of(t)
        if attr == "dtype":
            return e.elem.dtype if isinstance(e, ir.PointerType) else e.dtype
        if attr == "shape":
            return ir.shape_of(t)
        if attr == "numel":
            return t.numel if isinstance(t, ir.TileType) else 1
        if attr == "T":
            return core.BUILTINS["trans"].frontend(self, v)
        if attr in core.TILE_METHODS:
            return _BoundMethod(core.TILE_METHODS[attr], v)
        raise CompilationError(
            f"{semantic.describe(v)} has no attribute `{attr}`. Check the spelling, or call "
            "the tl function of that name, for example `tl.sum(x)` for `x.sum()`."
        )

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        args: list[Any] = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                v = core.unwrap(self.visit(a.value))
                if not isinstance(v, (tuple, list)):
                    raise CompilationError(
                        f"`*` in a call needs a compile-time tuple or list, but got "
                        f"{semantic.describe(v)}"
                    )
                args.extend(v)
            else:
                args.append(self.visit(a))
        kwargs: dict[str, Any] = {}
        for k in node.keywords:
            if k.arg is None:
                raise CompilationError(
                    "`**` in a call isn't supported in kernels. Pass keyword arguments by name."
                )
            kwargs[k.arg] = self.visit(k.value)
        return self.call(func, args, kwargs)

    def call(self, func: Any, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if isinstance(func, _BoundMethod):
            return func.fn.frontend(self, func.receiver, *args, **kwargs)
        if isinstance(func, core.Builtin):
            return func.frontend(self, *args, **kwargs)
        if is_jit_function(func):
            return self.call_function(func, args, kwargs)
        if func is core.constexpr:
            return core.unwrap(args[0])
        runtime = [a for a in [*args, *kwargs.values()] if isinstance(core.unwrap(a), ir.Value)]
        name = getattr(func, "__name__", type(func).__name__)
        if func is builtins.range:
            raise CompilationError("range() is supported only as the iterable of a `for` loop")
        if func is builtins.abs and runtime and len(args) == 1:
            return core.BUILTINS["abs"].frontend(self, args[0])
        if runtime:
            hints = {
                "min": "tl.minimum", "max": "tl.maximum", "print": "tl.device_print",
                "len": "x.shape[0]", "sum": "tl.sum",
                "float": "x.to(tl.float32)", "int": "x.to(tl.int32)", "bool": "x != 0",
            }  # fmt: skip
            hint = f" Use {hints[name]} instead." if name in hints else (
                " Call a @enceladus.jit function instead."
            )
            raise CompilationError(
                f"the Python function `{name}` can't take runtime values such as "
                f"{semantic.describe(core.unwrap(runtime[0]))}.{hint}"
            )
        if not callable(func):
            raise CompilationError(
                f"{semantic.describe(func)} isn't a function, so you can't call it"
            )
        self._check_helper(func, name)
        return core.unwrap(func(*[core.unwrap(a) for a in args],
                                **{k: core.unwrap(v) for k, v in kwargs.items()}))  # fmt: skip

    def _check_helper(self, func: Any, name: str) -> None:
        """Refuses a call to Python code that reads a value the kernel cache can't track.

        The frontend doesn't see inside a plain Python helper, a class, or a method that
        runs at compile time, so the dependency hash follows them instead. If one of them
        reaches a value that the hash can't track, such as an attribute of an object
        instance, a change to that value wouldn't recompile the kernel.

        Raises:
            CompilationError: `func` reaches a value that the dependency hash can't track.
        """
        target = func.__func__ if isinstance(func, types.MethodType) else func
        wrapped = getattr(target, "__wrapped__", None)
        if not isinstance(target, (types.FunctionType, functools.partial, type)) and \
                not isinstance(wrapped, types.FunctionType):  # fmt: skip
            return
        if id(target) in self._tracked_helpers:
            return
        from enceladus.runtime.jit import untracked_reason

        reason = untracked_reason(func)
        if reason is not None:
            raise CompilationError(
                f"the call to `{name}` runs Python code at compile time, and {reason}. The "
                "kernel cache can't detect a change to that object. In helpers, read only "
                "modules, classes, functions, and constant values such as tuples, or pass "
                "the value as a tl.constexpr argument."
            )
        self._tracked_helpers[id(target)] = target

    def binop(self, op: ast.operator, x: Any, y: Any) -> Any:
        x, y = core.unwrap(x), core.unwrap(y)
        if not isinstance(x, ir.Value) and not isinstance(y, ir.Value):
            return _PY_BINOPS[type(op)](x, y)
        name = _BINOPS.get(type(op))
        if name is None:
            sym = {ast.Pow: "**", ast.MatMult: "@"}.get(type(op), type(op).__name__)
            hint = (" Use tl.dot(a, b)." if isinstance(op, ast.MatMult) else
                    " Multiply explicitly, as in `x * x`, or use tl.exp2 and tl.log2.")
            raise CompilationError(f"`{sym}` isn't supported on runtime values.{hint}")
        return semantic.binary(self.b, name, x, y)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        return self.binop(node.op, self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op = node.op
        if isinstance(op, ast.Not):
            v = self.visit_truth(node.operand)
        else:
            v = core.unwrap(self.visit(node.operand))
        if not isinstance(v, ir.Value):
            return {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert,
                    ast.Not: operator.not_}[type(op)](v)  # fmt: skip
        if isinstance(op, ast.UAdd):
            return v
        if isinstance(op, ast.USub):
            return semantic.unary(self.b, "neg", v)
        if isinstance(op, ast.Invert):
            return semantic.unary(self.b, "not", v)
        if semantic.is_tile(v):
            raise CompilationError(
                "`not` on a tile isn't supported, because a tile has no single truth value. To "
                "negate a mask elementwise, use `~mask` or `x == 0`."
            )
        return semantic.unary(self.b, "not", semantic.to_bool(self.b, v))

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        truth, self.truth_context = self.truth_context, False
        return self.bool_chain(node.op, node.values, truth)

    def visit_truth(self, node: ast.expr) -> Any:
        """Visits an expression whose truth value is all that matters, such as an `if` test.

        `and` and `or` chains then combine the truth values of their operands, so the
        operands can have different types.
        """
        self.truth_context = isinstance(node, ast.BoolOp)  # Read by `visit_BoolOp`.
        return core.unwrap(self.visit(node))

    def bool_chain(self, op: ast.boolop, values: Sequence[ast.expr], truth: bool) -> Any:
        """Emits `values[0] and values[1] and ...` (or `or`) with Python's semantics.

        `a and b` is `a` when `a` is falsy and `b` otherwise; `a or b` is `a` when `a` is
        truthy and `b` otherwise. The right side runs only when it's needed: a compile-time
        left side folds, and a runtime one emits an `if` op, unless the right side is
        cheap and has no side effects. With `truth`, operands become `int1` truth values.
        """
        is_and = isinstance(op, ast.And)
        kw = "and" if is_and else "or"
        node = values[0]
        v = self.visit_truth(node) if truth else core.unwrap(self.visit(node))
        if len(values) == 1:
            return self._truth_value(v, kw) if truth and isinstance(v, ir.Value) else v
        if not isinstance(v, ir.Value):
            if bool(v) != is_and:
                return v  # Short-circuits on a compile-time value.
            return self.bool_chain(op, values[1:], truth)
        saved = self.b.loc
        self.b.loc = self.src.loc(node)
        try:
            v = self._truth_value(v, kw) if truth else self._bool_operand(v, kw)
            cond = semantic.to_bool(self.b, v)
        finally:
            self.b.loc = saved
        rest_blk = ir.Block()
        with self.b.at(rest_blk):
            rest = self.bool_chain(op, values[1:], truth)
            speculatable = all(o.name in _SPECULATABLE for o in rest_blk.ops)
            if (
                not isinstance(rest, ir.Value)
                and bool(rest) != is_and
                and speculatable
                and (truth or (v.type == ir.i1 and isinstance(rest, bool)))
            ):
                # A compile-time right side decides the result either way: `c and False`
                # is falsy and `c or True` is truthy. Folding keeps code that the result
                # guards, such as `if pid > 0 and HAS_BIAS:`, from being compiled.
                return bool(rest)
            if truth and not isinstance(rest, ir.Value):
                rest = bool(rest)
            if isinstance(rest, ir.Value):
                rest = self._bool_operand(rest, kw)
            rest = self._bool_rest(rest, v, kw)
        x, y = (rest, v) if is_and else (v, rest)
        if speculatable:
            for o in rest_blk.ops:
                self.b.block.append(o)
            if v.type == ir.i1:
                return semantic.binary(self.b, kw, v, rest)
            return self.b.create("select", [cond, x, y], [v.type]).result
        keep_blk = ir.Block()
        then_blk, else_blk = (rest_blk, keep_blk) if is_and else (keep_blk, rest_blk)
        (r,) = self._emit_if(cond, then_blk, else_blk, [(f"the result of `{kw}`", x, y)])
        return r

    def _bool_operand(self, v: ir.Value, kw: str) -> ir.Value:
        if semantic.is_tile(v):
            raise CompilationError(
                f"`{kw}` on a tile isn't supported, because a tile has no single truth value. "
                f"To combine masks elementwise, use `{'&' if kw == 'and' else '|'}` instead, "
                "with parentheses around comparisons."
            )
        if not isinstance(v.type, ir.ScalarType):
            raise CompilationError(
                f"`{kw}` needs numeric scalar operands, but got {semantic.describe(v)}"
            )
        return v

    def _truth_value(self, v: ir.Value, kw: str) -> ir.Value:
        return semantic.to_bool(self.b, self._bool_operand(v, kw))

    def _bool_rest(self, rest: Any, left: ir.Value, kw: str) -> ir.Value:
        """Returns the right side of `and` or `or` as a value of the left side's type."""
        t = left.type
        if isinstance(rest, ir.Value):
            if rest.type == t:
                return rest
        elif semantic.is_literal(rest):
            # The literal must keep its value in the left side's type: `flag or 5` can't
            # become `int1`, and `n or 2.5` can't become `int32`.
            dt = t.dtype
            if dt.is_floating():
                return semantic.const(self.b, rest, dt)
            lo, hi = semantic.int_range(dt)
            if not isinstance(rest, float) and lo <= rest <= hi:
                return semantic.const(self.b, rest, dt)
        raise CompilationError(
            f"`{kw}` returns one of its operands unchanged, so both need the same type, but "
            f"got {semantic.type_str(t)} and {semantic.describe(rest)}. Convert one operand "
            "with `.to(...)`, or "
            "use tl.where to choose between values of different types."
        )

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = core.unwrap(self.visit(node.left))
        return self.compare_chain(left, node.ops, node.comparators)

    def compare_chain(self, left: Any, ops: Sequence[ast.cmpop],
                      comps: Sequence[ast.expr]) -> Any:  # fmt: skip
        """Emits `left ops[0] comps[0] ops[1] comps[1] ...` with Python's semantics.

        `a < b < c` means `a < b and b < c`: `b` runs once, and `c` runs only when `a < b`
        holds. A chain on tiles is an error, because `and` needs a single truth value.
        """
        right = core.unwrap(self.visit(comps[0]))
        op = ops[0]
        if isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)) or not (
            isinstance(left, ir.Value) or isinstance(right, ir.Value)
        ):
            r = _PY_CMPOPS[type(op)](left, right)
        else:
            r = semantic.binary(self.b, _CMPOPS[type(op)], left, right)
        if len(ops) == 1:
            return r
        if not isinstance(r, ir.Value):
            return self.compare_chain(right, ops[1:], comps[1:]) if r else r
        _check_chain_scalar(r)
        rest_blk = ir.Block()
        with self.b.at(rest_blk):
            rest = self.compare_chain(right, ops[1:], comps[1:])
            if isinstance(rest, ir.Value):
                _check_chain_scalar(rest)
                rest = semantic.to_bool(self.b, rest)
            else:
                rest = semantic.const(self.b, bool(rest), core.int1)
        if all(o.name in _SPECULATABLE for o in rest_blk.ops):
            for o in rest_blk.ops:
                self.b.block.append(o)
            return semantic.binary(self.b, "and", r, rest)
        (res,) = self._emit_if(r, rest_blk, ir.Block(), [("the comparison", rest, False)])
        return res

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        cond = self.visit_truth(node.test)
        if not isinstance(cond, ir.Value):
            return self.visit(node.body if cond else node.orelse)
        c = self._runtime_cond(cond, "a conditional expression")
        then_blk, else_blk = ir.Block(), ir.Block()
        with self.b.at(then_blk):
            tv = core.unwrap(self.visit(node.body))
        with self.b.at(else_blk):
            ev = core.unwrap(self.visit(node.orelse))
        (r,) = self._emit_if(c, then_blk, else_blk, [("value", tv, ev)])
        return r

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        obj = core.unwrap(self.visit(node.value))
        idx = core.unwrap(self.visit(node.slice))
        if isinstance(obj, ir.Value):
            return self.index_tile(obj, idx)
        if isinstance(idx, ir.Value):
            raise CompilationError(
                f"indexing {type(obj).__name__} needs a compile-time index, but got "
                f"{semantic.describe(idx)}. Use a literal, a tl.constexpr value, or the "
                "variable of a tl.static_range loop."
            )
        return core.unwrap(obj[idx])

    def index_tile(self, v: ir.Value, idx: Any) -> ir.Value:
        items = idx if isinstance(idx, tuple) else (idx,)
        rank = len(ir.shape_of(v.type))
        if isinstance(v.type, ir.DescType) or rank == 0:
            raise CompilationError(
                f"{semantic.describe(v)} can't be indexed. Only tiles take subscripts, such as "
                "`x[:, None]`."
            )
        axis, kept = 0, 0
        for it in items:
            if it is None:
                v = semantic.expand_dims(self.b, v, axis)
                axis += 1
            elif isinstance(it, slice) and it == slice(None):
                axis += 1
                kept += 1
            else:
                raise CompilationError(
                    "tiles support only `None` and `:` in subscripts, as in `x[:, None]`, but "
                    f"got {semantic.describe(it)}. To select elements, use a mask or tl.where, "
                    "or load the element from memory with tl.load."
                )
        if kept > rank:
            raise CompilationError(
                f"too many `:` for a tile of rank {rank}. Use at most one `:` per dimension."
            )
        return v

    def visit_Slice(self, node: ast.Slice) -> slice:
        parts = [core.unwrap(self.visit(p)) if p is not None else None
                 for p in (node.lower, node.upper, node.step)]  # fmt: skip
        if any(isinstance(p, ir.Value) for p in parts):
            raise CompilationError(
                "slice bounds must be compile-time constants. Use literals or tl.constexpr "
                "values."
            )
        return slice(*parts)

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        out: list[Any] = []
        for e in node.elts:
            if isinstance(e, ast.Starred):
                v = core.unwrap(self.visit(e.value))
                if not isinstance(v, (tuple, list)):
                    raise CompilationError(
                        f"`*` in a tuple needs a compile-time tuple, but got {semantic.describe(v)}"
                    )
                out.extend(v)
            else:
                out.append(self.visit(e))
        return tuple(out)

    def visit_List(self, node: ast.List) -> list:
        out = list(self.visit_Tuple(ast.Tuple(elts=node.elts, ctx=node.ctx)))
        _check_no_tile_list(out)
        return out

    def visit_JoinedStr(self, node: ast.JoinedStr) -> str:
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            else:
                parts.append(self.visit(v))
        return "".join(parts)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> str:
        v = core.unwrap(self.visit(node.value))
        if isinstance(v, ir.Value):
            return f"<{v.type}>"
        spec = self.visit(node.format_spec) if node.format_spec is not None else ""
        if node.conversion == ord("r"):
            v = repr(v)
        elif node.conversion == ord("s"):
            v = str(v)
        return format(v, spec)


def _same_value(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if isinstance(a, ir.Value) or isinstance(b, ir.Value):
        return False
    try:
        return type(a) is type(b) and bool(a == b)
    except Exception:  # noqa: BLE001 - comparing arbitrary compile-time objects.
        return False


def _check_chain_scalar(v: ir.Value) -> None:
    if semantic.is_tile(v):
        raise CompilationError(
            "chained comparisons such as `a < b < c` aren't supported on tiles, because Python "
            "evaluates them as `a < b and b < c`, and a tile has no single truth value. "
            "Combine the comparisons elementwise with `&`, as in `(a < b) & (b < c)`."
        )


def _counter_dtype(bounds: Sequence[Any]) -> core.dtype:
    """Returns the type of a `for` counter that holds every value of the bounds and step.

    Bounds that fit in `int32` count in `int32`. Bounds of type `uint32` or `int64`, or
    literals outside the `int32` range, count in `int64`. Bounds of type `uint64` count in
    `uint64` when no bound is signed or negative.

    Raises:
        CompilationError: A `uint64` bound meets a signed bound or a negative literal.
    """

    def fits(v: Any, dt: core.dtype) -> bool:
        if isinstance(v, ir.Value):
            lo, hi = semantic.int_range(v.type.dtype)
            dlo, dhi = semantic.int_range(dt)
            return dlo <= lo and hi <= dhi
        return semantic.literal_fits(v, dt)

    for dt in (core.int32, core.int64, core.uint64):
        if all(fits(v, dt) for v in bounds):
            return dt
    raise CompilationError(
        "range() has a tl.uint64 bound and a signed bound or negative step, and no counter "
        "type holds the values of both. Convert the bounds with `.to(tl.int64)`, or make "
        "them all unsigned."
    )


def _check_no_tile_list(v: Any) -> None:
    if isinstance(v, list) and any(isinstance(x, ir.Value) and semantic.is_tile(x) for x in v):
        raise CompilationError("lists of tiles aren't supported. Use a tuple instead.")


def _last_line(fdef: ast.FunctionDef) -> int:
    return getattr(fdef, "end_lineno", None) or fdef.lineno


def _attr_value(v: Any) -> Any:
    """Converts a constexpr value to a deterministic module attribute."""
    v = core.unwrap(v)
    if isinstance(v, (bool, int, float, str, type(None), core.dtype)):
        return v
    if isinstance(v, (tuple, list)):
        return [_attr_value(x) for x in v]
    return getattr(v, "__qualname__", None) or getattr(v, "__name__", None) or type(v).__name__


def build_ir(
    fn: Any,
    arg_types: Mapping[str, ir.Type],
    arg_facts: Mapping[str, Mapping[str, Any]] | None = None,
    constexprs: Mapping[str, Any] | None = None,
    num_warps: int = 4,
    math_mode: str = "relaxed",
    verify: bool = True,
    debug: bool | None = None,
) -> ir.Module:
    """Builds and verifies the IR for one specialization of a kernel.

    This function does no NumPy or device work, so the compiled path can call it directly
    with types and facts computed from launch arguments.

    Args:
        fn: The `JITFunction` to compile.
        arg_types: The IR type of each runtime (non-constexpr) parameter, by name, for
            example `{"x_ptr": ir.PointerType(ir.f32), "n": ir.i32}`.
        arg_facts: Specialization facts per runtime parameter, by name, for example
            `{"n": {"divisibility": 16}, "stride": {"equal_to_1": True}}`. The facts
            become function-argument attributes. They never remove an argument.
        constexprs: The value of each `tl.constexpr` parameter, by name. Missing values
            fall back to the parameter's default.
        num_warps: The number of SIMD groups per program, stored as a module attribute.
        math_mode: `"relaxed"` or `"fast"`, stored as a module attribute.
        verify: Whether to run the IR verifier on the result.
        debug: Whether to keep `tl.device_assert` checks. `None` reads
            `ENCELADUS_DEBUG`. A debug module has the attribute `debug = True`.

    Returns:
        The kernel's `ir.Module`. `str(module)` is the printed IR.

    Raises:
        CompilationError: The kernel uses an unsupported construct or is ill-typed.
    """
    arg_facts = arg_facts or {}
    constexprs = constexprs or {}
    src = fn.source_info()
    def_loc = src.loc(src.tree)
    runtime = [p for p in fn.params if not p.is_constexpr]
    missing = [p.name for p in runtime if p.name not in arg_types]
    if missing:
        raise CompilationError(
            f"no argument type given for {missing}. Pass a value for every runtime parameter.",
            def_loc,
        )
    block = ir.Block([arg_types[p.name] for p in runtime], [p.name for p in runtime])
    facts = [dict(arg_facts.get(p.name, {})) for p in runtime]
    func = ir.Op(
        "func",
        attrs={"sym_name": fn.__name__, "arg_names": [p.name for p in runtime], "arg_attrs": facts},
        regions=[ir.Region(block)],
        loc=def_loc,
    )
    scope: dict[str, Any] = {p.name: a for p, a in zip(runtime, block.args, strict=True)}
    cvals: dict[str, Any] = {}
    for p in fn.params:
        if not p.is_constexpr:
            continue
        if p.name in constexprs:
            cvals[p.name] = core.unwrap(constexprs[p.name])
        elif p.default is not inspect.Parameter.empty:
            cvals[p.name] = core.unwrap(p.default)
        else:
            raise CompilationError(
                f"the kernel needs a value for the tl.constexpr parameter `{p.name}`. Pass it "
                f"as a keyword argument, for example `{p.name}=128`.",
                def_loc,
            )
        if isinstance(cvals[p.name], ir.Value):
            raise CompilationError(
                f"the tl.constexpr parameter `{p.name}` needs a compile-time Python value, such "
                "as an int, a float, a bool, a string, or a dtype",
                def_loc,
            )
    scope.update(cvals)
    known_one = {id(a) for a, f in zip(block.args, facts, strict=True) if f.get("equal_to_1")}
    builder = ir.Builder()
    builder.block = block
    if debug is None:
        debug = core.debug_enabled()
    cg = CodeGenerator(fn, builder, known_one)
    cg.debug = debug
    cg.run_kernel(scope)
    attrs = {
        "num_warps": num_warps,
        "math_mode": math_mode,
        "constexprs": {k: _attr_value(v) for k, v in cvals.items()},
    }
    if debug:
        attrs["debug"] = True
    module = ir.Module(func, attrs)
    if verify:
        ir.verify(module)
    return module
