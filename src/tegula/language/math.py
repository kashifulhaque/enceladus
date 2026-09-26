"""Elementwise math functions. Each needs a floating-point input.

`float16` and `bfloat16` inputs compute in `float32` in the interpreter and round the
result back to the input dtype.
"""

from __future__ import annotations

from tegula.compiler import semantic
from tegula.interpreter import interp as I  # noqa: N812
from tegula.language.core import TILE_METHODS, Builtin, builtin

_DOCS = {
    "exp": "Returns e raised to the power `x`, elementwise.",
    "exp2": "Returns 2 raised to the power `x`, elementwise.",
    "log": "Returns the natural logarithm of `x`, elementwise.",
    "log2": "Returns the base-2 logarithm of `x`, elementwise.",
    "sqrt": "Returns the square root of `x`, elementwise.",
    "rsqrt": "Returns `1 / sqrt(x)`, elementwise.",
    "sin": "Returns the sine of `x` (in radians), elementwise.",
    "cos": "Returns the cosine of `x` (in radians), elementwise.",
    "tanh": "Returns the hyperbolic tangent of `x`, elementwise.",
    "sigmoid": "Returns `1 / (1 + exp(-x))`, elementwise.",
    "erf": "Returns the Gauss error function of `x`, elementwise.",
    "floor": "Returns the largest integer value not greater than `x`, elementwise.",
    "ceil": "Returns the smallest integer value not less than `x`, elementwise.",
}


def _make(op: str) -> Builtin:
    def frontend(ctx, x):
        return semantic.unary(ctx.b, op, x, name=op)

    def interp(x):
        return I.unary(op, x, name=op)

    frontend.__doc__ = _DOCS[op]
    b = builtin(interp=interp, name=op)(frontend)
    TILE_METHODS[op] = b
    return b


exp = _make("exp")
exp2 = _make("exp2")
log = _make("log")
log2 = _make("log2")
sqrt = _make("sqrt")
rsqrt = _make("rsqrt")
sin = _make("sin")
cos = _make("cos")
tanh = _make("tanh")
sigmoid = _make("sigmoid")
erf = _make("erf")
floor = _make("floor")
ceil = _make("ceil")
