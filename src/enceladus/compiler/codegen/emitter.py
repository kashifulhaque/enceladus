"""An indentation-aware source builder and deterministic identifier generator."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

# Names that a user's kernel argument or kernel name must never take, because generated
# code would then refer to the user's value instead of the name it means, or because MSL
# gives them a meaning of their own. `NameGen.reserve` escapes them, and `NameGen.fresh`
# skips them.
_SCALAR_TYPES = "bool char uchar short ushort int uint long ulong half bfloat float"
# Object-like macros that `metal_stdlib` defines, apart from the `METAL_` and `TARGET_OS_`
# families, which `_escape_prefix` handles.
_LIMITS = "DIG DECIMAL_DIG EPSILON MANT_DIG MAX MAX_10_EXP MAX_EXP MIN MIN_10_EXP MIN_EXP RADIX"
_M_CONSTANTS = "E LOG2E LOG10E LN2 LN10 PI PI_2 PI_4 1_PI 2_PI 2_SQRTPI SQRT2 SQRT1_2"
_MACROS = (
    [f"{t}_{s}" for t in ("FLT", "DBL", "HALF", "BFLT") for s in _LIMITS.split()]
    + [f"M_{c}{s}" for c in _M_CONSTANTS.split() for s in ("", "_F", "_H", "_BF")]
    + """
    CHAR_BIT CHAR_MAX CHAR_MIN SCHAR_MAX SCHAR_MIN UCHAR_MAX SHRT_MAX SHRT_MIN USHRT_MAX
    INT_MAX INT_MIN UINT_MAX LONG_MAX LONG_MIN ULONG_MAX LLONG_MAX LLONG_MIN ULLONG_MAX
    FP_ILOGB0 FP_ILOGBNAN HUGE_VAL HUGE_VALF HUGE_VALH HUGE_VALBF INFINITY NAN MAXFLOAT
    MAXHALF MAXBFLOAT MAXDOUBLE
    """.split()
)
RESERVED = frozenset(
    # MSL and C++ keywords, including alternative operator spellings.
    """
    alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t
    char16_t char32_t class compl concept const consteval constexpr constinit const_cast
    continue decltype default delete do double dynamic_cast else enum explicit export extern
    false float for friend goto if inline int long mutable namespace new noexcept not not_eq
    nullptr operator or or_eq private protected public register reinterpret_cast requires
    return short signed sizeof static static_assert static_cast struct switch template this
    thread_local throw true try typedef typeid typename union unsigned using virtual void
    volatile wchar_t while xor xor_eq
    half bfloat uint ushort uchar ulong size_t ptrdiff_t device constant threadgroup thread
    kernel vertex fragment metal simd quad vec packed atomic sampler texture buffer
    threadgroup_imageblock ray_data object_data main
    """.split()
    # Vector types, such as `float4` and `uint3`.
    + [f"{t}{n}" for t in _SCALAR_TYPES.split() for n in (2, 3, 4)]
    # Metal functions, templates, and macros that generated code and the prelude call.
    + """
    abs min max fabs fmin fmax fmod fma exp exp2 log log2 sqrt rsqrt sin cos tanh floor
    ceil as_type simd_sum simd_max simd_min simd_shuffle_xor simdgroup_matrix
    simdgroup_load simdgroup_store simdgroup_multiply_accumulate
    make_filled_simdgroup_matrix threadgroup_barrier mem_flags
    """.split()
    # Metal atomics and SIMD functions that `atomic.py` and `scan.py` call.
    + """
    atomic_int atomic_uint atomic_float atomic_ulong memory_order_relaxed atomic_load_explicit
    atomic_exchange_explicit atomic_compare_exchange_weak_explicit atomic_fetch_add_explicit
    atomic_fetch_max_explicit atomic_fetch_min_explicit atomic_fetch_and_explicit
    atomic_fetch_or_explicit atomic_fetch_xor_explicit atomic_max_explicit atomic_min_explicit
    simd_shuffle simd_prefix_exclusive_sum
    """.split()
    + _MACROS
    # Kernel parameters, and the fixed locals of `msl.py`, `dot.py`, and `reduce.py`: the
    # register loop index `r`, the exchange buffer `buf`, the MMA loop's `kk`, `i`, `j`,
    # `fa`, and `fb`, and the argmax tie flag `tk`.
    + "pid npid lane warp tid tg_mem r buf kk i j fa fb tk".split()
)


# Prefixes that belong to the compiler (`__`), the prelude (`tg_`), or macro families in
# `metal_stdlib` that grow with each SDK (`METAL_`, `TARGET_OS_`).
_PREFIXES = ("__", "tg_", "METAL_", "TARGET_OS_")


def _escape_prefix(name: str) -> str:
    return "u" + name if name.startswith(_PREFIXES) else name


class Emitter:
    """Collects lines of source text with consistent indentation."""

    def __init__(self, indent: str = "  ") -> None:
        self._lines: list[str] = []
        self._level = 0
        self._indent = indent

    def line(self, text: str = "") -> None:
        self._lines.append(self._indent * self._level + text if text else "")

    def lines(self, text: str) -> None:
        for t in text.splitlines():
            self.line(t)

    @contextmanager
    def block(self, header: str, footer: str = "}") -> Iterator[None]:
        self.line(header + " {")
        self._level += 1
        try:
            yield
        finally:
            self._level -= 1
            self.line(footer)

    @contextmanager
    def indented(self) -> Iterator[None]:
        self._level += 1
        try:
            yield
        finally:
            self._level -= 1

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"


class NameGen:
    """Produces readable, unique MSL identifiers from Python names.

    Names get a numeric suffix (`acc_3`) in order of first request, so output is
    deterministic for a deterministic walk of the IR.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._used: set[str] = set(RESERVED)

    def fresh(self, hint: str | None) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", hint or "v") or "v"
        if base[0].isdigit():
            base = "v" + base
        base = _escape_prefix(base)
        while True:
            n = self._counts.get(base, 0)
            self._counts[base] = n + 1
            name = f"{base}_{n}"
            if name not in self._used:
                self._used.add(name)
                return name

    def reserve(self, name: str) -> str:
        """Returns `name` (escaped if reserved) and prevents later clashes with it."""
        safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if safe[0].isdigit():
            safe = "u" + safe
        safe = _escape_prefix(safe)
        while safe in self._used:
            safe += "_"
        self._used.add(safe)
        return safe
