"""An indentation-aware source builder and deterministic identifier generator."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

# MSL and C++ keywords plus names the generated code itself uses.
RESERVED = frozenset(
    """
    alignas alignof and asm auto bool break case catch char class const constexpr
    const_cast continue decltype default delete do double dynamic_cast else enum explicit
    export extern false float for friend goto if inline int long mutable namespace new
    noexcept not nullptr operator or private protected public register reinterpret_cast
    return short signed sizeof static static_assert static_cast struct switch template this
    throw true try typedef typeid typename union unsigned using virtual void volatile
    while xor half bfloat uint ushort uchar ulong device constant threadgroup thread kernel
    vertex fragment metal simd quad vec packed atomic sampler texture buffer
    pid npid lane warp tid tg_mem
    """.split()
)


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

    def fresh(self, hint: str | None) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", hint or "v") or "v"
        if base[0].isdigit():
            base = "v" + base
        if base.startswith("__"):
            base = "u" + base
        n = self._counts.get(base, 0)
        self._counts[base] = n + 1
        name = f"{base}_{n}"
        return f"{name}_" if name in RESERVED else name

    def reserve(self, name: str) -> str:
        """Returns `name` unchanged (escaped if reserved) and prevents clashes with it."""
        safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if safe in RESERVED:
            safe += "_"
        self._counts.setdefault(safe, 0)
        return safe
