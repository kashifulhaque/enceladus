"""Source locations and `CompilationError`."""

from __future__ import annotations

import linecache
from dataclasses import dataclass


@dataclass(frozen=True)
class Loc:
    """A source location. `line` and `col` are 1-based."""

    file: str
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"

    def source_line(self) -> str | None:
        """Returns the source text of the line, or `None` if it's unavailable."""
        text = linecache.getline(self.file, self.line)
        return text.rstrip("\n") if text else None


class CompilationError(Exception):
    """A kernel uses a construct that Enceladus can't compile, or uses it incorrectly.

    The message starts with `file:line:col`, followed by the source line and a caret under
    the offending column.

    Attributes:
        message: The error message without the location.
        loc: The source location, or `None` if it isn't known yet. The frontend fills in
            the location of the innermost AST node that's being compiled.
    """

    def __init__(self, message: str, loc: Loc | None = None) -> None:
        self.message = message
        self.loc = loc
        super().__init__(message)

    def with_loc(self, loc: Loc | None) -> CompilationError:
        """Sets the location if it isn't set yet, and returns this error."""
        if self.loc is None and loc is not None:
            self.loc = loc
            self.args = (str(self),)
        return self

    def __str__(self) -> str:
        return format_error(self.message, self.loc)


def format_error(message: str, loc: Loc | None) -> str:
    """Formats `message` with a `file:line:col` prefix, the source line, and a caret."""
    if loc is None:
        return message
    out = f"{loc}: {message}"
    text = loc.source_line()
    if text is not None:
        caret = " " * max(0, loc.col - 1) + "^"
        out += f"\n{text}\n{caret}"
    return out


class DeviceAssertionError(AssertionError):
    """A `tl.device_assert` failed while a kernel ran.

    The compiled path raises it when the stream synchronizes after the failing launch;
    the interpreter raises it as soon as the assert fails.

    Attributes:
        message: The assert's message.
        loc: The source location of the `tl.device_assert` call, or `None` if unknown.
        program_id: The (x, y, z) program ID of the first failing program.
    """

    def __init__(self, message: str, loc: Loc | None, program_id: tuple[int, int, int]):
        self.message = message
        self.loc = loc
        self.program_id = tuple(program_id)
        super().__init__(str(self))

    def __str__(self) -> str:
        pid = ", ".join(map(str, self.program_id))
        what = f"device assertion failed in program ({pid})"
        return format_error(f"{what}: {self.message}" if self.message else what, self.loc)
