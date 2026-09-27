"""The batching stream: launches append to an open command buffer.

The stream commits its command buffer every `flush_every` dispatches (64 by default,
`ENCELADUS_FLUSH_EVERY` overrides it) and whenever the host needs results.

Kernels that call `tl.device_print` need a command queue with an `MTLLogState`. The
first such launch moves the stream to a logging queue, after a sync, and the stream
stays there for the rest of the process. Metal delivers log messages on its
own thread after each command buffer completes, in order within a command buffer but
not across command buffers. The stream ends each command buffer that holds a printing
launch with a sentinel kernel, and `synchronize` waits until every sentinel has arrived
before it writes the messages to `sys.stderr`. So every line that a kernel prints
appears before the `synchronize` call that waits for the kernel returns.
"""

from __future__ import annotations

import ctypes
import sys
from typing import TYPE_CHECKING, Any

from enceladus import _C

if TYPE_CHECKING:
    from enceladus.runtime.device import Device

# Keep-alive lists longer than this force a sync, which bounds memory held by
# asynchronous launches over host memory.
_MAX_KEEPALIVE = 4096

LOG_BUFFER_BYTES = 8 << 20
"""The size of the GPU buffer that holds log messages until Metal reads them."""
LOG_TIMEOUT = 10.0
"""Seconds that `synchronize` waits for log messages after the GPU finishes."""
LOG_SENTINEL = "__enceladus_log_sentinel__"
_SENTINEL_SRC = f"""
#include <metal_stdlib>
#include <metal_logging>
using namespace metal;
[[kernel]] void enceladus_log_sentinel() {{ os_log_default.log("{LOG_SENTINEL}"); }}
"""


class Stream:
    """An ordered queue of GPU work, like a CUDA stream."""

    def __init__(self, device: Device, flush_every: int = 64) -> None:
        self.device = device
        self.native = _C.Stream(device.queue, flush_every)
        # Objects whose memory the GPU may still use; cleared at each sync.
        self._keepalive: list[Any] = []
        # Callbacks that run after the next sync, such as copy-back of host memory.
        self._after_sync: list[Any] = []
        # The logging queue, once a kernel that prints has launched.
        self.log_queue: _C.Queue | None = None
        # Kernels with device asserts launched since the previous sync, by id.
        self._assert_kernels: dict[int, Any] = {}
        # Sentinels that the stream committed but the logging queue never received.
        self._lost_sentinels = 0

    @property
    def pending(self) -> int:
        """The number of dispatches in the open command buffer."""
        return self.native.pending

    @property
    def flush_every(self) -> int:
        return self.native.flush_every

    @flush_every.setter
    def flush_every(self, value: int) -> None:
        self.native.flush_every = value

    def keep_alive(self, obj: Any) -> None:
        """Holds a reference to `obj` until the next sync."""
        self._keepalive.append(obj)
        if len(self._keepalive) > _MAX_KEEPALIVE:
            self.synchronize()

    def after_sync(self, fn) -> None:
        """Runs `fn()` after the next sync completes."""
        self._after_sync.append(fn)

    def flush(self) -> None:
        """Commits the open command buffer without waiting."""
        self.native.flush()

    def synchronize(self) -> None:
        """Commits pending work and waits for all of it to finish.

        Writes the lines that kernels printed with `tl.device_print` to `sys.stderr`
        before it returns.

        Raises:
            enceladus.MetalError: A command buffer since the previous sync failed.
            enceladus.DeviceAssertionError: A `tl.device_assert` failed in a kernel that
                ran since the previous sync.
        """
        # Take what belongs to the work launched so far before waiting for it. Launches
        # register these after they dispatch, so each entry's dispatch is in the wait
        # below. Another thread can launch during the wait; what it registers stays for
        # a later sync.
        keepalive, self._keepalive = self._keepalive, []
        callbacks, self._after_sync = self._after_sync, []
        asserts, self._assert_kernels = self._assert_kernels, {}
        failed = True
        try:
            self.native.sync()
            failed = False
        finally:
            keepalive.clear()
            if self.log_queue is not None:
                # A failed command buffer might never log its sentinel.
                self._forward_logs(0.1 if failed else LOG_TIMEOUT)
            if failed:
                # The command buffer error is what this sync reports. Reset the assert
                # buffers, so that a later sync doesn't report a stale assert.
                _reset_asserts(asserts)
        for fn in callbacks:
            fn()
        if asserts:
            _check_asserts(asserts)

    # ---- device printing ----

    def enable_logging(self) -> None:
        """Moves the stream to a command queue that collects shader log messages.

        Raises:
            RuntimeError: `enceladus.capture` is recording, and Metal can't capture a
                queue that logs.
        """
        if self.log_queue is not None:
            return
        from enceladus.runtime import debug

        if debug.capture_active():
            raise RuntimeError(
                "a kernel that calls tl.device_print can't run while enceladus.capture() "
                "records a GPU trace, because Metal can't capture shader logging. Remove the "
                "print or launch the kernel outside the capture block."
            )
        from enceladus.runtime.raw import compile_pipeline

        sentinel = compile_pipeline(_SENTINEL_SRC, "enceladus_log_sentinel", "3.2",
                                    enable_logging=True)  # fmt: skip
        self.synchronize()
        queue = _C.new_logging_queue(self.device.native, LOG_BUFFER_BYTES, LOG_SENTINEL)
        native = _C.Stream(queue, self.native.flush_every)
        native.set_log_sentinel(sentinel)
        self.native, self.log_queue = native, queue

    def _forward_logs(self, timeout: float) -> None:
        q = self.log_queue
        expected = self.native.log_sentinels - self._lost_sentinels
        complete = expected <= 0 or q.wait_log_sentinels(expected, timeout)
        lines = q.drain_logs()
        if not complete:
            # A sentinel can go missing, for example when kernels print more than the log
            # buffer holds or a command buffer fails. Stop waiting for it, or every later
            # sync would wait the full timeout too.
            self._lost_sentinels += max(expected - q.log_sentinels, 0)
            lines.append("enceladus: some tl.device_print output didn't arrive in time. It "
                         "might appear after a later sync, or Metal might have dropped it "
                         "because the kernels printed more than the log buffer holds.")  # fmt: skip
        if lines:
            sys.stderr.write("".join(line + "\n" for line in lines))
            sys.stderr.flush()

    # ---- device asserts ----

    def watch_asserts(self, kernel: Any) -> None:
        """Checks `kernel.assert_buffer` for a failed device assert at the next sync."""
        self._assert_kernels[id(kernel)] = kernel


def _check_asserts(kernels: dict[int, Any]) -> None:
    """Raises the first failed device assert in `kernels`, and resets their buffers."""
    error = None
    for k in kernels.values():
        words = (ctypes.c_uint32 * 5).from_address(k.assert_buffer.ptr)
        if words[0] and error is None:
            error = k.assert_error(words[1], (words[2], words[3], words[4]))
        ctypes.memset(words, 0, ctypes.sizeof(words))
    if error is not None:
        raise error


def _reset_asserts(kernels: dict[int, Any]) -> None:
    """Clears the assert buffers of `kernels` without reporting what they hold."""
    for k in kernels.values():
        ctypes.memset(k.assert_buffer.ptr, 0, 5 * ctypes.sizeof(ctypes.c_uint32))


def synchronize() -> None:
    """Waits for all work launched on the default stream to finish."""
    from enceladus.runtime.device import _device

    if _device is not None:
        _device.stream.synchronize()
