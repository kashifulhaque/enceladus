"""Selection of the `tl.dot` backend: `simdgroup_matrix` or Metal 4 `matmul2d`.

The `dot_backend` launch option and `enceladus.Config` field take one of three values:

- `"simdgroup"`: lower every `tl.dot` to `simdgroup_matrix` code.
- `"mpp"`: lower eligible `tl.dot` loops to Metal Performance Primitives (MPP)
  `matmul2d`. A kernel or a `tl.dot` that can't use MPP falls back to `simdgroup`, and
  the `enceladus` logger records the reason at debug level.
- `"auto"`: `"mpp"` on Apple10 (M5) and later GPUs, where `matmul2d` runs on the neural
  accelerators, and `"simdgroup"` on earlier GPUs.
"""

from __future__ import annotations

import logging
import platform

from enceladus.runtime.device import _simd_layout_lock, get_device

DOT_BACKENDS = ("auto", "simdgroup", "mpp")
# Metal 4 `matmul2d` and MSL 4.0 need macOS 26 or later.
MIN_MACOS = 26

log = logging.getLogger("enceladus")

_PROBE_SRC = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
kernel void probe(device half* a [[buffer(0)]], device half* b [[buffer(1)]],
                  device float* c [[buffer(2)]]) {
  metal::tensor<device half, metal::dextents<int, 2>, metal::tensor_inline> ta(
      a, metal::dextents<int, 2>(16, 16), metal::array<int, 2>{1, 16});
  metal::tensor<device half, metal::dextents<int, 2>, metal::tensor_inline> tb(
      b, metal::dextents<int, 2>(16, 16), metal::array<int, 2>{1, 16});
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 16, mpp::tensor_ops::dynamic_length_v<int>);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<1>> op;
  auto sa = ta.slice(0, 0);
  auto sb = tb.slice(0, 0);
  auto ct = op.template get_destination_cooperative_tensor<decltype(sa), decltype(sb), float>();
  op.run(sa, sb, ct);
  for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
    if (ct.is_valid_element(i)) {
      auto idx = ct.get_multidimensional_index(i);
      c[idx[1] * 16 + idx[0]] = ct[i];
    }
  }
}
"""

_mpp_ok: bool | None = None
_mpp_reason = ""
# Autotuning compiles on several threads, and the probe launches on the shared stream,
# which isn't thread-safe. It shares the lock of the `simdgroup_matrix` layout probe,
# which launches on the same stream from compile threads too.
_lock = _simd_layout_lock


def _macos_major() -> int:
    try:
        return int(platform.mac_ver()[0].split(".")[0])
    except ValueError:
        return 0


def _probe() -> tuple[bool, str]:
    caps = get_device().caps
    if not caps.metal4:
        return False, "the device doesn't support the Metal 4 GPU family"
    if _macos_major() < MIN_MACOS:
        return False, f"Metal 4 matmul2d needs macOS {MIN_MACOS} or later"
    import numpy as np

    from enceladus.runtime.raw import metal_kernel
    from enceladus.runtime.tensor import empty, from_numpy

    try:
        k = metal_kernel(_PROBE_SRC, "probe", language_version="4.0")
    except Exception as e:  # noqa: BLE001 - any compile failure disables the backend
        first = str(e).strip().splitlines()
        return False, f"the matmul2d probe kernel didn't compile ({first[0] if first else e})"
    rng = np.random.default_rng(0)
    a = rng.integers(-2, 3, (16, 16)).astype(np.float16)
    b = rng.integers(-2, 3, (16, 16)).astype(np.float16)
    c = empty((16, 16))
    k[(1,), (32,)](from_numpy(a), from_numpy(b), c)
    if not np.array_equal(c.numpy(), a.astype(np.float32) @ b.astype(np.float32)):
        return False, "the matmul2d probe kernel returned wrong results"
    return True, ""


def mpp_supported() -> bool:
    """Returns whether this device and OS run Metal 4 `matmul2d` kernels correctly.

    The first call compiles and runs a small probe kernel; later calls return the cached
    result.
    """
    global _mpp_ok, _mpp_reason
    if _mpp_ok is None:
        with _lock:
            if _mpp_ok is None:
                _mpp_ok, _mpp_reason = _probe()
    return _mpp_ok


def check(backend: str) -> str:
    """Returns `backend` after checking that it's one of `DOT_BACKENDS`.

    Raises:
        ValueError: `backend` isn't "auto", "simdgroup", or "mpp".
    """
    if backend not in DOT_BACKENDS:
        raise ValueError(f"dot_backend must be one of {DOT_BACKENDS}, not {backend!r}")
    return backend


def resolve(backend: str) -> str:
    """Returns the backend that a kernel compiles with: "simdgroup" or "mpp".

    "auto" selects "mpp" on Apple10 and later GPUs that pass the probe. An explicit
    "mpp" on a device that fails the probe falls back to "simdgroup" with a debug log.
    """
    check(backend)
    if backend == "simdgroup":
        return backend
    if backend == "auto":
        if get_device().caps.apple_family < 10:
            return "simdgroup"
        return "mpp" if mpp_supported() else "simdgroup"
    if mpp_supported():
        return "mpp"
    log.debug("enceladus: dot_backend='mpp' falls back to 'simdgroup': %s", _mpp_reason)
    return "simdgroup"
