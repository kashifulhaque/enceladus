"""Checks that an installed Enceladus wheel imports and runs a kernel.

The kernel runs in interpreter mode, so the check works on CI runners whose GPU can't
run compiled kernels. Run it from outside the source tree so that `import enceladus`
resolves to the installed wheel.
"""

import os

os.environ["ENCELADUS_INTERPRET"] = "1"

import numpy as np  # noqa: E402

import enceladus  # noqa: E402
import enceladus.language as tl  # noqa: E402
from enceladus.runtime import device  # noqa: E402


@enceladus.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
             mask=mask)  # fmt: skip


x = np.arange(1000, dtype=np.float32)
out = np.empty_like(x)
add_kernel[(enceladus.cdiv(x.size, 256),)](x, x, out, x.size, BLOCK=256)
np.testing.assert_array_equal(out, 2 * x)
assert "site-packages" in enceladus.__file__, f"imported from the source tree: {enceladus.__file__}"
print(f"enceladus {enceladus.__version__} from {enceladus.__file__}: OK "
      f"(Metal device created: {device._device is not None})")  # fmt: skip
