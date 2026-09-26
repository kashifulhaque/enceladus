"""Prototype of a Forge "stream": an open command buffer that launches append to,
flushed every `flush_every` dispatches or on sync. Measures sustained per-launch
cost including a minimal Python launcher (argument binding + scalar packing)."""

import json
import struct

import bridges
import kernels
from bench_util import now_ns, stats_us

br = bridges.NanobindBridge()
nb = br.m


class Stream:
    def __init__(self, queue, flush_every=64):
        self.q, self.flush_every = queue, flush_every
        self.batch, self.pending = None, 0

    def dispatch(self, pso, bufs, args, args_index, grid, tg):
        if self.batch is None:
            self.batch = nb.batch_begin(self.q)
        self.batch.dispatch(pso, bufs, args, args_index, grid, tg)
        self.pending += 1
        if self.pending >= self.flush_every:
            self.flush()

    def flush(self, wait=False):
        if self.batch is not None:
            r = self.batch.end(wait)
            self.batch, self.pending = None, 0
            return r

    def synchronize(self):
        self.flush()
        # An empty CB on the same queue completes after all earlier ones.
        nb.batch_begin(self.q).end(True)


class Kernel:
    """Minimal launcher: kernel[grid](*args). Buffers bind in order; scalars are
    packed into one setBytes blob with a precompiled struct format."""

    def __init__(self, pso, sig, stream, tg=256):
        self.pso, self.stream, self.tg = pso, stream, (tg, 1, 1)
        self.nbuf = sig.count("*")
        self.packer = struct.Struct("<" + "".join(c for c in sig if c != "*"))

    def __getitem__(self, grid):
        grid = (grid[0], 1, 1) if len(grid) == 1 else grid

        def launch(*args):
            bufs = args[: self.nbuf]
            blob = self.packer.pack(*args[self.nbuf:])
            self.stream.dispatch(self.pso, bufs, blob, self.nbuf, grid, self.tg)

        return launch


pso = br.pipeline(br.compile(kernels.VECADD), "vecadd")
n = 1024
x, y, o = (br.new_buffer(n * 4) for _ in range(3))
out = {}
# Warm up: the first ~100 ms of GPU work in a process runs at low clocks.
_w = Stream(br.q, 64)
_k = Kernel(pso, "***I", _w)
for _ in range(20000):
    _k[(4,)](x, y, o, n)
_w.synchronize()

import sys  # noqa: E402
for fe in [int(v) for v in sys.argv[1:]] or (1, 16, 64, 256, 1024):
    s = Stream(br.q, flush_every=fe)
    k = Kernel(pso, "***I", s)
    L = 5000
    per_call = []
    tot = []
    for rep in range(3):
        s.synchronize()
        t0 = now_ns()
        for _ in range(L):
            t1 = now_ns()
            k[(4,)](x, y, o, n)
            per_call.append(now_ns() - t1)
        s.synchronize()
        tot.append((now_ns() - t0) / 1e3 / L)
    out[f"flush_every={fe}"] = {"sustained_us_per_launch": sorted(tot)[1], "call": stats_us(per_call)}
    print(fe, out[f"flush_every={fe}"]["sustained_us_per_launch"], out[f"flush_every={fe}"]["call"])
json.dump(out, open("results/stream.json", "w"), indent=1)
