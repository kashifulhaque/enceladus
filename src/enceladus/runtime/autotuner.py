"""`@enceladus.autotune`, `@enceladus.heuristics`, and `enceladus.Config`.

The autotuner compiles every candidate configuration in parallel (Metal compilation
releases the GIL), times each with GPU timestamps, rejects probable register-spill
cliffs, and remembers the winner per key, in memory and on disk under
`~/.cache/enceladus/autotune/<kernel-hash>/<architecture>.json`.

Set `ENCELADUS_PRINT_AUTOTUNING=1` to print each winner as a pasteable `enceladus.Config(...)`.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from enceladus.runtime import cache
from enceladus.runtime.device import get_device
from enceladus.runtime.tensor import Tensor, from_numpy

log = logging.getLogger("enceladus.autotune")
SPILL_FACTOR = 3.0  # configs slower than this multiple of the median are rejected


@dataclass
class Config:
    """One candidate configuration: constexpr values plus launch options.

    Args:
        kwargs: Constexpr values, for example `{"BM": 64, "BN": 64, "BK": 32}`.
        num_warps: SIMD groups per threadgroup.
        dot_warps: The (WM, WN) grid of SIMD groups for `tl.dot`, or None for the default.
        dot_backend: "auto" or "simdgroup". ("mpp" arrives in M8.)
        pre_hook: Called with the bound arguments before each launch of this config.
    """

    kwargs: dict[str, Any]
    num_warps: int = 4
    dot_warps: tuple[int, int] | None = None
    dot_backend: str = "auto"
    pre_hook: Callable[[dict[str, Any]], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.dot_backend not in ("auto", "simdgroup"):
            raise ValueError(f"dot_backend {self.dot_backend!r} isn't available; use 'auto' or "
                             "'simdgroup'")  # fmt: skip
        if self.dot_warps is not None:
            self.dot_warps = tuple(self.dot_warps)

    def launch_options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {"num_warps": self.num_warps}
        if self.dot_warps is not None:
            opts["dot_warps"] = self.dot_warps
        return opts

    def __str__(self) -> str:
        parts = [repr(self.kwargs), f"num_warps={self.num_warps}"]
        if self.dot_warps is not None:
            parts.append(f"dot_warps={self.dot_warps}")
        return f"enceladus.Config({', '.join(parts)})"

    def to_json(self) -> dict[str, Any]:
        """Returns the fields that identify this config. `pre_hook` can't be serialized."""
        return {"kwargs": self.kwargs, "num_warps": self.num_warps,
                "dot_warps": list(self.dot_warps) if self.dot_warps else None,
                "dot_backend": self.dot_backend}  # fmt: skip

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Config:
        """Rebuilds a config from `to_json` output, without its `pre_hook`."""
        dw = d.get("dot_warps")
        return cls(dict(d["kwargs"]), d["num_warps"], tuple(dw) if dw else None,
                   d.get("dot_backend", "auto"))  # fmt: skip

    def _identity(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True)


def _innermost(fn: Any):
    while not hasattr(fn, "_binder") and hasattr(fn, "fn"):
        fn = fn.fn
    return fn


class Heuristics:
    """Computes constexpr values from the other arguments. Created by `@enceladus.heuristics`."""

    def __init__(self, fn: Any, values: dict[str, Callable[[dict[str, Any]], Any]]) -> None:
        self.fn = fn
        self.values = values
        self.jit = _innermost(fn)
        self.arg_names = self.jit.arg_names

    def resolve(self, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
        named = dict(zip(self.arg_names, args, strict=False))
        named.update(kwargs)
        extra = {k: f(named) for k, f in self.values.items()}
        return {**kwargs, **extra}

    def __getitem__(self, grid: Any) -> Callable[..., None]:
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    def run(self, *args: Any, grid: Any, **kwargs: Any) -> None:
        self.fn.run(*args, grid=grid, **self.resolve(args, kwargs))

    def warmup(self, *args: Any, **kwargs: Any):
        return self.fn.warmup(*args, **self.resolve(args, kwargs))


def heuristics(values: dict[str, Callable[[dict[str, Any]], Any]]):
    """Decorates a kernel so that constexpr values come from functions of the arguments.

    Example:
        @enceladus.heuristics({"BLOCK": lambda args: enceladus.next_power_of_2(args["n"])})
    """
    return lambda fn: Heuristics(fn, values)


class Autotuner:
    """Picks the fastest `Config` per key. Created by `@enceladus.autotune`."""

    def __init__(
        self,
        fn: Any,
        configs: Sequence[Config],
        key: Sequence[str],
        prune_configs_by: dict[str, Callable] | None = None,
        reset_to_zero: Iterable[str] | None = None,
        restore_value: Iterable[str] | None = None,
        warmup_ms: float = 50,
        rep: int = 20,
    ) -> None:
        if not configs:
            raise ValueError("@enceladus.autotune needs at least one Config")
        self.fn = fn
        self.jit = _innermost(fn)
        self.configs = list(configs)
        self.key = list(key)
        self.prune = prune_configs_by or {}
        self.reset_to_zero = list(reset_to_zero or [])
        self.restore_value = list(restore_value or [])
        self.warmup_ms = warmup_ms
        self.rep = rep
        self.arg_names = self.jit.arg_names
        for k in (*self.key, *self.reset_to_zero, *self.restore_value):
            if k not in self.arg_names:
                raise ValueError(f"@enceladus.autotune names unknown argument {k!r}")
        self.best: dict[str, Config] = {}
        self._saved: dict[str, Config] = {}  # Results loaded from disk, not yet used.
        self.timings: dict[str, dict[str, float]] = {}
        self._loaded = False
        self._lock = threading.Lock()

    # ---- persistence ----

    def _path(self):
        arch = get_device().caps.architecture or "unknown"
        return cache.cache_dir() / "autotune" / self.jit.cache_key[:32] / f"{arch}.json"

    def _load(self) -> None:
        self._loaded = True
        if cache.env_flag("ENCELADUS_ALWAYS_COMPILE"):
            return
        try:
            data = json.loads(self._path().read_text())
        except (OSError, ValueError):
            return
        for k, v in data.items():
            try:
                self._saved.setdefault(k, Config.from_json(v))
            except (KeyError, TypeError, ValueError):
                log.debug("ignoring malformed saved result %r for key %s", v, k)

    def _from_saved(self, key: str, named: dict[str, Any]) -> Config | None:
        """Returns the candidate config that matches the saved result for `key`, if any.

        Matching against the candidates rather than `configs` keeps the config's `pre_hook`,
        and finds configs that `early_config_prune` builds. A saved result that matches no
        candidate is re-tuned.
        """
        saved = self._saved.pop(key, None)
        if saved is None:
            return None
        ident = saved._identity()
        cfg = next((c for c in self._candidates(named) if c._identity() == ident), None)
        if cfg is None:
            log.debug("ignoring saved result %s for key %s: it isn't a current candidate",
                      saved, key)  # fmt: skip
        return cfg

    def _save(self) -> None:
        p = self._path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if p.exists():
                data = json.loads(p.read_text())
            data.update({k: c.to_json() for k, c in self.best.items()})
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
            tmp.replace(p)
        except (OSError, ValueError) as e:
            log.debug("couldn't save autotuning results: %s", e)

    # ---- tuning ----

    def _key(self, named: dict[str, Any]) -> str:
        parts = []
        for k in self.key:
            v = named[k]
            parts.append(repr(v) if not hasattr(v, "dtype") else f"{list(v.shape)}")
        dtypes = [str(getattr(named[n], "dtype", "")) for n in self.arg_names
                  if hasattr(named.get(n), "dtype")]  # fmt: skip
        return "|".join(parts) + "|" + ",".join(dtypes)

    def __getitem__(self, grid: Any) -> Callable[..., None]:
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    def config_for(self, *args: Any, **kwargs: Any) -> Config | None:
        """Returns the tuned config for these launch arguments, or None if not tuned yet."""
        named = dict(zip(self.arg_names, args, strict=False))
        named.update(kwargs)
        return self.best.get(self._key(named))

    def _resolve(self, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
        return self.fn.resolve(args, kwargs) if isinstance(self.fn, Heuristics) else kwargs

    def run(self, *args: Any, grid: Any, **kwargs: Any) -> None:
        if not self._loaded:
            self._load()
        named = dict(zip(self.arg_names, args, strict=False))
        named.update(kwargs)
        key = self._key(named)
        cfg = self.best.get(key)
        if cfg is None:
            with self._lock:
                cfg = self.best.get(key) or self._from_saved(key, named)
                if cfg is None:
                    cfg = self._tune(key, args, kwargs, grid, named)
                self.best[key] = cfg
        if cfg.pre_hook is not None:
            cfg.pre_hook(named)
        self.fn.run(*args, grid=grid, **kwargs, **cfg.kwargs, **cfg.launch_options())

    def _candidates(self, named: dict[str, Any]) -> list[Config]:
        configs = self.configs
        early = self.prune.get("early_config_prune")
        if early is not None:
            configs = list(early(configs, named))
        model = self.prune.get("perf_model")
        top_k = self.prune.get("top_k")
        if model is not None and top_k:
            scored = sorted(configs, key=lambda c: model(**named, **c.kwargs,
                                                          num_warps=c.num_warps))  # fmt: skip
            n = int(top_k * len(configs)) if isinstance(top_k, float) else int(top_k)
            configs = scored[: max(1, n)]
        return configs

    def _tune(self, key: str, args: tuple, kwargs: dict, grid: Any, named: dict) -> Config:
        from enceladus.testing import do_bench

        configs = self._candidates(named)
        # Benchmark on enceladus.Tensor views so launches don't synchronize.
        bench_args = tuple(from_numpy(a) if isinstance(a, np.ndarray) else a for a in args)
        bench_kwargs = {k: from_numpy(v) if isinstance(v, np.ndarray) else v
                        for k, v in kwargs.items()}  # fmt: skip
        self.jit._binder()  # build the binder before compiling from threads

        def compile_one(cfg: Config):
            kw = self._resolve(bench_args, {**bench_kwargs, **cfg.kwargs})
            return self.jit.warmup(*bench_args, **kw, **cfg.launch_options(), _record=False)

        workers = max(1, min(8, get_device().caps.max_concurrent_compilations, len(configs)))
        errors: dict[int, Exception] = {}
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(compile_one, c) for c in configs]
            for i, f in enumerate(futures):
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001 - reported below
                    errors[i] = e
        if len(errors) == len(configs):
            raise RuntimeError(
                f"every autotuning config of {self.jit.__name__} failed to compile; the first "
                f"error was: {errors[0]}"
            ) from errors[0]
        for i, e in errors.items():
            warnings.warn(f"{self.jit.__name__}: skipping {configs[i]}: {e}", stacklevel=4)

        named_bench = dict(zip(self.arg_names, bench_args, strict=False))
        named_bench.update(bench_kwargs)
        saved = {n: named_bench[n].numpy().copy() for n in self.restore_value
                 if isinstance(named_bench.get(n), Tensor)}  # fmt: skip
        timings: dict[int, float] = {}
        for i, cfg in enumerate(configs):
            if i in errors:
                continue

            def launch(cfg=cfg) -> None:
                for n in self.reset_to_zero:
                    t = named_bench[n]
                    if isinstance(t, Tensor):
                        t._view()[...] = 0
                if cfg.pre_hook is not None:
                    cfg.pre_hook(named_bench)
                self.fn.run(*bench_args, grid=grid, **bench_kwargs, **cfg.kwargs,
                            **cfg.launch_options())  # fmt: skip

            get_device().stream.synchronize()
            try:
                timings[i] = do_bench(launch, self.warmup_ms, self.rep)
            except Exception as e:  # noqa: BLE001 - a config that fails to launch is skipped
                errors[i] = e
                warnings.warn(f"{self.jit.__name__}: skipping {cfg}: {e}", stacklevel=4)
        get_device().stream.synchronize()
        for n, v in saved.items():
            named_bench[n]._view()[...] = v
        for n in self.reset_to_zero:
            t = named_bench[n]
            if isinstance(t, Tensor):
                t._view()[...] = 0
        if not timings:
            first = next(iter(errors.values()))
            raise RuntimeError(
                f"every autotuning config of {self.jit.__name__} failed to compile or run; the "
                f"first error was: {first}"
            ) from first
        med = statistics.median(timings.values())
        valid = {i: t for i, t in timings.items() if t <= SPILL_FACTOR * med}
        for i in set(timings) - set(valid):
            log.debug("rejecting %s: %.3f ms against a median of %.3f ms (likely spilling)",
                      configs[i], timings[i], med)  # fmt: skip
        best_i = min(valid, key=valid.get)
        best = configs[best_i]
        self.best[key] = best
        self.timings[key] = {str(configs[i]): t for i, t in timings.items()}
        self._save()
        if os.environ.get("ENCELADUS_PRINT_AUTOTUNING", "0") not in ("", "0"):
            print(f"enceladus: autotuning {self.jit.__name__} [{key}] chose {best} "
                  f"({timings[best_i]:.3f} ms)")  # fmt: skip
        return best


def autotune(
    configs: Sequence[Config],
    key: Sequence[str],
    prune_configs_by: dict[str, Callable] | None = None,
    reset_to_zero: Iterable[str] | None = None,
    restore_value: Iterable[str] | None = None,
    warmup_ms: float = 50,
    rep: int = 20,
):
    """Decorates a kernel so that each launch uses the fastest of `configs`.

    Args:
        configs: The candidate configurations.
        key: Argument names whose values select a tuning result. A new combination
            (together with the argument dtypes) triggers tuning.
        prune_configs_by: Optional `{"early_config_prune": fn(configs, named_args)}`, and
            `{"perf_model": fn(**args), "top_k": n}` to benchmark only the top `n`.
        reset_to_zero: Arguments zeroed before each benchmark run and after tuning.
        restore_value: Arguments restored to their original values after tuning.
        warmup_ms: GPU warm-up time before measuring each config.
        rep: Timed repetitions per config.
    """

    def deco(fn: Any) -> Autotuner:
        return Autotuner(fn, configs, key, prune_configs_by, reset_to_zero, restore_value,
                         warmup_ms, rep)  # fmt: skip

    return deco
