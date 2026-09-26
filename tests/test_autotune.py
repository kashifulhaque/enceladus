"""Tests for `@tegula.autotune` and `@tegula.heuristics`."""

import numpy as np
import pytest

import tegula
import tegula.language as tl
from tegula.runtime.autotuner import Autotuner


@tegula.jit
def _accumulate(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.load(out_ptr + offs, mask=mask) + tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, acc, mask=mask)


def _grid(meta):
    return (tegula.cdiv(meta["n"], meta["BLOCK"]),)


CONFIGS = [tegula.Config({"BLOCK": b}) for b in (100, 256, 1024)]  # 100 isn't a power of two


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TEGULA_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TEGULA_ALWAYS_COMPILE", raising=False)
    monkeypatch.setenv("TEGULA_INTERPRET", "0")
    return tmp_path


def test_failing_configs_are_skipped_and_state_is_reset():
    tuned = Autotuner(_accumulate, CONFIGS, key=["n"], reset_to_zero=["out_ptr"], rep=3,
                      warmup_ms=1)  # fmt: skip
    x, out = tegula.randn(5000), tegula.zeros(5000)
    with pytest.warns(UserWarning, match=r"skipping .*'BLOCK': 100.*power of two"):
        tuned[_grid](x, out, 5000)
    # Benchmark runs accumulated into `out`, but it was reset: one real launch remains.
    np.testing.assert_array_equal(out.numpy(), x.numpy())
    assert tuned.best[tuned._key({"n": 5000, "x_ptr": x, "out_ptr": out})].kwargs["BLOCK"] in (
        256, 1024)


def test_all_configs_failing_raises():
    tuned = Autotuner(_accumulate, CONFIGS[:1], key=["n"], rep=3, warmup_ms=1)
    with pytest.raises(RuntimeError, match="every autotuning config"):
        tuned[_grid](tegula.randn(64), tegula.zeros(64), 64)


def test_results_persist_across_processes(monkeypatch):
    first = Autotuner(_accumulate, CONFIGS[1:], key=["n"], reset_to_zero=["out_ptr"], rep=3,
                      warmup_ms=1)  # fmt: skip
    x = tegula.randn(4096)
    first[_grid](x, tegula.zeros(4096), 4096)
    chosen = next(iter(first.best.values()))

    # A fresh autotuner (as in a new process) must reuse the saved result without
    # benchmarking anything.
    def no_bench(*a, **k):
        raise AssertionError("re-benchmarked a persisted key")

    monkeypatch.setattr("tegula.testing.do_bench", no_bench)
    second = Autotuner(_accumulate, CONFIGS[1:], key=["n"], rep=3, warmup_ms=1)
    out = tegula.zeros(4096)
    second[_grid](x, out, 4096)
    assert next(iter(second.best.values())) == chosen
    np.testing.assert_array_equal(out.numpy(), x.numpy())


def test_heuristics_compute_constexprs():
    kernel = tegula.heuristics({"BLOCK": lambda a: tegula.next_power_of_2(a["n"])})(_accumulate)
    x, out = np.arange(300, dtype=np.float32), np.zeros(300, np.float32)
    kernel[(1,)](x, out, 300)
    np.testing.assert_array_equal(out, x)
