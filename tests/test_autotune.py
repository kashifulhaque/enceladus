"""Tests for `@enceladus.autotune` and `@enceladus.heuristics`."""

import ml_dtypes
import numpy as np
import pytest

import enceladus
import enceladus.language as tl
from enceladus.configs import matmul_configs
from enceladus.runtime.autotuner import Autotuner


@enceladus.jit
def _accumulate(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.load(out_ptr + offs, mask=mask) + tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, acc, mask=mask)


def _grid(meta):
    return (enceladus.cdiv(meta["n"], meta["BLOCK"]),)


CONFIGS = [enceladus.Config({"BLOCK": b}) for b in (100, 256, 1024)]  # 100 isn't a power of two


def _fail_launch(args):
    raise RuntimeError("launch failed")


def _no_bench(*args, **kwargs):
    raise AssertionError("re-benchmarked a persisted key")


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ENCELADUS_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ENCELADUS_ALWAYS_COMPILE", raising=False)
    monkeypatch.setenv("ENCELADUS_INTERPRET", "0")
    return tmp_path


def test_failing_configs_are_skipped_and_state_is_reset():
    tuned = Autotuner(_accumulate, CONFIGS, key=["n"], reset_to_zero=["out_ptr"], rep=3,
                      warmup_ms=1)  # fmt: skip
    x, out = enceladus.randn(5000), enceladus.zeros(5000)
    with pytest.warns(UserWarning, match=r"skipping .*'BLOCK': 100.*power of two"):
        tuned[_grid](x, out, 5000)
    # Benchmark runs accumulated into `out`, but it was reset: one real launch remains.
    np.testing.assert_array_equal(out.numpy(), x.numpy())
    assert tuned.best[tuned._key({"n": 5000, "x_ptr": x, "out_ptr": out})].kwargs["BLOCK"] in (
        256, 1024)


def test_configs_failing_at_launch_are_skipped():
    configs = [enceladus.Config({"BLOCK": 256}, pre_hook=_fail_launch), CONFIGS[2]]
    tuned = Autotuner(_accumulate, configs, key=["n"], rep=3, warmup_ms=1)
    x, out = enceladus.randn(4096), enceladus.zeros(4096)
    with pytest.warns(UserWarning, match=r"skipping .*'BLOCK': 256.*launch failed"):
        tuned[_grid](x, out, 4096)
    assert tuned.config_for(x, out, 4096).kwargs["BLOCK"] == 1024


@pytest.mark.parametrize(
    "config",
    [CONFIGS[0], enceladus.Config({"BLOCK": 256}, pre_hook=_fail_launch)],
    ids=["compile", "launch"],
)
@pytest.mark.filterwarnings("ignore:.*skipping")
def test_all_configs_failing_raises(config):
    tuned = Autotuner(_accumulate, [config], key=["n"], rep=3, warmup_ms=1)
    with pytest.raises(RuntimeError, match="every autotuning config"):
        tuned[_grid](enceladus.randn(64), enceladus.zeros(64), 64)


def test_results_persist_across_processes(monkeypatch):
    first = Autotuner(_accumulate, CONFIGS[1:], key=["n"], reset_to_zero=["out_ptr"], rep=3,
                      warmup_ms=1)  # fmt: skip
    x = enceladus.randn(4096)
    first[_grid](x, enceladus.zeros(4096), 4096)
    chosen = next(iter(first.best.values()))

    # A fresh autotuner (as in a new process) must reuse the saved result without
    # benchmarking anything.
    monkeypatch.setattr("enceladus.testing.do_bench", _no_bench)
    calls = []
    hooked = [enceladus.Config(c.kwargs, pre_hook=lambda a, b=c.kwargs["BLOCK"]: calls.append(b))
              for c in CONFIGS[1:]]  # fmt: skip
    second = Autotuner(_accumulate, hooked, key=["n"], rep=3, warmup_ms=1)
    out = enceladus.zeros(4096)
    second[_grid](x, out, 4096)
    # The saved result maps back to the matching entry of the new list, pre-hook included.
    assert calls == [chosen.kwargs["BLOCK"]]
    np.testing.assert_array_equal(out.numpy(), x.numpy())


def _rebuild_configs(configs, named):
    """Returns new `Config` objects, as many Triton prune functions do."""
    return [enceladus.Config(c.kwargs, num_warps=8) for c in configs]


def test_saved_result_of_rebuilt_configs_is_reused(monkeypatch):
    prune = {"early_config_prune": _rebuild_configs}
    x = enceladus.randn(4096)
    first = Autotuner(_accumulate, CONFIGS[1:], key=["n"], prune_configs_by=prune, rep=3,
                      warmup_ms=1)  # fmt: skip
    first[_grid](x, enceladus.zeros(4096), 4096)
    chosen = next(iter(first.best.values()))

    monkeypatch.setattr("enceladus.testing.do_bench", _no_bench)
    second = Autotuner(_accumulate, CONFIGS[1:], key=["n"], prune_configs_by=prune, rep=3,
                       warmup_ms=1)  # fmt: skip
    out = enceladus.zeros(4096)
    second[_grid](x, out, 4096)
    assert second.config_for(x, out, 4096)._identity() == chosen._identity()
    np.testing.assert_array_equal(out.numpy(), x.numpy())


def test_saved_result_missing_from_configs_is_retuned():
    x = enceladus.randn(4096)
    first = Autotuner(_accumulate, CONFIGS[1:2], key=["n"], rep=3, warmup_ms=1)
    first[_grid](x, enceladus.zeros(4096), 4096)
    # The config list changed, so the saved BLOCK=256 result is stale and must not run.
    second = Autotuner(_accumulate, CONFIGS[2:], key=["n"], rep=3, warmup_ms=1)
    out = enceladus.zeros(4096)
    second[_grid](x, out, 4096)
    assert second.config_for(x, out, 4096).kwargs["BLOCK"] == 1024


def test_heuristics_compute_constexprs():
    block = {"BLOCK": lambda a: enceladus.next_power_of_2(a["n"])}
    kernel = enceladus.heuristics(block)(_accumulate)
    x, out = np.arange(300, dtype=np.float32), np.zeros(300, np.float32)
    kernel[(1,)](x, out, 300)
    np.testing.assert_array_equal(out, x)


@pytest.mark.parametrize(
    ("dtype", "half"),
    [(np.float32, False), (np.dtype("float32"), False), (tl.float32, False), ("float32", False),
     (np.float16, True), (tl.bfloat16, True), (ml_dtypes.bfloat16, True), ("bfloat16", True)],
)  # fmt: skip
def test_matmul_configs_exclude_spill_cliffs_per_dtype(dtype, half):
    strips = {(c.kwargs["BM"] // c.dot_warps[0], c.kwargs["BN"] // c.dot_warps[1])
              for c in matmul_configs(dtype)}  # fmt: skip
    assert ((32, 32) in strips) == half  # 32x32 per SIMD group spills in float32 only
    assert not strips & {(64, 32), (16, 128)}


@pytest.mark.parametrize("dtype", [np.float64, tl.int32, "int8"])
def test_matmul_configs_reject_unsupported_dtypes(dtype):
    with pytest.raises(ValueError, match="supports float32, float16, and bfloat16"):
        matmul_configs(dtype)
