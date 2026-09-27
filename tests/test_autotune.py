"""Tests for `@enceladus.autotune` and `@enceladus.heuristics`."""

from concurrent.futures import ThreadPoolExecutor

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


def test_saved_result_is_retuned_after_a_compiler_change(monkeypatch):
    x = enceladus.randn(4096)
    first = Autotuner(_accumulate, CONFIGS[1:], key=["n"], rep=3, warmup_ms=1)
    first[_grid](x, enceladus.zeros(4096), 4096)

    monkeypatch.setattr("enceladus.runtime.cache.compiler_hash", lambda: "a different compiler")
    monkeypatch.setattr("enceladus.testing.do_bench", _no_bench)
    second = Autotuner(_accumulate, CONFIGS[1:], key=["n"], rep=3, warmup_ms=1)
    with pytest.raises(RuntimeError, match="re-benchmarked"):
        second[_grid](x, enceladus.zeros(4096), 4096)


def test_saved_result_missing_from_configs_is_retuned():
    x = enceladus.randn(4096)
    first = Autotuner(_accumulate, CONFIGS[1:2], key=["n"], rep=3, warmup_ms=1)
    first[_grid](x, enceladus.zeros(4096), 4096)
    # The config list changed, so the saved BLOCK=256 result is stale and must not run.
    second = Autotuner(_accumulate, CONFIGS[2:], key=["n"], rep=3, warmup_ms=1)
    out = enceladus.zeros(4096)
    second[_grid](x, out, 4096)
    assert second.config_for(x, out, 4096).kwargs["BLOCK"] == 1024


@enceladus.jit
def _scale_copy(x_ptr, out_ptr, n, alpha=2.0, BLOCK: tl.constexpr = 256):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * alpha, mask=mask)


def test_key_argument_can_take_its_default():
    tuned = Autotuner(_scale_copy, CONFIGS[1:], key=["n", "alpha"], rep=3, warmup_ms=1)
    x, out = enceladus.randn(4096), enceladus.zeros(4096)
    tuned[_grid](x, out, 4096)
    np.testing.assert_array_equal(out.numpy(), x.numpy() * 2)
    assert tuned.config_for(x, out, 4096) is tuned.config_for(x, out, 4096, 2.0) is not None


_X32 = np.zeros(8, np.float32)


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ((_X32, _X32, np.int64(1 << 20)), (_X32, _X32, np.int64(1000)), False),
        ((_X32, _X32, np.int64(1000)), (_X32, _X32, 1000), True),
        ((_X32, _X32, np.array(1 << 20)), (_X32, _X32, np.array(1000)), False),
        ((_X32, _X32, 8), (_X32, _X32.astype(np.float16), 8), False),
        ((_X32, _X32, 8), (enceladus.from_numpy(_X32), _X32, 8), True),
    ],
    ids=["numpy_scalar_values", "numpy_scalar_is_int", "0d_array_values", "dtype",
         "tensor_and_array"],
)  # fmt: skip
def test_key_distinguishes_values_and_normalizes_dtypes(a, b, same):
    tuned = Autotuner(_accumulate, CONFIGS, key=["n"])
    assert (tuned._key(dict(zip(tuned.arg_names, a, strict=False))) ==
            tuned._key(dict(zip(tuned.arg_names, b, strict=False)))) == same  # fmt: skip


def test_restore_value_resets_before_every_benchmark_run():
    seen = set()
    config = enceladus.Config({"BLOCK": 256},
                              pre_hook=lambda a: seen.add(float(a["out_ptr"]._view()[0])))
    tuned = Autotuner(_accumulate, [config], key=["n"], restore_value=["out_ptr"], rep=3,
                      warmup_ms=1)  # fmt: skip
    x, out = enceladus.ones(4096), enceladus.full(4096, 5.0)
    tuned[_grid](x, out, 4096)
    # Every run, benchmarked or real, starts from the original values.
    assert seen == {5.0}
    np.testing.assert_array_equal(out.numpy(), 6.0)


# Scripted GPU times in ms by BLOCK, for the first phase (every config) and the second
# (interleaved re-timing of the fastest few), and the config that tuning must pick.
TWO_PHASE_CASES = {
    # Phase 1 favors 512 by noise; phase 2 finds 1024 10% faster.
    "second_phase_decides": ({256: 1.10, 512: 1.00, 1024: 1.05},
                             {256: 1.10, 512: 1.00, 1024: 0.90}, 1024),
    # In phase 2, 256 is within the 1% tolerance of 512, and it comes first in the list.
    "tolerance_prefers_earlier": ({256: 1.10, 512: 1.00, 1024: 1.05},
                                  {256: 1.005, 512: 1.00, 1024: 1.20}, 256),
}  # fmt: skip


@pytest.mark.parametrize("case", list(TWO_PHASE_CASES))
def test_second_phase_and_tolerance_pick_the_winner(monkeypatch, case):
    phase1, phase2, expected = TWO_PHASE_CASES[case]
    running = []
    configs = [enceladus.Config({"BLOCK": b}, pre_hook=lambda a, b=b: running.append(b))
               for b in (256, 512, 1024)]  # fmt: skip

    def fake_do_bench(fn, warmup_ms=50.0, rep=20, return_mode="median"):
        fn()
        enceladus.synchronize()
        block = running[-1]
        if return_mode == "all":  # the second phase drops the first sample of a batch
            return [100.0] + [phase2[block]] * (rep - 1)
        return phase1[block]

    monkeypatch.setattr("enceladus.testing.do_bench", fake_do_bench)
    tuned = Autotuner(_accumulate, configs, key=["n"], reset_to_zero=["out_ptr"], rep=8)
    x, out = enceladus.randn(4096), enceladus.zeros(4096)
    tuned[_grid](x, out, 4096)
    assert tuned.config_for(x, out, 4096).kwargs["BLOCK"] == expected
    np.testing.assert_array_equal(out.numpy(), x.numpy())


@enceladus.jit
def _strided_copy(x_ptr, out_ptr, n, stride, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs * stride, mask=mask), mask=mask)


def test_tuning_benchmarks_the_specialization_that_the_launch_runs():
    # Views that start 4 bytes into their arrays: neither pointer is 16-byte aligned, and
    # `x` has a stride of 2 that the kernel applies itself.
    x = np.arange(2 * 4096, dtype=np.float32)[1::2]
    out = np.zeros(4096 + 1, np.float32)[1:]
    tuned = Autotuner(_strided_copy, CONFIGS[1:], key=["n"], rep=3, warmup_ms=1)
    tuned[_grid](x, out, 4096, 2)
    np.testing.assert_array_equal(out, x)
    # Tuning compiled every config, and the launch compiled or reused the chosen one. All
    # of them specialize both pointers as unaligned, as the launch does.
    f32 = np.dtype(np.float32)
    assert len(_strided_copy._compiled) == 2
    assert {k[0][:2] for k in _strided_copy._compiled} == {((f32, False), (f32, False))}


def test_concurrent_saves_keep_every_result():
    tuners = [Autotuner(_accumulate, CONFIGS, key=["n"]) for _ in range(16)]
    for i, t in enumerate(tuners):
        t.best[f"key{i}"] = CONFIGS[i % len(CONFIGS)]
    with ThreadPoolExecutor(len(tuners)) as pool:
        list(pool.map(Autotuner._save, tuners))
    fresh = Autotuner(_accumulate, CONFIGS, key=["n"])
    fresh._load()
    assert sorted(fresh._saved) == sorted(f"key{i}" for i in range(16))


def test_explain_and_warmup_use_the_tuned_config():
    configs = [enceladus.Config({"BLOCK": 256}, num_warps=2),
               enceladus.Config({"BLOCK": 1024}, num_warps=8)]  # fmt: skip
    tuned = Autotuner(_accumulate, configs, key=["n"], rep=3, warmup_ms=1)
    x, out = enceladus.randn(1 << 16), enceladus.zeros(1 << 16)
    # Before tuning, they use the first candidate and don't benchmark.
    assert "num_warps=2 " in tuned.explain(x, out, 1 << 16)
    assert not tuned.best
    tuned[_grid](x, out, 1 << 16)
    best = tuned.config_for(x, out, 1 << 16)
    assert f"num_warps={best.num_warps} " in tuned.explain(x, out, 1 << 16)
    assert tuned.warmup(x, out, 1 << 16).num_warps == best.num_warps


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
