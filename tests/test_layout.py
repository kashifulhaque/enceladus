"""Tests for the bit-linear layout algebra."""

import numpy as np
import pytest

from tegula.compiler import layout as L


def owners(lay: L.BitLayout) -> np.ndarray:
    """Returns coordinates held by owner threads as an [n, rank] array."""
    coords = L.materialize(lay)
    lm, wm = L.owner_mask(lay)
    keep = [
        coords[w, lane]
        for w in range(lay.num_warps)
        for lane in range(32)
        if not (lane & lm) and not (w & wm)
    ]
    return np.concatenate(keep).reshape(-1, lay.rank)


@pytest.mark.parametrize(
    "shape,num_warps,elem_bytes,order",
    [
        ((4096,), 4, 4, None),
        ((64,), 4, 2, None),  # fewer elements than threads
        ((1,), 1, 4, None),
        ((32, 128), 8, 2, None),
        ((64, 16), 4, 4, (0, 1)),  # column-major order
        ((2, 4, 256), 2, 4, None),
    ],
)
def test_blocked_covers_every_element_once(shape, num_warps, elem_bytes, order):
    lay = L.blocked(shape, num_warps, elem_bytes, order)
    got = owners(lay)
    assert len(got) == np.prod(shape)
    assert len({tuple(c) for c in got}) == np.prod(shape)


def test_blocked_1d_is_interleaved_16_byte_vectors():
    lay = L.blocked((4096,), 4, 4)
    coords = L.materialize(lay)[..., 0]  # [warp, lane, reg]
    threads, vec = 128, 4
    for w in (0, 3):
        for lane in (0, 5, 31):
            t = w * 32 + lane
            expect = [t * vec + (r % vec) + (r // vec) * threads * vec for r in range(lay.num_regs)]
            assert coords[w, lane].tolist() == expect


def test_simd_acc_matches_fragment_formula():
    lay = L.simd_acc(64, 32, 4, 1)
    coords = L.materialize(lay)
    tm, tn = 2, 4  # fragments per SIMD group
    for w in range(4):
        for lane in range(32):
            row = (lane >> 2 & 4) + (lane >> 1 & 3)
            col = (lane >> 1 & 4) + (lane << 1 & 2)
            for i in range(tm):
                for j in range(tn):
                    for e in range(2):
                        r = e + 2 * j + 2 * tn * i
                        assert tuple(coords[w, lane, r]) == (w * 16 + i * 8 + row, j * 8 + col + e)


def test_reduce_then_broadcast_back_needs_no_exchange():
    for lay in (L.blocked((16, 256), 4, 4), L.simd_acc(64, 64, 4, 1), L.simd_acc(64, 64, 2, 2)):
        for dim in (0, 1):
            small = L.expand(L.slice_layout(lay, dim), dim)
            m = L.reg_map(small, lay)
            assert m is not None
            # Every register maps to the register holding the same reduced coordinate.
            sc, bc = L.materialize(small), L.materialize(lay)
            proj = bc.copy()
            proj[..., dim] = 0
            np.testing.assert_array_equal(sc[:, :, list(m)], proj)


def test_reg_map_detects_cross_thread_moves():
    a = L.blocked((32, 32), 4, 4)
    assert L.reg_map(a, a) == tuple(range(a.num_regs))
    assert L.reg_map(a, L.blocked((32, 32), 4, 4, order=(0, 1))) is None
    assert L.reg_map(L.permute(a, (1, 0)), L.blocked((32, 32), 4, 4)) is None


def test_permute_and_reshape_track_elements():
    a = L.blocked((8, 64), 2, 4)
    ref = np.arange(8 * 64).reshape(8, 64)

    def values(lay, arr):
        c = L.materialize(lay)
        return arr[tuple(c[..., d] for d in range(lay.rank))]

    np.testing.assert_array_equal(values(L.permute(a, (1, 0)), ref.T), values(a, ref))
    np.testing.assert_array_equal(values(L.reshape(a, (4, 2, 64)), ref.reshape(4, 2, 64)),
                                  values(a, ref))
    np.testing.assert_array_equal(values(L.reshape(a, (512,)), ref.reshape(-1)), values(a, ref))


def test_thread_terms_reproduce_coordinates():
    for lay in (L.blocked((16, 256), 4, 2), L.simd_acc(32, 64, 2, 2)):
        coords = L.materialize(lay)
        for w in range(lay.num_warps):
            for lane in range(32):
                c = list(L.coords_of(lay, 0))
                for t in L.thread_terms(lay):
                    idx = lane if t.source == "lane" else w
                    c[t.dim] += ((idx >> t.src_shift) & ((1 << t.width) - 1)) << t.dst_shift
                assert tuple(coords[w, lane, 0]) == tuple(c)


def test_invalid_layouts_are_rejected():
    with pytest.raises(ValueError, match="power of two"):
        L.blocked((100,), 4, 4)
    with pytest.raises(ValueError, match="more than once"):
        L.BitLayout((4,), ((1,), (1,)), ((0,),) * 5, ())
