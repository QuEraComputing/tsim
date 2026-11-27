import numpy as np
import pytest

from tsim.statevector.metal.runtime import lanes_in_order


def _probed(tid, lane, group):
    return (tid | (lane << 12) | (group << 20)).astype(np.uint32)


@pytest.mark.parametrize("threads", [128, 1024])
def test_lanes_in_thread_order_pass_the_probe(threads):
    tid = np.tile(np.arange(threads), 3)
    assert lanes_in_order(_probed(tid, tid % 32, tid // 32), threads)


@pytest.mark.parametrize(
    "lane_of",
    [
        lambda tid: (tid * 5) % 32,  # lanes permuted within each SIMD group
        lambda tid: (tid % 32) ^ ((tid // 32) & 1),  # swapped in every other group
    ],
)
def test_lanes_out_of_order_fail_the_probe(lane_of):
    tid = np.tile(np.arange(256), 2)
    assert not lanes_in_order(_probed(tid, lane_of(tid), tid // 32), 256)


def test_simd_groups_out_of_order_fail_the_probe():
    tid = np.arange(256)
    assert not lanes_in_order(_probed(tid, tid % 32, 7 - tid // 32), 256)
