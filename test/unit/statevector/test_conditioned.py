"""Tests of ``strategy="conditioned_statevector"``."""

import itertools
from test.helpers import conditioned as exact
from test.helpers.statevector import (
    CIRCUITS,
    assert_counts_match,
    assert_distributions_close,
    component_ops,
    f_patterns,
    ops_distribution,
)

import numba
import numpy as np
import pytest

import tsim.statevector.conditioned.component as component_module
from tsim.circuit import Circuit
from tsim.core.types import SamplingStrategy
from tsim.statevector.conditioned.component import _packed, conditioned_component
from tsim.statevector.conditioned.gauss import gauss_sum
from tsim.statevector.conditioned.lane import TooWide, new_state
from tsim.statevector.conditioned.plan import is_t_like, step_plan
from tsim.statevector.conditioned.runner import reference_prefix
from tsim.statevector.conditioned.through import prepare_through, root_outcome
from tsim.statevector.pbc import MEASUREMENT, ROTATION, ComponentLayout, PauliOps

# A phase polynomial on |+++> computed onto an ancilla with T gates, the
# ancilla's Z measurement and an X readout: several layers to measure through.
CCZ_CHECK = """
    RX 0 1 2
    R 3
    T 0 1 2
    CNOT 0 3 1 3
    T_DAG 3
    CNOT 2 3
    T 3
    CNOT 1 3
    T_DAG 3
    CNOT 2 3
    T 3
    CNOT 0 3
    DEPOLARIZE1(0.02) 0 1 2
    M 3
    MX 0 1 2
    DETECTOR rec[-4]
    OBSERVABLE_INCLUDE(0) rec[-3]
    OBSERVABLE_INCLUDE(1) rec[-2]
    OBSERVABLE_INCLUDE(2) rec[-1]
"""
SAMPLER_CIRCUITS = {**CIRCUITS, "ccz_check": CCZ_CHECK}


def test_gauss_sum_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(500):
        d = int(rng.integers(0, 8))
        c = int(rng.integers(4))
        lin = rng.integers(0, 4, d)
        quad = np.triu(rng.integers(0, 2, (d, d)), 1)
        brute = 0j
        for t in itertools.product((0, 1), repeat=d):
            t = np.array(t, dtype=np.int64)
            brute += 1j ** int((c + lin @ t + 2 * (t @ quad @ t)) % 4)
        sym = quad | quad.T
        masks = np.array(
            [sum(int(sym[j, k]) << k for k in range(d)) for j in range(d)],
            dtype=np.uint64,
        )
        assert abs(gauss_sum(c, lin.astype(np.int64).copy(), masks) - brute) < 1e-9


@pytest.mark.parametrize("index", range(len(component_ops())))
def test_key_leaves_match_dense_distribution(index):
    ops = component_ops()[index]
    layout = ComponentLayout(tuple(range(ops.num_f)), {}, tuple(range(ops.num_outputs)))
    component = conditioned_component(ops, layout)
    reference = ops
    ops = component.ops  # the component may reorder its ops
    for f in f_patterns(ops.num_f, limit=8):
        f = np.asarray(f, dtype=np.int64)
        signs = (
            ops.sign[:, 0].astype(np.int64) + ops.sign[:, 1:].astype(np.int64) @ f
        ) & 1
        flipped = ((ops.sign[:, 1:].astype(np.int64) @ f) & 1).astype(bool)
        flipped &= ops.kind == ROTATION
        leaves, probs = exact.key_leaves(component, flipped)
        assert abs(probs.sum() - 1) < 1e-9
        got: dict[tuple[int, ...], float] = {}
        for raw, p in zip(leaves, probs, strict=True):
            key = []
            for slot in range(ops.num_outputs):
                (j,) = np.flatnonzero(ops.slot == slot)
                key.append(int(raw[slot]) ^ int(signs[j]))
            got[tuple(key)] = got.get(tuple(key), 0.0) + p
        assert_distributions_close(
            got, ops_distribution(reference, f.tolist()), atol=1e-9
        )


def test_root_outcome_matches_measure_through():
    ops = next(o for o in component_ops() if (o.kind == ROTATION).sum() > 2)
    layout = ComponentLayout(tuple(range(ops.num_f)), {}, tuple(range(ops.num_outputs)))
    component = conditioned_component(ops, layout)
    ops = component.ops  # the component may reorder its ops
    plan, n = component.plan, ops.num_lines
    words = -(-n // 64)
    ref = np.where(ops.sign[:, 0] & 1, -ops.angle, ops.angle)
    rng = np.random.default_rng(0)
    for step in np.flatnonzero(component.root_amp):
        layer = plan.layer(int(step))
        for _ in range(20):
            flipped = (rng.random(ops.num_ops) < 0.3) & (ops.kind == ROTATION)
            theta = np.where(flipped, -ref, ref)
            frame, active, pos, amps, cache, cache_ver = new_state(
                n, words, ops.num_ops
            )
            got = root_outcome(
                layer,
                flipped,
                ref,
                component.op_x,
                component.op_z,
                component.op_e,
                component.root_amp[step],
                words,
            )
            p1, _, phi, _ = prepare_through(
                frame,
                active,
                pos,
                amps,
                cache,
                cache_ver,
                component.op_x,
                component.op_z,
                component.op_e,
                layer,
                theta[layer].copy(),
                int(plan.measure[step]),
                words,
                26,
            )
            if got < 0:
                assert len(phi) > 0
            else:
                assert len(phi) == 0 and got == int(p1 > 0.5)


def test_step_plan_measures_every_measurement_once():
    for ops in component_ops():
        plan = step_plan(ops)
        assert sorted(plan.measure) == sorted(np.flatnonzero(ops.kind == 1))
        layered = set(plan.layer_ops) | set(plan.apply_ops)
        assert layered <= set(np.flatnonzero(ops.kind == ROTATION))
        for step in range(plan.num_steps):
            assert all(is_t_like(float(ops.angle[r])) for r in plan.layer(step))


@pytest.mark.parametrize("name", sorted(SAMPLER_CIRCUITS))
@pytest.mark.parametrize("postselect", [False, True])
def test_sampler_matches_active_statevector(name, postselect):
    circuit = Circuit(SAMPLER_CIRCUITS[name])
    mask = np.ones(circuit.num_detectors, bool) if postselect else None
    if mask is not None and not mask.any():
        mask = None
    samples = {}
    runs: tuple[tuple[SamplingStrategy, int], ...] = (
        ("active_statevector", 1),
        ("conditioned_statevector", 2),
    )
    for strategy, seed in runs:
        sampler = circuit.compile_detector_sampler(seed=seed, strategy=strategy)
        detectors, observables = sampler.sample(
            20_000, separate_observables=True, postselection_mask=mask
        )
        samples[strategy] = np.concatenate([detectors, observables], axis=1)
    a, b = samples["active_statevector"], samples["conditioned_statevector"]
    if postselect:
        assert abs(len(a) - len(b)) <= 5 * np.sqrt(20_000) + 3
    if a.shape[1] and len(a) and len(b):
        assert_counts_match(b[:, :12], a[:, :12])


def test_samples_do_not_depend_on_threads():
    circuit = Circuit(CCZ_CHECK)
    out = []
    config = numba.config
    limit = config.NUMBA_NUM_THREADS  # pyright: ignore[reportAttributeAccessIssue]
    for threads in (1, limit):
        numba.set_num_threads(threads)
        sampler = circuit.compile_detector_sampler(
            seed=7, strategy="conditioned_statevector"
        )
        out.append(sampler.sample(5000, batch_size=700))
    numba.set_num_threads(limit)
    assert np.array_equal(out[0], out[1])


def test_compress_frees_every_qubit_of_a_stabilizer_state():
    from tsim.statevector.conditioned import lane

    rng = np.random.default_rng(5)
    for _ in range(20):
        n, w = 8, int(rng.integers(1, 7))
        frame, active, pos, amps, _, _ = lane.new_state(n, 1, 1)
        for q in rng.choice(n, w, replace=False):
            amps = lane.promote(active, pos, amps, int(q), 26)
        qubits = active[:w].copy()
        for _ in range(30):
            code = int(rng.choice([lane.H, lane.S, lane.CX, lane.CZ]))
            a, b = (
                (int(q) for q in rng.choice(qubits, 2, replace=False))
                if w > 1
                else (int(qubits[0]), 0)
            )
            if code >= lane.CX and w == 1:
                continue
            lane.gate(frame, pos, amps, code, a, b, 1)
        before = amps.copy()
        amps = lane.compress(frame, active, pos, amps, 1)
        assert lane.width_of(amps) == 0
        assert abs(abs(amps[0]) - 1) < 1e-12 and np.linalg.norm(before) > 0


def test_compress_keeps_magic():
    from tsim.statevector.conditioned import lane

    frame, active, pos, amps, _, _ = lane.new_state(2, 1, 1)
    amps = lane.promote(active, pos, amps, 0, 26)
    amps = lane.promote(active, pos, amps, 1, 26)
    t = np.exp(1j * np.pi / 4)
    amps[:] = np.array([1, t, t, 1j]) / 2  # T|+> (x) T|+> up to CZ-free phases
    amps = lane.compress(frame, active, pos, amps, 1)
    assert lane.width_of(amps) == 2


def test_a_too_wide_reference_key_raises_the_clear_error(monkeypatch):
    sampler = Circuit(CCZ_CHECK).compile_detector_sampler(
        seed=1, strategy="conditioned_statevector"
    )
    # NOTE: the reference tree is built at the first sample, under this limit.
    monkeypatch.setattr(component_module, "MAX_WIDTH", 1)
    with pytest.raises(ValueError, match="active qubits"):
        sampler.sample(100)


def test_a_first_measure_through_wider_than_the_limit_compiles(monkeypatch):
    # NOTE: exp(-i pi/8 X_j) on every qubit, then Y...Y, which the first step
    # measures through on a register of all the qubits.
    n = 6
    num_ops = 2 * n + 1
    x = np.zeros((num_ops, n), bool)
    z = np.zeros((num_ops, n), bool)
    x[np.arange(n), np.arange(n)] = True
    x[n] = z[n] = True
    z[n + 1 + np.arange(n), np.arange(n)] = True
    ops = PauliOps(
        kind=np.array([ROTATION] * n + [MEASUREMENT] * (n + 1)),
        x=x,
        z=z,
        angle=np.array([np.pi / 8] * n + [0.0] * (n + 1)),
        sign=np.zeros((num_ops, 1), np.uint8),
        slot=np.array([-1] * n + list(range(n + 1))),
    )
    plan = step_plan(ops)
    with pytest.raises(TooWide):
        reference_prefix(
            n,
            *_packed(ops),
            ops.angle,
            ops.sign[:, 0].astype(np.int64),
            plan.measure,
            plan.apply_start,
            plan.layer_start,
            plan.layer_ops,
            n - 2,
        )
    monkeypatch.setattr(component_module, "MAX_WIDTH", n - 2)
    layout = ComponentLayout((), {}, tuple(range(n + 1)))
    component = conditioned_component(ops, layout)
    assert not component.root_amp.any()
