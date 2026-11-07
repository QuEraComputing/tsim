"""Single-lane Python driver over the numba kernels, and a dense reference."""
import numpy as np
import tsim
import tsim.statevector.program as program
from tsim.statevector.pbc import MEASUREMENT, ROTATION, anticommute
from tsim.statevector.schedule import _ancestors
from tsim.statevector.conditioned.plan import step_plan
from tsim.statevector.conditioned import lane as L
from tsim.statevector.conditioned.through import prepare_through, finish_through

C = "/home/rafaelha/quera/tsim-conditioned/experiments/conditioned/circuits/"
CASES = {
    "cultivation_d3": C + "clifft-bench/msc_d3_inject_cultivate_p1e-3.stim",
    "cultivation_d5": C + "clifft-bench/msc_d5_inject_cultivate_p1e-3.stim",
    "cultivation_d7": C + "symft/MSC_circuit_d7_p0.0005.stim",
    "end2end_d5": C + "clifft-paper/msc_d5_end2end_p1e-3.stim",
    "distillation": C + "clifft-bench/distillation.stim",
    "coherent_d3_r3": C + "clifft-bench/coherent_d3_r3.stim",
    "blp": "/home/rafaelha/quera/circs/tricycle_factories/blp_27_3_3.stim",
}
for k in (6, 9, 12, 15, 18, 21, 24, 27):
    CASES[f"tricycle_{k}"] = f"/home/rafaelha/quera/circs/tricycle_factories/tricycle_{k}_3_{3 if k in (21, 27) else 2}.stim"


def capture(name):
    circuit = tsim.Circuit(open(CASES[name]).read())
    got = []
    orig = program.read_pauli_ops
    def cap(*a, **k):
        o = orig(*a, **k); got.append(o); return o
    program.read_pauli_ops = cap
    try:
        sampler = circuit.compile_detector_sampler(seed=0)
    finally:
        program.read_pauli_ops = orig
    return circuit, sampler, got


def packed(ops):
    n = ops.num_lines; W = (n + 63) // 64
    def pack(b):
        out = np.zeros((len(b), W), dtype=np.uint64)
        for j in range(len(b)):
            for q in np.flatnonzero(b[j]):
                out[j, q >> 6] |= np.uint64(1) << np.uint64(q & 63)
        return out
    ox, oz = pack(ops.x), pack(ops.z)
    oe = (np.count_nonzero(ops.x & ops.z, axis=1) % 4).astype(np.int64)
    return n, W, ox, oz, oe


def concrete_signs(ops, f):
    s = ops.sign[:, 0].astype(np.int64)
    if ops.num_f:
        s = s ^ ((ops.sign[:, 1:].astype(np.int64) @ np.asarray(f, np.int64)) & 1)
    return s


def run_lane(ops, plan, f, rng, max_width=26, forced=None):
    """Returns list of (op, p1, raw bit, width after) for one shot."""
    n, W, ox, oz, oe = packed(ops)
    s = concrete_signs(ops, f)
    theta = np.where(s & 1, -ops.angle, ops.angle)
    frame, active, pos, amps, cache, cver = L.new_state(n, W, ops.num_ops)
    v = np.empty(2 * W + 1, dtype=np.uint64)
    trace = []
    for step in range(plan.num_steps):
        for r in plan.applied(step):
            amps = L.rotate(frame, active, pos, amps, cache, cver, int(r), ox, oz, oe, float(theta[r]), W, max_width)
        m = int(plan.measure[step]); layer = plan.layer(step)
        if len(layer):
            p1, keep, phi, kv = prepare_through(frame, active, pos, amps, cache, cver, ox, oz, oe, layer, theta[layer].copy(), m, W, max_width)
            if len(phi) == 0:
                bit = int(round(p1))
            else:
                bit = int(rng.random() < p1) if forced is None else forced[m]
                amps = finish_through(active, pos, keep, phi, kv, bit)
        else:
            kind, p1, d = L.prepare_measure(frame, active, pos, amps, cache, cver, m, ox, oz, oe, W, v)
            if p1 < 1e-12 or p1 > 1 - 1e-12:
                bit = int(round(p1))
            else:
                bit = int(rng.random() < p1) if forced is None else forced[m]
            amps = L.finish_measure(frame, active, pos, amps, kind, d, bit, v, W)
        amps = L.compress(frame, active, pos, amps, W)
        trace.append((m, p1, bit, L.width_of(amps)))
    return trace


def dense_check(ops, f, trace):
    """Follow the trace on a dense state; return the largest |p1 - p1_dense|."""
    n = ops.num_lines
    s = concrete_signs(ops, f)
    anti = anticommute(ops.x, ops.z, ops.x, ops.z)
    anc = _ancestors(np.triu(anti, 1))
    applied = np.zeros(ops.num_ops, bool)
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    idx = np.arange(2 ** n)
    def pauli(j, v):
        xm = sum(1 << int(q) for q in np.flatnonzero(ops.x[j])); zm = sum(1 << int(q) for q in np.flatnonzero(ops.z[j]))
        ny = int(np.count_nonzero(ops.x[j] & ops.z[j]))
        par = np.array([bin(int(i) & zm).count("1") & 1 for i in idx]) if zm else np.zeros(len(idx), int)
        out = np.empty_like(v); out[idx ^ xm] = (1j ** ny) * (1 - 2 * par) * v
        return out
    worst = 0.0
    for m, p1, bit, _ in trace:
        for r in np.flatnonzero(anc[m] & ~applied):
            if ops.kind[r] == ROTATION:
                th = -ops.angle[r] if s[r] & 1 else ops.angle[r]
                psi = np.cos(th) * psi - 1j * np.sin(th) * pauli(r, psi); applied[r] = True
        mp = pauli(m, psi)
        e = np.vdot(psi, mp).real
        worst = max(worst, abs((1 - e) / 2 - p1))
        lam = 1 if bit == 0 else -1
        psi = (psi + lam * mp) / 2; psi /= np.linalg.norm(psi); applied[m] = True
    return worst
