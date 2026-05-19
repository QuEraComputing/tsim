import time

import stim

import tsim


def test_classical_detector_sampling_time_per_shot():
    """At p=1e-6 the detector sampler should produce shots faster than 5e-8 s/shot."""
    d = 7
    p = 1e-6
    shots = 10_000_000

    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
    )
    tc = tsim.Circuit(str(stim_circuit))
    sampler = tc.compile_detector_sampler()

    # Warm up JIT compilation
    sampler.sample(shots=shots)

    t0 = time.perf_counter()
    sampler.sample(shots=shots)
    sample_time_s = time.perf_counter() - t0
    time_per_shot = sample_time_s / shots

    assert (
        time_per_shot < 5e-8
    ), f"Time per shot {time_per_shot * 1e6:.4f} us exceeds 5e-8 s budget"
