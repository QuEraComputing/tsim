"""
Statevector sampler for stim circuits.

Based on code from:
Gidney, C., Jones, C., & Shutty, N. (2024). "Magic state cultivation: growing
T states as cheap as CNOT gates." arXiv:2409.17595


Code dataset: https://doi.org/10.5281/zenodo.13777072
Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

Modifications:
- Removed sinter dependency, modifyied T replacement logic.
"""

import random
import numpy as np
import stim

from tsim.external.vec_sim import VecSim


class VecSampler:
    def __init__(self, stim_circuit: stim.Circuit, sweep_bit_randomization: bool):
        self.circuit = stim_circuit
        self.sweep_bit_randomization = sweep_bit_randomization

    def sample(self, shots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the circuit and return the measurements, detectors, and observables."""
        measurements, detectors, observables = [], [], []
        for _ in range(shots):
            m, d, o = sample_circuit_with_vec_sim_return_data(
                self.circuit,
                self.sweep_bit_randomization,
            )
            measurements.append(m)
            detectors.append(d)
            observables.append(o)
        return (
            np.array(measurements, dtype=np.uint8),
            np.array(detectors, dtype=np.uint8),
            np.array(observables, dtype=np.uint8),
        )


def sample_circuit_with_vec_sim_return_data(
    circuit: stim.Circuit, sweep_bit_randomization: bool
) -> tuple[list[bool], list[bool], list[bool]]:
    sim = VecSim()
    measurements = []
    detectors = []
    observables = []
    sweep_bits = {
        b: sweep_bit_randomization and random.random() < 0.5
        for b in range(circuit.num_sweep_bits)
    }
    for q in range(circuit.num_qubits):
        sim.do_qalloc_z(q)
    for inst in circuit:
        if inst.name == "S" and inst.tag == "T":
            for q in inst.targets_copy():
                sim.do_t(q.qubit_value)
        elif inst.name == "S_DAG" and inst.tag == "T":
            for q in inst.targets_copy():
                sim.do_t_dag(q.qubit_value)
        else:
            sim.do_stim_instruction(
                inst,
                sweep_bits=sweep_bits,
                out_measurements=measurements,
                out_detectors=detectors,
                out_observables=observables,
            )
    return measurements, detectors, observables
