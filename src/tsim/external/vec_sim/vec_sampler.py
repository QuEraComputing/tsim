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
from typing import Literal, Union, overload

import numpy as np
import stim

from tsim.external.vec_sim import VecSim
from tsim.parse import parse_parametric_tag


class VecSampler:
    def __init__(
        self, stim_circuit: stim.Circuit, sweep_bit_randomization: bool = False
    ):
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

    def state_vector(self) -> np.ndarray:
        return sample_circuit_with_vec_sim_return_data(
            self.circuit,
            self.sweep_bit_randomization,
            return_state_vector=True,
        )


@overload
def sample_circuit_with_vec_sim_return_data(
    circuit: stim.Circuit,
    sweep_bit_randomization: bool,
    return_state_vector: Literal[False] = False,
) -> tuple[list[bool], list[bool], list[bool]]: ...


@overload
def sample_circuit_with_vec_sim_return_data(
    circuit: stim.Circuit,
    sweep_bit_randomization: bool,
    return_state_vector: Literal[True],
) -> np.ndarray: ...


def sample_circuit_with_vec_sim_return_data(
    circuit: stim.Circuit,
    sweep_bit_randomization: bool,
    return_state_vector: bool = False,
) -> Union[tuple[list[bool], list[bool], list[bool]], np.ndarray]:
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
        assert not isinstance(inst, stim.CircuitRepeatBlock)
        if inst.name == "S" and inst.tag == "T":
            for q in inst.targets_copy():
                sim.do_t(q.qubit_value)
        elif inst.name == "S_DAG" and inst.tag == "T":
            for q in inst.targets_copy():
                sim.do_t_dag(q.qubit_value)
        elif inst.name == "I" and inst.tag:
            result = parse_parametric_tag(inst.tag)
            if result is not None:
                gate_name, params = result
                for q in inst.targets_copy():
                    if gate_name == "R_Z":
                        sim.do_r_z(q.qubit_value, float(params["theta"]))
                    elif gate_name == "R_X":
                        sim.do_r_x(q.qubit_value, float(params["theta"]))
                    elif gate_name == "R_Y":
                        sim.do_r_y(q.qubit_value, float(params["theta"]))
                    elif gate_name == "U3":
                        sim.do_u3(
                            q.qubit_value,
                            theta=float(params["theta"]),
                            phi=float(params["phi"]),
                            lam=float(params["lambda"]),
                        )
        else:
            sim.do_stim_instruction(
                inst,
                sweep_bits=sweep_bits,
                out_measurements=measurements,
                out_detectors=detectors,
                out_observables=observables,
            )
    if return_state_vector:
        return sim.normalized_state()
    return measurements, detectors, observables
