"""Determined measurements of a published cultivation circuit with T gates.

The circuit and its source are in ``test/data/determined``.
"""

from pathlib import Path

import pytest
import stim

from tsim.circuit import Circuit
from tsim.determined import (
    _Declared,
    count_determined_measurements,
    missing_detectors,
    nondeterministic_detectors,
)
from tsim.flows import flow_generators, has_all_flows
from tsim.utils.gf2_span import Span

CIRCUIT = Path(__file__).parent.parent / "data" / "determined"
CIRCUIT = CIRCUIT / "chan_d5a19_uncorrected_T.stim"


@pytest.mark.parametrize("shots", [None, 2000])
def test_chan_uncorrected_d5_circuit_has_random_detectors(shots):
    # NOTE: Chan et al. sim/d5-correction.md: "Bell growth leaves seven
    # first-round check results that control a Pauli frame", so swapping S for
    # T needs feedforward first. Without it, four detectors are random, and
    # D91^D97 and D91^D102 are the determined parities of them.
    circuit = Circuit(CIRCUIT.read_text())
    assert circuit.num_measurements == 112
    assert count_determined_measurements(circuit, shots=shots, seed=0) == 106
    assert nondeterministic_detectors(circuit, shots=shots, seed=0) == [
        stim.target_relative_detector_id(i) for i in (69, 91, 97, 102)
    ]
    missing = missing_detectors(circuit, shots=shots, seed=0)
    assert len(missing) == 2
    words = _Declared((circuit + missing).stim_circuit).words
    span = Span(circuit.num_measurements)
    span.insert(words[[107, 108]])
    assert not span.insert(words[[91, 91]] ^ words[[97, 102]]).any()


def test_chan_uncorrected_d5_circuit_has_flows_that_end_in_its_state():
    # NOTE: the circuit starts with resets, so no flow has an input. Its flows
    # of measurements alone are its determined parities. The others stabilize
    # the 42 qubits it ends with, which hold a T state: one stabilizer fewer
    # than a stabilizer state has.
    circuit = Circuit(CIRCUIT.read_text())
    flows = flow_generators(circuit)
    assert not any(flow.input_copy().weight for flow in flows)
    outputs = [flow.output_copy().weight > 0 for flow in flows]
    assert outputs.count(False) == count_determined_measurements(circuit) == 106
    assert outputs.count(True) == circuit.num_qubits - 1 == 41
    assert has_all_flows(circuit, flows)
