"""Determined measurements of a published cultivation circuit with T gates.

The circuit and its source are in ``test/data/determined``.
"""

from pathlib import Path

import pytest
import stim

from tsim.circuit import Circuit
from tsim.determined import (
    count_determined_measurements,
    missing_detectors,
    nondeterministic_detectors,
)

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
    assert str(missing) == str(stim.Circuit("""
            DETECTOR rec[-46] rec[-45] rec[-16] rec[-10]
            DETECTOR rec[-47] rec[-45] rec[-39] rec[-16] rec[-5]
            """))
