"""Determined measurements of published cultivation circuits with T gates.

The circuits and their sources are in ``test/data/determined``.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import stim

from tsim.circuit import Circuit
from tsim.determined import (
    _Declared,
    count_determined_measurements,
    missing_detectors,
    nondeterministic_detectors,
)
from tsim.utils.gf2_span import unpack

DATA = Path(__file__).parent.parent / "data" / "determined"


def _circuit(name: str) -> Circuit:
    return Circuit((DATA / f"{name}.stim").read_text())


def _detectors(*indices: int) -> list[stim.DemTarget]:
    return [stim.target_relative_detector_id(i) for i in indices]


def _healed(circuit: Circuit) -> list[set[int]]:
    """Return each missing detector as the random detectors whose XOR it is."""
    num_m = circuit.num_measurements
    declared = unpack(_Declared(circuit.stim_circuit).words, num_m).astype(np.int64)
    random = [target.val for target in nondeterministic_detectors(circuit)]
    result = []
    for instruction in missing_detectors(circuit):
        assert isinstance(instruction, stim.CircuitInstruction)
        parity = np.zeros(num_m, dtype=np.int64)
        for target in instruction.targets_copy():
            parity[num_m + target.value] ^= 1
        found = [
            set(subset)
            for size in (2, 3)
            for subset in combinations(random, size)
            if np.array_equal(declared[list(subset)].sum(axis=0) % 2, parity)
        ]
        result.append(found[0] if found else set())
    return result


@pytest.mark.parametrize(
    "name, num_detectors",
    [
        ("chan_d5a19_T", 107),
        ("chan_d5a19f13_T", 122),
        ("clifft_d5_T", 107),
        ("wan_clifft_d3_T", 20),
        ("wan_soft_d5_T", 107),
    ],
)
@pytest.mark.parametrize("shots", [None, 2000])
def test_published_t_circuits_have_every_detector_determined(
    name, num_detectors, shots
):
    # NOTE: Chan et al. (arXiv:2609.17706) and Clifft (arXiv:2604.27058) build
    # these from S proxies with feedforward and detector healing; Wan and
    # Zapirain (arXiv:2609.18922) use them as given. Each has one observable.
    circuit = _circuit(name)
    assert circuit.num_detectors == num_detectors
    assert count_determined_measurements(circuit, shots=shots, seed=0) == (
        num_detectors + 1
    )
    assert nondeterministic_detectors(circuit, shots=shots, seed=0) == []
    assert len(missing_detectors(circuit, shots=shots, seed=0)) == 0


def test_chan_uncorrected_d5_circuits_have_random_detectors():
    # NOTE: Chan et al. sim/d5-correction.md: "Bell growth leaves seven
    # first-round check results that control a Pauli frame", so swapping S for
    # T needs feedforward first.
    circuit = _circuit("chan_d5a19_uncorrected_T")
    assert count_determined_measurements(circuit) == 106
    assert nondeterministic_detectors(circuit) == _detectors(69, 91, 97, 102)
    assert _healed(circuit) == [{91, 97}, {91, 102}]
    flagged = _circuit("chan_d5a19f13_uncorrected_T")
    assert count_determined_measurements(flagged) == 121
    assert nondeterministic_detectors(flagged) == _detectors(71, 106, 112, 117)
    assert _healed(flagged) == [{106, 112}, {106, 117}]


def test_each_clifft_conversion_rule_heals_detectors():
    # NOTE: Clifft's convert_s_to_t.py: rule 1 swaps S for T, rule 2 flips six
    # T gates (Gidney et al.'s erratum), rules 3 and 4 add feedforward and heal
    # the detectors.
    proxy = _circuit("clifft_d5_sproxy")
    assert count_determined_measurements(proxy) == (
        proxy.stim_circuit.count_determined_measurements()
    )
    assert count_determined_measurements(proxy) == 108
    swapped = _circuit("clifft_d5_T_rule1")
    assert count_determined_measurements(swapped) == 105
    assert nondeterministic_detectors(swapped) == _detectors(69, 93, 95, 99, 101, 103)
    flipped = _circuit("clifft_d5_T_rule12")
    assert count_determined_measurements(flipped) == 106
    assert nondeterministic_detectors(flipped) == _detectors(69, 99, 101, 103)
    assert _healed(flipped) == [{99, 101}, {99, 103}]
    assert nondeterministic_detectors(_circuit("clifft_d5_T")) == []


def _feedforward_sources(circuit: stim.Circuit) -> list[int]:
    measured, sources = 0, set()
    for instruction in circuit.flattened():
        assert isinstance(instruction, stim.CircuitInstruction)
        if instruction.name in ("CX", "CZ"):
            for target in instruction.targets_copy():
                if target.is_measurement_record_target:
                    sources.add(measured + target.value)
        measured += instruction.num_measurements
    return sorted(sources)


def test_three_of_clifft_feedforward_sources_are_determined():
    # NOTE: convert_s_to_t.py rule 3 says that seven growth-check measurements
    # "become random with T gates (they were deterministic with S)", but rule 4
    # that the (4,1), (4,5), (6,2) ones are "deterministically 0". Alone, those
    # three are determined and the other four random, with S and with T.
    circuit = _circuit("clifft_d5_T")
    sources = _feedforward_sources(circuit.stim_circuit)
    assert len(sources) == 7
    num_m = circuit.num_measurements
    for name in ("clifft_d5_sproxy", "clifft_d5_T_rule12", "clifft_d5_T"):
        alone = Circuit(
            str(_circuit(name).stim_circuit.without_noise())
            + "\n"
            + "\n".join(f"DETECTOR rec[{m - num_m}]" for m in sources)
        )
        first = alone.num_detectors - len(sources)
        random = [t.val - first for t in nondeterministic_detectors(alone)]
        assert [sources[i] for i in range(7) if i not in random] == [21, 23, 25]
