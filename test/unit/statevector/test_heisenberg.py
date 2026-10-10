from fractions import Fraction
from math import pi
from test.helpers.statevector import (
    CIRCUITS,
    assert_distributions_close,
    extracted_distribution,
    f_patterns,
    hand_built_circuits,
    ops_distribution,
    quantum_components,
)

import pytest
from zx_param import Phase, Scalar
from zx_param.extract import ExtractedCircuit, Gate, extract_circuit

from tsim.statevector.heisenberg import (
    MEASUREMENT,
    ROTATION,
    ComponentLayout,
    heisenberg_ops,
)

LAYOUT = ComponentLayout(f_selection=(0,), f_position={7: 0}, output_indices=(0,))


def _one_line(gates, *, line_start=None):
    return ExtractedCircuit(
        gates=gates,
        n_lines=1,
        line_start={0: line_start},
        line_end={0: 0},
        scalar=Scalar.one(),
        classical_ports=frozenset({0}),
    )


def test_t_on_plus_is_a_rotation_about_x():
    ops = heisenberg_ops(_one_line([Gate("Z", (0,), Phase(Fraction(1, 4)))]), LAYOUT)
    assert ops.kind.tolist() == [ROTATION, MEASUREMENT]
    assert ops.x.tolist() == [[True], [True]]
    assert ops.z.tolist() == [[False], [False]]
    assert ops.angle[0] == pytest.approx(pi / 8)
    assert ops.sign.tolist() == [[False, False], [False, False]]
    assert ops.slot.tolist() == [-1, 0]
    assert ops.num_outputs == 1


def test_clifford_part_of_a_phase_goes_into_the_frame():
    # 3π/4 = π (a Z, into the frame) − π/4 (a rotation by −π/8).
    ops = heisenberg_ops(_one_line([Gate("Z", (0,), Phase(Fraction(3, 4)))]), LAYOUT)
    assert ops.kind.tolist() == [ROTATION, MEASUREMENT]
    assert ops.angle[0] == pytest.approx(-pi / 8)
    assert ops.sign[:, 0].tolist() == [False, False]


def test_noise_flips_the_ops_it_anticommutes_with():
    # |+⟩, Z^f, H, T, measure Z: in the |0⟩ frame the noise is X while the T
    # and the measurement are about Z, so both signs depend on f.
    gates = [
        Gate("Z", (0,), Phase(Fraction(0), frozenset({7}))),
        Gate("H", (0,)),
        Gate("Z", (0,), Phase(Fraction(1, 4))),
    ]
    ops = heisenberg_ops(_one_line(gates), LAYOUT)
    assert ops.kind.tolist() == [ROTATION, MEASUREMENT]
    assert ops.x.tolist() == [[False], [False]]
    assert ops.z.tolist() == [[True], [True]]
    assert ops.sign.tolist() == [[False, True], [False, True]]


def test_a_line_from_an_input_is_rejected():
    with pytest.raises(ValueError, match="input port"):
        heisenberg_ops(_one_line([], line_start=3), LAYOUT)


def test_component_layout_reads_f_bits_and_ports():
    (component,), layout = quantum_components(CIRCUITS["t_ghz"], sample_detectors=True)
    component_layout = ComponentLayout.of(component, layout)
    assert list(component_layout.f_selection) == sorted(component_layout.f_selection)
    assert len(component_layout.f_selection) == 3
    assert component_layout.output_indices == (0, 1, 2)
    assert sorted(component_layout.f_position.values()) == [0, 1, 2]


@pytest.mark.parametrize("name", sorted(CIRCUITS))
@pytest.mark.parametrize("sample_detectors", [True, False])
def test_ops_reproduce_the_extracted_circuit(name, sample_detectors):
    components, layout = quantum_components(
        CIRCUITS[name], sample_detectors=sample_detectors
    )
    assert components
    for component in components:
        component_layout = ComponentLayout.of(component, layout)
        circuit = extract_circuit(component)
        ops = heisenberg_ops(circuit, component_layout)
        for f in f_patterns(ops.num_f):
            bindings = {p: f[i] for p, i in component_layout.f_position.items()}
            assert_distributions_close(
                ops_distribution(ops, f),
                extracted_distribution(circuit, component_layout, bindings),
            )


@pytest.mark.parametrize("index", [0, 1])
def test_ops_reproduce_hand_built_circuits(index):
    _, circuit, component_layout = hand_built_circuits()[index]
    ops = heisenberg_ops(circuit, component_layout)
    for f in f_patterns(ops.num_f):
        bindings = {p: f[i] for p, i in component_layout.f_position.items()}
        assert_distributions_close(
            ops_distribution(ops, f),
            extracted_distribution(circuit, component_layout, bindings),
        )
