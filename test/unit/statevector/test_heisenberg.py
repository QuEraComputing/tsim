from fractions import Fraction
from math import pi
from test.helpers.statevector import (
    CIRCUITS,
    ONE_LINE_LAYOUT,
    assert_distributions_close,
    extracted_circuits,
    extracted_distribution,
    f_patterns,
    one_line,
    ops_distribution,
    output_indices,
    quantum_components,
)

import pytest
from zxparam import Phase
from zxparam.extract import Gate

from tsim.statevector.heisenberg import (
    MEASUREMENT,
    ROTATION,
    ComponentLayout,
    heisenberg_ops,
)


def test_t_on_plus_is_a_rotation_about_x():
    ops = heisenberg_ops(
        one_line([Gate("Z", (0,), Phase(Fraction(1, 4)))]), ONE_LINE_LAYOUT
    )
    assert ops.kind.tolist() == [ROTATION, MEASUREMENT]
    assert ops.x.tolist() == [[True], [True]]
    assert ops.z.tolist() == [[False], [False]]
    assert ops.angle[0] == pytest.approx(pi / 8)
    assert ops.sign.tolist() == [[False, False], [False, False]]
    assert ops.slot.tolist() == [-1, 0]
    assert ops.num_outputs == 1


def test_clifford_part_of_a_phase_goes_into_the_frame():
    # 3π/4 = π (a Z, into the frame) − π/4 (a rotation by −π/8).
    ops = heisenberg_ops(
        one_line([Gate("Z", (0,), Phase(Fraction(3, 4)))]), ONE_LINE_LAYOUT
    )
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
    ops = heisenberg_ops(one_line(gates), ONE_LINE_LAYOUT)
    assert ops.kind.tolist() == [ROTATION, MEASUREMENT]
    assert ops.x.tolist() == [[False], [False]]
    assert ops.z.tolist() == [[True], [True]]
    assert ops.sign.tolist() == [[False, True], [False, True]]


def test_a_line_from_an_input_is_rejected():
    with pytest.raises(ValueError, match="input port"):
        heisenberg_ops(one_line([], line_start=3), ONE_LINE_LAYOUT)


def test_component_layout_reads_f_bits_and_ports():
    (component,), layout = quantum_components(CIRCUITS["t_ghz"], sample_detectors=True)
    component_layout = ComponentLayout.of(component, output_indices(component), layout)
    assert list(component_layout.f_selection) == sorted(component_layout.f_selection)
    assert len(component_layout.f_selection) == 3
    assert component_layout.output_indices == (0, 1, 2)
    assert sorted(component_layout.f_position.values()) == [0, 1, 2]


def test_ops_reproduce_the_extracted_circuits():
    for circuit, component_layout in extracted_circuits():
        ops = heisenberg_ops(circuit, component_layout)
        for f in f_patterns(ops.num_f):
            bindings = {p: f[i] for p, i in component_layout.f_position.items()}
            assert_distributions_close(
                ops_distribution(ops, f),
                extracted_distribution(circuit, component_layout, bindings),
            )
