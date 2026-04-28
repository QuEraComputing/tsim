from fractions import Fraction
from typing import Literal

import pytest
import stim

from tsim.circuit import Circuit
from tsim.utils.diagram import (
    GateLabel,
    _parse_parametric_tag,
    _width_from_viewbox,
    placeholders_to_t,
    render_svg,
    tagged_gates_to_placeholder,
    wrap_svg,
)


def test_width_from_viewbox_scales_width():
    svg = '<svg viewBox="0 0 10 5"></svg>'
    assert _width_from_viewbox(svg, 20.0) == pytest.approx(40.0)


def test_wrap_svg_uses_given_width():
    svg = '<svg viewBox="0 0 4 2"></svg>'
    wrapped = wrap_svg(svg, width=20.0)
    assert "width: 20.0px" in wrapped
    assert svg in wrapped


def test_placeholders_replace_err_and_annotation_removed():
    placeholder_id = 0.123456
    svg = f"""
    <svg viewBox="0 0 10 10">
      <text x="5" y="5"><tspan>I</tspan></text>
      <text stroke="red">{placeholder_id}</text>
    </svg>
    """
    result = placeholders_to_t(svg, {placeholder_id: GateLabel("T", None)})
    assert "T" in result
    assert 'stroke="red"' not in result
    assert "<tspan>I</tspan>" not in result


def test_tagged_gates_to_placeholder_adds_error_and_mapping():
    c = stim.Circuit("I[R_Z(theta=0.25*pi)] 0")
    modified, placeholder_map = tagged_gates_to_placeholder(c)
    assert len(placeholder_map) == 1
    assert "I_ERROR" in str(modified)


def test_tagged_gates_to_placeholder_passes_unknown_parametric_through_once():
    c = stim.Circuit("I[XYZ(theta=0.1*pi)] 0 1 2")
    modified, placeholder_map = tagged_gates_to_placeholder(c)
    assert placeholder_map == {}
    assert len(modified) == 1
    instr = modified[0]
    assert isinstance(instr, stim.CircuitInstruction)
    assert instr.name == "I"
    assert instr.tag == "XYZ(theta=0.1*pi)"
    assert [t.qubit_value for t in instr.targets_copy()] == [0, 1, 2]


def test_parse_parametric_tag_accepts_scientific_notation():
    assert _parse_parametric_tag("R_Z(theta=2.5e-1*pi)") == (
        "R_Z",
        {"theta": Fraction(1, 4)},
    )
    assert _parse_parametric_tag("R_X(theta=1.5E+0*pi)") == (
        "R_X",
        {"theta": Fraction(3, 2)},
    )
    assert _parse_parametric_tag("R_Y(theta=-2.5e-1*pi)") == (
        "R_Y",
        {"theta": Fraction(-1, 4)},
    )


def test_tagged_gates_to_placeholder_handles_scientific_notation():
    c = stim.Circuit("I[R_Z(theta=2.5e-1*pi)] 0")
    modified, placeholder_map = tagged_gates_to_placeholder(c)
    assert len(placeholder_map) == 1
    label = next(iter(placeholder_map.values()))
    assert "Z" in label.label  # R subscripted with Z
    assert label.annotation == "0.25π"
    assert len(modified) == 1
    assert modified[0].name == "I_ERROR"


def test_render_svg_wraps_when_width_given():
    c = stim.Circuit("I[R_Z(theta=0.25*pi)] 0")
    diagram = render_svg(c, "timeline-svg", width=50, zoomable=False)
    # str() returns raw SVG suitable for saving to .svg files
    assert str(diagram).strip().startswith("<svg")
    # _repr_html_() returns the HTML-wrapped version for Jupyter
    html = diagram._repr_html_()
    assert "<div" in html
    assert "width: 50" in html


@pytest.mark.parametrize("diagram_type", ["timeline-svg", "timeslice-svg"])
def test_render_svg_labels_all_gates(
    diagram_type: Literal["timeline-svg", "timeslice-svg"],
):
    c = Circuit("""
        S[T] 0
        TICK
        S_DAG[T] 1
        TICK
        I[R_Z(theta=0.25*pi)] 0
        I[R_X(theta=0.5*pi)] 1
        I[R_Y(theta=-0.75*pi)] 2
        TICK
        I[U3(theta=0.1*pi, phi=0.2*pi, lambda=0.3*pi)] 0
        """)
    html = str(c.diagram(diagram_type))

    # T and T† labels
    assert "T" in html
    assert '<tspan baseline-shift="super" font-size="14">†</tspan>' in html

    # Parametric R axis labels and annotations
    assert '<tspan baseline-shift="sub" font-size="14">Z</tspan>' in html
    assert '<tspan baseline-shift="sub" font-size="14">X</tspan>' in html
    assert '<tspan baseline-shift="sub" font-size="14">Y</tspan>' in html
    assert "0.25π" in html
    assert "0.5π" in html
    assert "-0.75π" in html

    # U3 label
    assert '<tspan baseline-shift="sub" font-size="14">3</tspan>' in html


@pytest.mark.parametrize("diagram_type", ["timeline-svg", "timeslice-svg"])
def test_render_svg_tpp_labels(
    diagram_type: Literal["timeline-svg", "timeslice-svg"],
):
    c = Circuit("""
        TPP X0*Y1*Z2
        TICK
        TPP_DAG Z0*X1
        """)
    html = str(c.diagram(diagram_type))

    # TPP labels should appear (not SPP)
    assert "TPP" in html
    # TPP† for the dagger variant
    assert "TPP†" in html
    # Pauli subscripts should be preserved
    assert '<tspan baseline-shift="sub" font-size="10">X</tspan>' in html
    assert '<tspan baseline-shift="sub" font-size="10">Y</tspan>' in html
    assert '<tspan baseline-shift="sub" font-size="10">Z</tspan>' in html
    # Original SPP labels should NOT appear
    assert "SPP" not in html


def test_render_svg_mixed_spp_and_tpp():
    c = Circuit("""
        SPP X0*Z1
        TICK
        TPP Y0*Z1
        """)
    html = str(c.diagram("timeline-svg"))

    # Both SPP and TPP labels should appear
    assert "SPP" in html
    assert "TPP" in html


def test_diagram_repeat_block():
    c = Circuit("""
        T 0
        REPEAT 100 {
            T 0
        }
    """)
    diagram = c.diagram("timeline-svg", height=150)
    assert "REP100" in str(diagram)
