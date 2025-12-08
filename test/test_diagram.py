from typing import Literal

import pytest
import stim

from tsim.circuit import Circuit
from tsim.util.diagram import (
    GateLabel,
    _width_from_viewbox,
    placeholders_to_t,
    render_svg,
    tagged_gates_to_placeholder,
    wrap_svg,
)


def test_width_from_viewbox_scales_width():
    svg = '<svg viewBox="0 0 10 5"></svg>'
    assert _width_from_viewbox(svg, 20.0) == pytest.approx(40.0)


def test_wrap_svg_infers_width_from_height():
    svg = '<svg viewBox="0 0 4 2"></svg>'
    wrapped = wrap_svg(svg, height=10.0)
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


def test_render_svg_wraps_when_width_given():
    c = stim.Circuit("I[R_Z(theta=0.25*pi)] 0")
    html = str(render_svg(c, "timeline-svg", width=50))
    assert "<div" in html
    assert "width: 50" in html


@pytest.mark.parametrize("diagram_type", ["timeline-svg", "timeslice-svg"])
def test_render_svg_labels_all_gates(
    diagram_type: Literal["timeline-svg", "timeslice-svg"],
):
    c = Circuit(
        """
        S[T] 0
        TICK
        S_DAG[T] 1
        TICK
        I[R_Z(theta=0.25*pi)] 0
        I[R_X(theta=0.5*pi)] 1
        I[R_Y(theta=-0.75*pi)] 2
        TICK
        I[U3(theta=0.1*pi, phi=0.2*pi, lambda=0.3*pi)] 0
        """
    )
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
