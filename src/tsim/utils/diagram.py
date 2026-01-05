import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

import numpy as np
import stim
from lxml import etree  # type: ignore


class Diagram:
    def __init__(self, svg: str):
        self._svg = svg

    def __str__(self) -> str:
        return self._svg

    def _repr_html_(self) -> Any:
        return self._svg


@dataclass
class GateLabel:
    """Label for a gate in the SVG diagram."""

    label: str  # The gate label (can contain SVG markup)
    annotation: str | None = None  # Optional annotation (shown as text below the gate)


def _width_from_viewbox(svg: str, height: float) -> float | None:
    """Compute width from an SVG viewBox while preserving aspect ratio."""
    m = re.search(r'viewBox="[^"]*\s([\d.]+)\s+([\d.]+)"', svg)
    if m is None:
        return None

    w, h = map(float, m.groups())
    if h == 0:
        return None
    return float(height) / h * w


def wrap_svg(
    svg: str,
    *,
    width: float | None = None,
    height: float | None = None,
) -> str:
    """
    Optionally wrap an SVG string in a scrolling container.

    Args:
        svg: Raw SVG markup.
        width: Explicit width for the container.
        height: Desired height; used to infer width from viewBox if width is not given.
    """
    computed_width = width
    if (
        computed_width is None
        and height is not None
        and isinstance(height, (float, int))
    ):
        computed_width = _width_from_viewbox(svg, float(height))

    if computed_width is None:
        return svg

    return f"""
    <div style="overflow-x: scroll; ">
    <div style="width: {computed_width}px">
    {svg}
    </div>
    </div>
    """


def _subscript(text: str) -> str:
    """Wrap text in a subscript tspan."""
    return f'<tspan baseline-shift="sub" font-size="14">{text}</tspan>'


def _is_err_element(elem: etree._Element) -> bool:
    """Check if an element is an ERR text (contains <tspan>I</tspan>)."""
    for child in elem:
        if child.tag.endswith("tspan") and child.text == "I":
            return True
    return False


def placeholders_to_t(
    svg_string: str, placeholder_id_to_labels: dict[float, GateLabel]
) -> str:
    """
    Replace I_ERROR placeholder gates in an SVG diagram with the actual gate names,
    e.g., T, T†, R_Z, R_X, R_Y, U_3

    Args:
        svg_string: The SVG string from stim's diagram() method containing I_ERROR
        placeholder gates whose p-value are used as identifiers.
        placeholder_id_to_labels: Mapping from identifier (float), i.e. the p values of
        I_ERROR gates, to GateLabel.

    Returns:
        Modified SVG string with I_ERROR gates replaced by the actual gate names.
    """
    root = etree.fromstring(svg_string.encode())

    # Collect all red text elements (the identifier labels)
    red_texts = []
    for elem in root.iter():
        if elem.tag.endswith("text") and elem.get("stroke") == "red" and elem.text:
            red_texts.append(elem)

    # Collect all replacements needed (without modifying the tree)
    replacements: list[tuple[etree._Element, etree._Element, GateLabel]] = []

    for placeholder_id, gate_label in placeholder_id_to_labels.items():
        for red_text in red_texts:
            if str(placeholder_id) in red_text.text:
                err_text = red_text.getprevious()
                if err_text is not None and _is_err_element(err_text):
                    replacements.append((red_text, err_text, gate_label))
                break

    # Perform all modifications
    for red_text, err_text, gate_label in replacements:
        x = err_text.get("x")
        y = err_text.get("y")

        # Create the replacement text element
        new_text = etree.Element(err_text.tag)
        new_text.set("dominant-baseline", "central")
        new_text.set("text-anchor", "middle")
        new_text.set("font-family", "monospace")
        new_text.set("font-size", "30")
        new_text.set("x", x)
        new_text.set("y", y)

        # Handle labels that may contain XML markup
        label = gate_label.label
        if "<" in label:
            fragment = etree.fromstring(f"<root>{label}</root>")
            new_text.text = fragment.text
            for child in fragment:
                new_text.append(child)
        else:
            new_text.text = label

        # Replace ERR element
        parent = err_text.getparent()
        if parent is not None:
            parent.replace(err_text, new_text)

        # Handle red text: remove or update
        if gate_label.annotation is None:
            red_parent = red_text.getparent()
            if red_parent is not None:
                red_parent.remove(red_text)
        else:
            red_text.text = gate_label.annotation
            red_text.set("stroke", "black")

    return etree.tostring(root, encoding="unicode")


def _parse_parametric_tag(tag: str) -> tuple[str, dict[str, Fraction]] | None:
    """Parse a parametric gate tag like R_Z(theta=0.3*pi)."""
    match = re.match(r"^(\w+)\((.*)\)$", tag)
    if not match:
        return None

    gate_name = match.group(1)
    params_str = match.group(2)

    params = {}
    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue
        param_match = re.match(r"^(\w+)=([-+]?[\d.]+)\*pi$", param)
        if not param_match:
            return None
        param_name = param_match.group(1)
        value = Fraction(param_match.group(2)).limit_denominator(10000)
        params[param_name] = value

    return gate_name, params


def tagged_gates_to_placeholder(
    circuit: stim.Circuit,
) -> tuple[stim.Circuit, dict[float, GateLabel]]:
    """
    Replaces tagged gates S[T], S_DAG[T], I[R_X(...)], I[R_Y(...)], I[R_Z(...)],
    I[U3(...)] with I_ERROR placeholder gates whose p-values are used as identifiers.

    Args:
        circuit: The stim circuit to replace tagged gates with I_ERROR placeholder gates.

    Returns:
        A tuple containing the modified circuit and a dictionary mapping the p-values
        of the I_ERROR placeholder gates to the actual gate names.
    """
    modified_circ = stim.Circuit()
    replace_dict: dict[float, GateLabel] = {}

    for instr in circuit:
        assert not isinstance(instr, stim.CircuitRepeatBlock)

        # Handle T gates (S[T] and S_DAG[T])
        if instr.tag == "T" and instr.name in ["S", "S_DAG"]:
            for target in instr.targets_copy():
                identifier = np.round(np.random.rand(), 6)
                DAG = '<tspan baseline-shift="super" font-size="14">†</tspan>'
                label = "T" + DAG if instr.name == "S_DAG" else "T"
                replace_dict[identifier] = GateLabel(label)
                modified_circ.append("I_ERROR", [target], identifier)
            continue

        # Handle parametric gates (I with R_X/R_Y/R_Z/U3 tag)
        if instr.name == "I" and instr.tag:
            result = _parse_parametric_tag(instr.tag)
            if result is not None:
                gate_name, params = result

                for target in instr.targets_copy():
                    identifier = np.round(np.random.rand(), 6)

                    if gate_name in ["R_X", "R_Y", "R_Z"]:
                        axis = gate_name[-1]
                        label = "R" + _subscript(axis)
                        theta = float(params["theta"])
                        annotation = f"{theta:.4g}π"
                        replace_dict[identifier] = GateLabel(label, annotation)

                    elif gate_name == "U3":
                        label = "U" + _subscript("3")
                        replace_dict[identifier] = GateLabel(label, None)

                    else:
                        # Unknown parametric gate, pass through
                        modified_circ.append(instr)
                        continue

                    modified_circ.append("I_ERROR", [target], identifier)
                continue

        modified_circ.append(instr)
    return modified_circ, replace_dict


def render_svg(
    c: stim.Circuit,
    type: str,
    *,
    tick: int | range | None = None,
    filter_coords: Iterable[Iterable[float] | stim.DemTarget] = ((),),
    rows: int | None = None,
    width: float | None = None,
    height: float | None = None,
) -> Diagram:
    """
    Render a stim circuit timeline/timeslice diagram with custom labels applied.
    """
    modified_circ, placeholder_id_to_labels = tagged_gates_to_placeholder(c)
    svg_with_placeholders = str(
        modified_circ.diagram(type, tick=tick, filter_coords=filter_coords, rows=rows)
    )
    svg = placeholders_to_t(svg_with_placeholders, placeholder_id_to_labels)
    wrapped = wrap_svg(svg, width=width, height=height)
    return Diagram(wrapped)
