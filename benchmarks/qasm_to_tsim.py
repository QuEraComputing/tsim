"""Convert benchmark QASM strings into TSIM/STIM-compatible programs.

This module targets the OpenQASM flavor emitted in ``benchmarks/result.json``.
It supports:

- ``h q[i];`` -> ``H i``
- ``rx(<expr>) q[i];`` -> ``R_X(<angle_in_pi_units>) i``
- ``pp q[i], q[j], ...;`` -> decomposition into TSIM gates

The ``pp`` instruction is treated as a parity-phase gate over Z-parity and is
decomposed using the same helper used by benchmark circuit generation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    # Works when executed as a module from repo root:
    #   uv run python -m benchmarks.qasm_to_tsim ...
    from benchmarks.circuits import exponentiated_pauli
except ModuleNotFoundError:
    # Works when executed directly as a script path:
    #   uv run python benchmarks/qasm_to_tsim.py ...
    from circuits import exponentiated_pauli

_RX_RE = re.compile(r"^rx\((?P<angle>[^)]+)\)\s+q\[(?P<q>\d+)\]$", re.IGNORECASE)
_H_RE = re.compile(r"^h\s+q\[(?P<q>\d+)\]$", re.IGNORECASE)
_PP_RE = re.compile(
    r"^pp(?:\((?P<angle>[^)]+)\))?\s+(?P<args>.+)$",
    re.IGNORECASE,
)
_QIDX_RE = re.compile(r"^q\[(\d+)\]$", re.IGNORECASE)


def _parse_pi_expression(expr: str) -> float:
    """Parse an expression written in units of pi and return the scalar.

    Examples:
        "0.5*pi" -> 0.5
        "-0.5*pi" -> -0.5
        "pi/2" -> 0.5
        "-pi/2" -> -0.5
        "0.25" -> 0.25

    """
    s = expr.strip().lower().replace(" ", "")
    s = s.replace("π", "pi")

    if s.endswith("*pi"):
        return float(s[:-3])
    if s == "pi":
        return 1.0
    if s == "-pi":
        return -1.0
    if "pi/" in s:
        sign = -1.0 if s.startswith("-") else 1.0
        body = s[1:] if s.startswith("-") else s
        if body.startswith("pi/"):
            denom = float(body.split("/", 1)[1])
            return sign / denom

    # Fallback: treat as already in units of pi.
    return float(s)


def _parse_pp_qubits(args: str) -> list[int]:
    """Parse qubit arguments from ``pp q[a], q[b], ...``."""
    qubits: list[int] = []
    for token in args.split(","):
        t = token.strip()
        m = _QIDX_RE.match(t)
        if m is None:
            raise ValueError(f"Invalid pp argument: {token!r}")
        qubits.append(int(m.group(1)))
    if not qubits:
        raise ValueError("pp requires at least one qubit")
    return qubits


def qasm_to_tsim_program(qasm: str, *, pp_phase: float = 0.25) -> str:
    """Convert an OpenQASM string into a TSIM/STIM-style program string.

    Args:
        qasm: Input OpenQASM text.
        pp_phase: Phase parameter for ``pp`` in units of pi.

    Returns:
        A newline-separated TSIM program string.

    """
    out_lines: list[str] = []

    for raw in qasm.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith(";"):
            line = line[:-1].strip()

        lowered = line.lower()
        if lowered.startswith("openqasm ") or lowered.startswith("include "):
            continue
        if lowered.startswith("qreg "):
            continue

        m_h = _H_RE.match(line)
        if m_h is not None:
            out_lines.append(f"H {int(m_h.group('q'))}")
            continue

        m_rx = _RX_RE.match(line)
        if m_rx is not None:
            angle = _parse_pi_expression(m_rx.group("angle"))
            q = int(m_rx.group("q"))
            out_lines.append(f"R_X({angle:g}) {q}")
            continue

        m_pp = _PP_RE.match(line)
        if m_pp is not None:
            qubits = _parse_pp_qubits(m_pp.group("args"))
            phase_expr = m_pp.group("angle")
            phase = pp_phase if phase_expr is None else _parse_pi_expression(phase_expr)
            out_lines.extend(
                exponentiated_pauli(
                    qubits=qubits,
                    paulis=["Z"] * len(qubits),
                    angle=phase,
                )
            )
            continue

        raise ValueError(f"Unsupported QASM instruction: {raw!r}")

    return "\n".join(out_lines)


def json_result_to_tsim_program(
    json_path: str | Path,
    *,
    qasm_key: str = "circuit_qasm",
    pp_phase: float = 0.25,
) -> str:
    """Load a benchmark result JSON and convert its QASM circuit to TSIM text."""
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if qasm_key not in payload:
        raise KeyError(f"Missing key {qasm_key!r} in {json_path}")
    return qasm_to_tsim_program(str(payload[qasm_key]), pp_phase=pp_phase)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert benchmark QASM (from JSON) into TSIM program text."
    )
    p.add_argument(
        "--json",
        required=True,
        help="Path to JSON containing the QASM string (default key: circuit_qasm).",
    )
    p.add_argument(
        "--qasm-key",
        default="circuit_qasm",
        help="JSON key that stores the QASM text.",
    )
    p.add_argument(
        "--pp-phase",
        type=float,
        default=0.25,
        help="Parity-phase angle in pi units used for pp decomposition.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output path. If omitted, prints to stdout.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for converting benchmark QASM JSON payloads."""
    args = _build_parser().parse_args(argv)
    program = json_result_to_tsim_program(
        args.json,
        qasm_key=args.qasm_key,
        pp_phase=args.pp_phase,
    )
    if args.out:
        Path(args.out).write_text(program + "\n", encoding="utf-8")
    else:
        print(program)


if __name__ == "__main__":
    main()
