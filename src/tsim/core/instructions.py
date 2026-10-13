"""ZX graph representations of quantum gates and instructions.

Each gate is a function that builds onto a ``GraphBuilder`` through its methods
(``tsim.core.builder``); ``GATE_TABLE`` maps Stim gate names to them.

Gates are built up to a global phase: every graph is a doubled diagram (a
channel, ``U ⊗ Ū`` for a unitary), where global phases cancel, so none are
tracked.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Literal

from zxparam import VertexType

from tsim.core.builder import HELPER_LANE, GraphBuilder, error_phase
from tsim.noise.channels import (
    error_probs,
    heralded_pauli_channel_1_probs,
    pauli_channel_1_probs,
    pauli_channel_2_probs,
)

# =============================================================================
# Non-Clifford Gates
# =============================================================================


def t(builder: GraphBuilder, qubit: int) -> None:
    """Apply T gate (π/4 Z rotation)."""
    r_z(builder, qubit, Fraction(1, 4))


def t_dag(builder: GraphBuilder, qubit: int) -> None:
    """Apply T† gate (-π/4 Z rotation)."""
    r_z(builder, qubit, Fraction(-1, 4))


def r_z(builder: GraphBuilder, qubit: int, phase: Fraction) -> None:
    """Apply R_Z rotation gate with given phase (in units of π)."""
    builder.spider(qubit, VertexType.Z, phase)


def r_x(builder: GraphBuilder, qubit: int, phase: Fraction) -> None:
    """Apply R_X rotation gate with given phase (in units of π)."""
    builder.spider(qubit, VertexType.X, phase)


def r_y(builder: GraphBuilder, qubit: int, phase: Fraction) -> None:
    """Apply R_Y rotation gate with given phase (in units of π)."""
    h_yz(builder, qubit)
    r_z(builder, qubit, phase)
    h_yz(builder, qubit)


def u3(
    builder: GraphBuilder,
    qubit: int,
    theta: Fraction,
    phi: Fraction,
    lambda_: Fraction,
) -> None:
    """Apply U3 gate: U3(θ,φ,λ) = R_Z(φ)·R_Y(θ)·R_Z(λ)."""
    r_z(builder, qubit, lambda_)
    r_y(builder, qubit, theta)
    r_z(builder, qubit, phi)


# =============================================================================
# Pauli Gates
# =============================================================================


def i(builder: GraphBuilder, qubit: int, *_args: float) -> None:
    """Apply identity (advances the column)."""
    builder.advance(qubit)


def ii(builder: GraphBuilder, qubit1: int, qubit2: int, *_args: float) -> None:
    """Apply two-qubit identity (advances the column on both qubits)."""
    i(builder, qubit1)
    i(builder, qubit2)


def x(builder: GraphBuilder, qubit: int) -> None:
    """Apply Pauli X gate."""
    r_x(builder, qubit, Fraction(1, 1))


def y(builder: GraphBuilder, qubit: int) -> None:
    """Apply Pauli Y gate."""
    z(builder, qubit)
    x(builder, qubit)


def z(builder: GraphBuilder, qubit: int) -> None:
    """Apply Pauli Z gate."""
    r_z(builder, qubit, Fraction(1, 1))


# =============================================================================
# Single-Qubit Clifford gates
# =============================================================================


def c_xyz(builder: GraphBuilder, qubit: int) -> None:
    """Right handed period 3 axis cycling gate, sending X -> Y -> Z -> X."""
    s_dag(builder, qubit)
    h(builder, qubit)


def c_nxyz(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending -X -> Y -> Z -> -X."""
    s_dag(builder, qubit)
    sqrt_y_dag(builder, qubit)


def c_xnyz(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending X -> -Y -> Z -> X."""
    s(builder, qubit)
    h(builder, qubit)


def c_xynz(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending X -> Y -> -Z -> X."""
    s(builder, qubit)
    sqrt_y_dag(builder, qubit)


def c_zyx(builder: GraphBuilder, qubit: int) -> None:
    """Left handed period 3 axis cycling gate, sending Z -> Y -> X -> Z."""
    h(builder, qubit)
    s(builder, qubit)


def c_nzyx(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending -Z -> Y -> X -> -Z."""
    s_dag(builder, qubit)
    sqrt_x(builder, qubit)


def c_znyx(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending Z -> -Y -> X -> Z."""
    s(builder, qubit)
    sqrt_x(builder, qubit)


def c_zynx(builder: GraphBuilder, qubit: int) -> None:
    """Period 3 axis cycling gate, sending Z -> Y -> -X -> Z."""
    s(builder, qubit)
    sqrt_x_dag(builder, qubit)


def h(builder: GraphBuilder, qubit: int) -> None:
    """Apply Hadamard gate."""
    builder.hadamard(qubit)


def h_xy(builder: GraphBuilder, qubit: int) -> None:
    """Apply variant of Hadamard gate that swaps the X and Y axes (instead of X and Z)."""
    x(builder, qubit)
    s(builder, qubit)


def h_nxy(builder: GraphBuilder, qubit: int) -> None:
    """Apply Hadamard-like gate that sends -X <-> Y, Z -> -Z."""
    x(builder, qubit)
    s_dag(builder, qubit)


def h_nxz(builder: GraphBuilder, qubit: int) -> None:
    """Apply Hadamard-like gate that sends -X <-> Z."""
    z(builder, qubit)
    sqrt_y_dag(builder, qubit)


def h_yz(builder: GraphBuilder, qubit: int) -> None:
    """Apply variant of Hadamard gate that swaps the Y and Z axes (instead of X and Z)."""
    sqrt_x(builder, qubit)
    z(builder, qubit)


def h_nyz(builder: GraphBuilder, qubit: int) -> None:
    """Apply Hadamard-like gate that sends -Y <-> Z, X -> -X."""
    z(builder, qubit)
    sqrt_x(builder, qubit)


def s(builder: GraphBuilder, qubit: int) -> None:
    """Apply S gate (π/2 Z rotation)."""
    r_z(builder, qubit, Fraction(1, 2))


def sqrt_x(builder: GraphBuilder, qubit: int) -> None:
    """Apply √X gate (π/2 X rotation)."""
    r_x(builder, qubit, Fraction(1, 2))


def sqrt_x_dag(builder: GraphBuilder, qubit: int) -> None:
    """Apply √X† gate (-π/2 X rotation)."""
    r_x(builder, qubit, Fraction(-1, 2))


def sqrt_y(builder: GraphBuilder, qubit: int) -> None:
    """Apply √Y gate (π/2 Y rotation)."""
    z(builder, qubit)
    h(builder, qubit)


def sqrt_y_dag(builder: GraphBuilder, qubit: int) -> None:
    """Apply √Y† gate (-π/2 Y rotation)."""
    h(builder, qubit)
    z(builder, qubit)


def s_dag(builder: GraphBuilder, qubit: int) -> None:
    """Apply S† gate (-π/2 Z rotation)."""
    r_z(builder, qubit, Fraction(-1, 2))


# =============================================================================
# Two-Qubit Gates
# =============================================================================


def cnot(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Apply CNOT (controlled-X) gate."""
    builder.cx(control, target, classically_controlled)


def cy(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Apply controlled-Y gate."""
    s_dag(builder, target)
    cnot(builder, control, target, classically_controlled)
    s(builder, target)


def cz(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Apply controlled-Z gate."""
    builder.cz(control, target, classically_controlled)


def swap(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply SWAP gate."""
    builder.swap(qubit1, qubit2)


def cxswap(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply CX then SWAP."""
    cnot(builder, qubit1, qubit2)
    swap(builder, qubit1, qubit2)


def czswap(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply CZ then SWAP."""
    cz(builder, qubit1, qubit2)
    swap(builder, qubit1, qubit2)


def swapcx(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply SWAP then CX."""
    swap(builder, qubit1, qubit2)
    cnot(builder, qubit1, qubit2)


def swapcz(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Apply SWAP then CZ."""
    swap(builder, qubit1, qubit2)
    cz(builder, qubit1, qubit2)


def iswap(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Swap two qubits and phase the -1 eigenspace of the ZZ observable by i."""
    cnot(builder, qubit1, qubit2)
    s(builder, qubit2)
    cnot(builder, qubit1, qubit2)
    swap(builder, qubit1, qubit2)


def iswap_dag(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Swap two qubits and phase the -1 eigenspace of the ZZ observable by -i."""
    cnot(builder, qubit1, qubit2)
    s_dag(builder, qubit2)
    cnot(builder, qubit1, qubit2)
    swap(builder, qubit1, qubit2)


def sqrt_xx(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the XX observable by i."""
    cnot(builder, qubit1, qubit2)
    sqrt_x(builder, qubit1)
    cnot(builder, qubit1, qubit2)


def sqrt_xx_dag(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the XX observable by -i."""
    cnot(builder, qubit1, qubit2)
    sqrt_x_dag(builder, qubit1)
    cnot(builder, qubit1, qubit2)


def sqrt_yy(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the YY observable by i."""
    s(builder, qubit1)
    cnot(builder, qubit2, qubit1)
    z(builder, qubit1)
    h(builder, qubit2)
    cnot(builder, qubit2, qubit1)
    s(builder, qubit1)


def sqrt_yy_dag(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the YY observable by -i."""
    s_dag(builder, qubit1)
    cnot(builder, qubit2, qubit1)
    h(builder, qubit2)
    z(builder, qubit1)
    cnot(builder, qubit2, qubit1)
    s_dag(builder, qubit1)


def sqrt_zz(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the ZZ observable by i."""
    cnot(builder, qubit1, qubit2)
    s(builder, qubit2)
    cnot(builder, qubit1, qubit2)


def sqrt_zz_dag(builder: GraphBuilder, qubit1: int, qubit2: int) -> None:
    """Phases the -1 eigenspace of the ZZ observable by -i."""
    h(builder, qubit2)
    cnot(builder, qubit1, qubit2)
    h(builder, qubit2)
    s_dag(builder, qubit1)
    s_dag(builder, qubit2)


def xcx(builder: GraphBuilder, control: int, target: int) -> None:
    """X-controlled X gate. Applies X to target if control is in |-> state."""
    h(builder, control)
    cnot(builder, control, target)
    h(builder, control)


def xcy(builder: GraphBuilder, control: int, target: int) -> None:
    """X-controlled Y gate. Applies Y to target if control is in |-> state."""
    h(builder, control)
    cy(builder, control, target)
    h(builder, control)


def xcz(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """X-controlled Z gate. Applies Z to target if control is in |-> state."""
    cnot(
        builder,
        target,
        control,
        classically_controlled[::-1] if classically_controlled else None,
    )


def ycx(builder: GraphBuilder, control: int, target: int) -> None:
    """Y-controlled X gate. Applies X to target if control is in |-i> state."""
    h_yz(builder, control)
    cnot(builder, control, target)
    h_yz(builder, control)


def ycy(builder: GraphBuilder, control: int, target: int) -> None:
    """Y-controlled Y gate. Applies Y to target if control is in |-i> state."""
    h_yz(builder, control)
    cy(builder, control, target)
    h_yz(builder, control)


def ycz(
    builder: GraphBuilder,
    control: int,
    target: int,
    classically_controlled: list[bool] | None = None,
) -> None:
    """Y-controlled Z gate. Applies Z to target if control is in |-i> state."""
    cy(
        builder,
        target,
        control,
        classically_controlled[::-1] if classically_controlled else None,
    )


# =============================================================================
# Error Channels (stores probability arrays for channel sampling)
# =============================================================================


def pauli_channel_1(
    builder: GraphBuilder, qubit: int, px: float, py: float, pz: float
) -> None:
    """Apply single-qubit Pauli channel with given X, Y, Z error probabilities."""
    bit = builder.add_channel(pauli_channel_1_probs(px, py, pz))
    builder.spider(qubit, VertexType.Z, error_phase(bit))
    builder.spider(qubit, VertexType.X, error_phase(bit + 1))


def pauli_channel_2(
    builder: GraphBuilder,
    qubit_i: int,
    qubit_j: int,
    pix: float,
    piy: float,
    piz: float,
    pxi: float,
    pxx: float,
    pxy: float,
    pxz: float,
    pyi: float,
    pyx: float,
    pyy: float,
    pyz: float,
    pzi: float,
    pzx: float,
    pzy: float,
    pzz: float,
) -> None:
    """Apply two-qubit Pauli channel with given error probabilities for all 15 Pauli pairs."""
    bit = builder.add_channel(
        pauli_channel_2_probs(
            pix, piy, piz, pxi, pxx, pxy, pxz, pyi, pyx, pyy, pyz, pzi, pzx, pzy, pzz
        )
    )
    builder.spider(qubit_i, VertexType.Z, error_phase(bit))
    builder.spider(qubit_i, VertexType.X, error_phase(bit + 1))
    builder.spider(qubit_j, VertexType.Z, error_phase(bit + 2))
    builder.spider(qubit_j, VertexType.X, error_phase(bit + 3))


def depolarize1(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply single-qubit depolarizing channel with total error probability p."""
    pauli_channel_1(builder, qubit, p / 3, p / 3, p / 3)


def depolarize2(builder: GraphBuilder, qubit_i: int, qubit_j: int, p: float) -> None:
    """Apply two-qubit depolarizing channel with total error probability p."""
    pauli_channel_2(
        builder,
        qubit_i,
        qubit_j,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
        p / 15,
    )


def x_error(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply X error with probability p."""
    bit = builder.add_channel(error_probs(p))
    builder.spider(qubit, VertexType.X, error_phase(bit))


def y_error(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply Y error with probability p."""
    bit = builder.add_channel(error_probs(p))
    # Y = X·Z, so both vertices use the same error bit
    builder.spider(qubit, VertexType.Z, error_phase(bit))
    builder.spider(qubit, VertexType.X, error_phase(bit))


def z_error(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply Z error with probability p."""
    bit = builder.add_channel(error_probs(p))
    builder.spider(qubit, VertexType.Z, error_phase(bit))


def heralded_pauli_channel_1(
    builder: GraphBuilder,
    qubit: int,
    pi: float,
    px: float,
    py: float,
    pz: float,
) -> None:
    """Apply heralded single-qubit Pauli channel.

    Records a herald bit into the measurement record. When the channel fires
    (with total probability pi+px+py+pz), the herald is 1 and one of I/X/Y/Z
    is applied. When it doesn't fire, the herald is 0 and nothing happens.
    """
    bit = builder.add_channel(heralded_pauli_channel_1_probs(pi, px, py, pz))
    aux = HELPER_LANE
    r(builder, aux)
    builder.spider(aux, VertexType.X, error_phase(bit))  # herald bit flip
    m(builder, aux)
    builder.spider(qubit, VertexType.Z, error_phase(bit + 1))  # Z error bit
    builder.spider(qubit, VertexType.X, error_phase(bit + 2))  # X error bit


def heralded_erase(builder: GraphBuilder, qubit: int, p: float) -> None:
    """Apply heralded erasure channel.

    Special case of heralded_pauli_channel_1 with equal probabilities p/4
    for each of I, X, Y, Z when the channel fires.
    """
    heralded_pauli_channel_1(builder, qubit, p / 4, p / 4, p / 4, p / 4)


# =============================================================================
# Collapsing gates
# =============================================================================


def m(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in Z basis with optional bit-flip error probability p."""
    if invert:
        x(builder, qubit)
    if p > 0:
        # NOTE: the same X error before and after the measurement flips the
        # record but not the state.
        flip = error_phase(builder.add_channel(error_probs(p)))
        builder.spider(qubit, VertexType.X, flip)
        builder.measure(qubit)
        builder.spider(qubit, VertexType.X, flip)
    else:
        builder.measure(qubit)
    if invert:
        x(builder, qubit)


def mpp(
    builder: GraphBuilder,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    invert: bool = False,
    p: float = 0,
) -> None:
    """Measure a single Pauli product.

    Args:
        builder: The graph builder to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product.
        invert: Whether to invert the measurement result.
        p: Measurement flip error probability.

    """
    aux = HELPER_LANE
    r(builder, aux)
    h(builder, aux)
    _apply_pauli_controls(builder, aux, paulis)
    h(builder, aux)
    m(builder, aux, p=p, invert=invert)


def _apply_pauli_controls(
    builder: GraphBuilder,
    aux: int,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
) -> None:
    """Apply controlled-Pauli operations between aux and target qubits."""
    for pauli_type, qubit in paulis:
        if pauli_type == "X":
            cnot(builder, aux, qubit)
        elif pauli_type == "Z":
            cz(builder, aux, qubit)
        elif pauli_type == "Y":
            cy(builder, aux, qubit)
        else:
            raise ValueError(f"Invalid Pauli operator: {pauli_type}")


def _pauli_product_phase(
    builder: GraphBuilder,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    phase_gate: Callable[[GraphBuilder, int], None],
    phase_gate_dag: Callable[[GraphBuilder, int], None],
    dagger: bool,
) -> None:
    """Apply exp(-i theta P) (up to global phase) for a Pauli product P.

    If `dagger` is True, apply exp(+i theta P) instead.
    Phases the -1 eigenspace of P by exp(2i theta). Shared implementation for
    SPP (theta=pi/4, phase by i) and TPP (theta=pi/8, phase by exp(i pi/4))
    and their daggers.
    """
    if len(paulis) == 0:
        return

    # Rotate each qubit so its Pauli eigenvalue maps to Z
    for pauli_type, qubit in paulis:
        if pauli_type == "X":
            h(builder, qubit)
        elif pauli_type == "Y":
            s_dag(builder, qubit)
            h(builder, qubit)

    # Accumulate Z-parity into the last qubit
    _, last_qubit = paulis[-1]
    for _, qubit in paulis[:-1]:
        cnot(builder, qubit, last_qubit)

    # Phase the parity
    if dagger:
        phase_gate_dag(builder, last_qubit)
    else:
        phase_gate(builder, last_qubit)

    # Uncompute parity accumulation
    for _, qubit in reversed(paulis[:-1]):
        cnot(builder, qubit, last_qubit)

    # Undo basis rotations
    for pauli_type, qubit in paulis:
        if pauli_type == "X":
            h(builder, qubit)
        elif pauli_type == "Y":
            h(builder, qubit)
            s(builder, qubit)


def spp(
    builder: GraphBuilder,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    dagger: bool = False,
) -> None:
    """Apply exp(-i pi/4 P) (up to global phase) for a Pauli product P.

    Phases the -1 eigenspace of P by i (or -i if dagger). For a single qubit,
    ``SPP Z0`` is the S gate and ``SPP_DAG Z0`` is S_DAG.

    Args:
        builder: The graph builder to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product P.
        dagger: If True, apply exp(+i pi/4 P) (phase by -i) instead.

    """
    _pauli_product_phase(builder, paulis, s, s_dag, dagger)


def tpp(
    builder: GraphBuilder,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    dagger: bool = False,
) -> None:
    """Apply exp(-i pi/8 P) (up to global phase) for a Pauli product P.

    Phases the -1 eigenspace of P by exp(i pi/4) (or exp(-i pi/4) if dagger).
    For a single qubit, ``TPP Z0`` is the T gate and ``TPP_DAG Z0`` is T_DAG.

    Args:
        builder: The graph builder to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product P.
        dagger: If True, apply exp(+i pi/8 P) (phase by exp(-i pi/4)) instead.

    """
    _pauli_product_phase(builder, paulis, t, t_dag, dagger)


def r_pauli(
    builder: GraphBuilder,
    paulis: list[tuple[Literal["X", "Y", "Z"], int]],
    theta: Fraction,
    dagger: bool = False,
) -> None:
    """Apply exp(-i theta pi/2 P) for a Pauli product P, with theta in half-turns.

    Generalizes ``SPP`` (theta=1/2) and ``TPP`` (theta=1/4) to an arbitrary angle.
    For a single qubit, ``r_pauli`` reduces to the named single-qubit rotation in
    that Pauli's basis (e.g. ``R_PAULI Z0`` is ``R_Z``). If `dagger` is True, apply
    exp(+i theta pi/2 P) instead.

    Args:
        builder: The graph builder to modify.
        paulis: List of (pauli_type, qubit) pairs defining the Pauli product P.
        theta: Rotation angle in half-turns (units of pi).
        dagger: If True, negate the rotation angle.

    """
    _pauli_product_phase(
        builder,
        paulis,
        lambda builder, qubit: r_z(builder, qubit, theta),
        lambda builder, qubit: r_z(builder, qubit, -theta),
        dagger,
    )


def mpad(builder: GraphBuilder, value: int, p: float = 0) -> None:
    """Pad measurement record with a fixed bit value.

    Args:
        builder: The graph builder to modify.
        value: The bit value to record (0 or 1).
        p: Error probability for the recorded bit.

    """
    aux = HELPER_LANE
    r(builder, aux)
    if value == 1:
        x(builder, aux)
    m(builder, aux, p=p)


def mr(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Z-basis demolition measurement (optionally noisy).

    Projects each target qubit into |0> or |1>, reports its value (false=|0>, true=|1>),
    then resets to |0>.
    """
    m(builder, qubit, p=p, invert=invert)
    r(builder, qubit)


def mrx(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """X-basis demolition measurement (optionally noisy).

    Projects each target qubit into |+> or |->, reports its value (false=|+>, true=|->),
    then resets to |+>.
    """
    h(builder, qubit)
    m(builder, qubit, p=p, invert=invert)
    r(builder, qubit)
    h(builder, qubit)


def mry(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Y-basis demolition measurement (optionally noisy).

    Projects each target qubit into |i> or |-i>, reports its value (false=|i>, true=|-i>),
    then resets to |i>.
    """
    h_yz(builder, qubit)
    m(builder, qubit, p=p, invert=invert)
    r(builder, qubit)
    h_yz(builder, qubit)


def mx(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in X basis."""
    h(builder, qubit)
    m(builder, qubit, p=p, invert=invert)
    h(builder, qubit)


def my(builder: GraphBuilder, qubit: int, p: float = 0, invert: bool = False) -> None:
    """Measure qubit in Y basis."""
    h_yz(builder, qubit)
    m(builder, qubit, p=p, invert=invert)
    h_yz(builder, qubit)


def mxx(
    builder: GraphBuilder, q0: int, q1: int, p: float = 0, invert: bool = False
) -> None:
    """Measure two qubits in XX basis."""
    mpp(builder, [("X", q0), ("X", q1)], invert, p=p)


def myy(
    builder: GraphBuilder, q0: int, q1: int, p: float = 0, invert: bool = False
) -> None:
    """Measure two qubits in YY basis."""
    mpp(builder, [("Y", q0), ("Y", q1)], invert, p=p)


def mzz(
    builder: GraphBuilder, q0: int, q1: int, p: float = 0, invert: bool = False
) -> None:
    """Measure two qubits in ZZ basis."""
    mpp(builder, [("Z", q0), ("Z", q1)], invert, p=p)


def r(builder: GraphBuilder, qubit: int) -> None:
    """Z-basis reset.

    Forces each target qubit into the |0> state by silently measuring it in the Z basis
    and applying an X gate if it ended up in the |1> state.
    """
    builder.reset(qubit)


def rx(builder: GraphBuilder, qubit: int) -> None:
    """X-basis reset.

    Forces the qubit into the |+> state: a Z-basis reset followed by H. The reset
    traces out the old state, which does not depend on the basis.
    """
    r(builder, qubit)
    h(builder, qubit)


def ry(builder: GraphBuilder, qubit: int) -> None:
    """Y-basis reset.

    Forces the qubit into the |i> state: a Z-basis reset followed by H_YZ. The
    reset traces out the old state, which does not depend on the basis.
    """
    r(builder, qubit)
    h_yz(builder, qubit)


# =============================================================================
# Gate Dispatch Table
# =============================================================================

GATE_TABLE: dict[str, tuple[Callable[..., None], int]] = {
    # ---- Pauli gates -----------------------------------------------------------
    "I": (i, 1),
    "I_ERROR": (i, 1),
    # QUBIT_COORDS is a visualization annotation; dispatched as identity to allocate
    # the qubit lane. Coordinate args are accepted and ignored by `i`.
    "QUBIT_COORDS": (i, 1),
    "II": (ii, 2),
    "II_ERROR": (ii, 2),
    "X": (x, 1),
    "Y": (y, 1),
    "Z": (z, 1),
    # ---- Non-Clifford gates ---------------------------------------------------
    "T": (t, 1),
    "T_DAG": (t_dag, 1),
    # ---- Single-qubit gates ---------------------------------------------------
    "C_NXYZ": (c_nxyz, 1),
    "C_NZYX": (c_nzyx, 1),
    "C_XNYZ": (c_xnyz, 1),
    "C_XYNZ": (c_xynz, 1),
    "C_XYZ": (c_xyz, 1),
    "C_ZNYX": (c_znyx, 1),
    "C_ZYNX": (c_zynx, 1),
    "C_ZYX": (c_zyx, 1),
    "H": (h, 1),
    "H_NXY": (h_nxy, 1),
    "H_NXZ": (h_nxz, 1),
    "H_NYZ": (h_nyz, 1),
    "H_XY": (h_xy, 1),
    "H_XZ": (h, 1),
    "H_YZ": (h_yz, 1),
    "S": (s, 1),
    "SQRT_X": (sqrt_x, 1),
    "SQRT_X_DAG": (sqrt_x_dag, 1),
    "SQRT_Y": (sqrt_y, 1),
    "SQRT_Y_DAG": (sqrt_y_dag, 1),
    "SQRT_Z": (s, 1),
    "SQRT_Z_DAG": (s_dag, 1),
    "S_DAG": (s_dag, 1),
    # ---- Two-qubit gates ------------------------------------------------------
    "CNOT": (cnot, 2),
    "CX": (cnot, 2),
    "CXSWAP": (cxswap, 2),
    "CZ": (cz, 2),
    "CZSWAP": (czswap, 2),
    "CY": (cy, 2),
    "ISWAP": (iswap, 2),
    "ISWAP_DAG": (iswap_dag, 2),
    "SQRT_XX": (sqrt_xx, 2),
    "SQRT_XX_DAG": (sqrt_xx_dag, 2),
    "SQRT_YY": (sqrt_yy, 2),
    "SQRT_YY_DAG": (sqrt_yy_dag, 2),
    "SQRT_ZZ": (sqrt_zz, 2),
    "SQRT_ZZ_DAG": (sqrt_zz_dag, 2),
    "SWAP": (swap, 2),
    "SWAPCX": (swapcx, 2),
    "SWAPCZ": (swapcz, 2),
    "XCX": (xcx, 2),
    "XCY": (xcy, 2),
    "XCZ": (xcz, 2),
    "YCX": (ycx, 2),
    "YCY": (ycy, 2),
    "YCZ": (ycz, 2),
    "ZCX": (cnot, 2),
    "ZCY": (cy, 2),
    "ZCZ": (cz, 2),
    # ---- Noise channels -------------------------------------------------------
    "DEPOLARIZE1": (depolarize1, 1),
    "DEPOLARIZE2": (depolarize2, 2),
    "PAULI_CHANNEL_1": (pauli_channel_1, 1),
    "PAULI_CHANNEL_2": (pauli_channel_2, 2),
    "HERALDED_ERASE": (heralded_erase, 1),
    "HERALDED_PAULI_CHANNEL_1": (heralded_pauli_channel_1, 1),
    "X_ERROR": (x_error, 1),
    "Y_ERROR": (y_error, 1),
    "Z_ERROR": (z_error, 1),
    # ---- Collapsing gates -----------------------------------------------------
    "M": (m, 1),
    # MPP, SPP, SPP_DAG, MPAD handled by parser
    "MR": (mr, 1),
    "MRX": (mrx, 1),
    "MRY": (mry, 1),
    "MRZ": (mr, 1),
    "MX": (mx, 1),
    "MY": (my, 1),
    "MZ": (m, 1),
    "MXX": (mxx, 2),
    "MYY": (myy, 2),
    "MZZ": (mzz, 2),
    "R": (r, 1),
    "RX": (rx, 1),
    "RY": (ry, 1),
    "RZ": (r, 1),
    # ---- Annotations handled by parser -----------------------------------------
}
