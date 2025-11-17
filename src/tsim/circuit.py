import copy
import random
from collections import defaultdict
from fractions import Fraction
from functools import wraps
from typing import Any, Callable, Iterable, TypeVar, overload

import jax
import stim
from jax import Array

import tsim.external.pyzx as zx
from tsim.channels import (
    Channel,
    Depolarize1,
    Depolarize2,
    Error,
    PauliChannel1,
    PauliChannel2,
)
from tsim.external.pyzx import EdgeType, VertexType
from tsim.external.pyzx.graph.base import BaseGraph

_T = TypeVar("_T")


# Overload 1: Used as @accepts_qubit_list (no parameters)
# Returns a callable that accepts int | Iterable[int] for the first qubit parameter
@overload
def accepts_qubit_list(
    func: Callable[..., Any],
) -> Callable[..., None]: ...


# Overload 2: Used as @accepts_qubit_list(num_qubits=...) - returns a decorator
# The decorator transforms functions to accept int | Iterable[int] for qubit parameters
@overload
def accepts_qubit_list(
    func: None = None, *, num_qubits: int = 1
) -> Callable[[Callable[..., Any]], Callable[..., None]]: ...


def accepts_qubit_list(func=None, *, num_qubits=1):  # type: ignore[no-untyped-def]
    """Decorator that allows a method to accept either individual qubits or a list of qubits.

    When a list is provided, the decorated method is called once for each group of qubits.

    Args:
        num_qubits: Number of qubits the function expects (default: 1)
                   For single-qubit gates like H, use 1
                   For two-qubit gates like CNOT, use 2

    Example:
        @accepts_qubit_list
        def h(self, qubit: int):
            # implementation

        circ.h(0)        # Applies to single qubit
        circ.h([0,1,2])  # Applies to qubits 0, 1, and 2

        @accepts_qubit_list(num_qubits=2)
        def cnot(self, control: int, target: int):
            # implementation

        circ.cnot(0, 1)          # Single CNOT
        circ.cnot([0,1,2,3])     # CNOT on (0,1) and (2,3)
        circ.cnot([0,1,2,3,4,5]) # CNOT on (0,1), (2,3), and (4,5)
    """

    def decorator(f):  # type: ignore[no-untyped-def]
        @wraps(f)
        def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Check if the first argument is an iterable (but not a string)
            if args and isinstance(args[0], Iterable) and not isinstance(args[0], str):
                qubit_list = [int(a) if not isinstance(a, str) else a for a in args[0]]
                remaining_args = args[1:]

                # Split the qubit list into chunks of size num_qubits
                if len(qubit_list) % num_qubits != 0:
                    raise ValueError(
                        f"Qubit list length ({len(qubit_list)}) must be divisible by {num_qubits}"
                    )

                # Apply the function to each chunk
                for i in range(0, len(qubit_list), num_qubits):
                    chunk = qubit_list[i : i + num_qubits]
                    f(self, *chunk, *remaining_args, **kwargs)
                return None
            else:
                # Normal case - call function with all arguments
                return f(self, *args, **kwargs)

        return wrapper

    # Support both @accepts_qubit_list and @accepts_qubit_list(num_qubits=2)
    if func is not None:
        # Called without parentheses: @accepts_qubit_list
        return decorator(func)
    else:
        # Called with parentheses: @accepts_qubit_list(num_qubits=2)
        return decorator


class Circuit:
    """Quantum circuit represented as a ZX-diagram with noise support."""

    def __init__(self, key: Array | None = None):
        """Initialize an empty circuit with optional random key for noise sampling."""
        if key is None:
            key = jax.random.key(0)
        self._key = key
        self.g = zx.Graph()
        self._last_vertex: dict[int, int] = {}
        self.error_channels: list[Channel] = []

        self._num_error_bits: int = 0
        self._rec: list[int] = []
        self._silent_rec: list[int] = []
        self._detectors: list[int] = []
        self._observables_dict: dict[int, int] = {}  # idx: vertex
        self._qubit_to_input: dict[int, int] = {}

    @property
    def _observables(self) -> list[int]:
        return [self._observables_dict[i] for i in sorted(self._observables_dict)]

    def _last_row(self, qubit: int):
        return self.g.row(self._last_vertex[qubit])

    def _last_edge(self, qubit: int):
        edges = self.g.incident_edges(self._last_vertex[qubit])
        assert len(edges) == 1
        return edges[0]

    def _add_dummy(self, qubit: int, row: float | int | None = None):
        if row is None:
            row = self._last_row(qubit) + 1
        v1 = self.g.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=row)  # type: ignore[arg-type]
        self._last_vertex[qubit] = v1
        return v1

    def _add_lane(self, qubit: int):
        v1 = self.g.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=0)
        self._qubit_to_input[qubit] = v1
        v2 = self.g.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=1)
        self.g.add_edge((v1, v2))
        self._last_vertex[qubit] = v2
        return v1

    @property
    def num_measurements(self) -> int:
        """Counts the number of bits produced when sampling the circuit's measurements."""
        return len(self._rec)

    @property
    def num_detectors(self) -> int:
        """Counts the number of bits produced when sampling the circuit's detectors."""
        return len(self._detectors)

    @property
    def num_observables(self) -> int:
        """
        Counts the number of bits produced when sampling the circuit's logical observables.

        This is one more than the largest observable index given to OBSERVABLE_INCLUDE.
        """
        return max(self._observables_dict.keys(), default=-1) + 1

    @property
    def num_qubits(self) -> int:
        """Counts the number of qubits used when simulating the circuit.

        This is always one more than the largest qubit index used by the circuit.
        """
        return max(self._last_vertex.keys(), default=-1) + 1

    @accepts_qubit_list
    def h(self, qubit: int):
        """Apply Hadamard gate to qubit(s)."""
        g = self.g
        if qubit not in self._last_vertex:
            self._add_lane(qubit)

        e = self._last_edge(qubit)
        g.set_edge_type(
            e,
            EdgeType.HADAMARD if g.edge_type(e) == EdgeType.SIMPLE else EdgeType.SIMPLE,
        )

    @accepts_qubit_list
    def x_phase(self, qubit: int, phase: Fraction):
        g = self.g
        if qubit not in self._last_vertex:
            self._add_lane(qubit)
        v1 = self._last_vertex[qubit]
        g.set_type(v1, VertexType.X)
        g.set_phase(v1, phase)
        v2 = self._add_dummy(qubit)
        g.add_edge((v1, v2))

    @accepts_qubit_list
    def x(self, qubit: int):
        """Apply Pauli X gate to qubit(s)."""
        self.x_phase(qubit, Fraction(1, 1))

    @accepts_qubit_list
    def z_phase(self, qubit: int, phase: Fraction):
        g = self.g
        if qubit not in self._last_vertex:
            self._add_lane(qubit)
        v1 = self._last_vertex[qubit]
        g.set_type(v1, VertexType.Z)
        g.set_phase(v1, phase)
        v2 = self._add_dummy(qubit)
        g.add_edge((v1, v2))

    @accepts_qubit_list
    def z(self, qubit: int):
        """Apply Pauli Z gate to qubit(s)."""
        self.z_phase(qubit, Fraction(1, 1))

    @accepts_qubit_list
    def y(self, qubit: int):
        """Apply Pauli Y gate to qubit(s)."""
        self.z(qubit)
        self.x(qubit)
        self.g.scalar.add_phase(Fraction(1, 2))

    @accepts_qubit_list
    def s(self, qubit: int):
        self.z_phase(qubit, Fraction(1, 2))

    @accepts_qubit_list
    def s_dag(self, qubit: int):
        self.z_phase(qubit, Fraction(-1, 2))

    @accepts_qubit_list
    def t(self, qubit: int):
        self.z_phase(qubit, Fraction(1, 4))

    @accepts_qubit_list
    def t_dag(self, qubit: int):
        self.z_phase(qubit, Fraction(-1, 4))

    @accepts_qubit_list
    def sqrt_x(self, qubit: int):
        self.x_phase(qubit, Fraction(1, 2))

    @accepts_qubit_list
    def sqrt_y(self, qubit: int):
        self.z(qubit)
        self.h(qubit)
        self.g.scalar.add_phase(Fraction(1, 4))

    @accepts_qubit_list
    def sqrt_y_dag(self, qubit: int):
        self.h(qubit)
        self.z(qubit)
        self.g.scalar.add_phase(Fraction(-1, 4))

    @accepts_qubit_list
    def sqrt_x_dag(self, qubit: int):
        self.x_phase(qubit, Fraction(-1, 2))

    @accepts_qubit_list
    def sqrt_z(self, qubit: int):
        self.s(qubit)

    @accepts_qubit_list
    def sqrt_z_dag(self, qubit: int):
        self.s_dag(qubit)

    @accepts_qubit_list
    def r(self, qubit: int):
        """Reset qubit(s) to |0⟩ state."""
        g = self.g
        if qubit not in self._last_vertex:
            v1 = self._add_lane(qubit)
            g.set_type(v1, VertexType.X)
        else:
            # If the last vertex is not a measurement, we need to perform silent measur.
            v = self._last_vertex[qubit]
            neighbors = list(g.neighbors(v))
            assert len(neighbors) == 1
            n = neighbors[0]
            last_vertex_is_measurement = any("rec" in var for var in g._phaseVars[n])
            if not last_vertex_is_measurement:
                self._m(qubit, silent=True)
            r = self._last_row(qubit)
            v1 = self._last_vertex[qubit]
            g.set_type(v1, VertexType.X)
            v2 = list(g.neighbors(v1))[0]
            g.remove_edge((v1, v2))
            v3 = self._add_dummy(qubit, r + 1)

            g.add_edge((v1, v3))

    @accepts_qubit_list
    def rx(self, qubit: int):
        if qubit in self._last_vertex:
            self.h(qubit)
        self.r(qubit)
        self.h(qubit)

    @accepts_qubit_list
    def mr(self, qubit: int, p: float = 0):
        """Measure and reset qubit(s) (optionally noisy)"""
        if p > 0:
            self.x_error(qubit, p)
        self.m(qubit, p=p)
        self.r(qubit)

    def _m(self, qubit: int, p: float = 0, silent: bool = False):
        if p > 0:
            self.x_error(qubit, p)
        g = self.g
        if qubit not in self._last_vertex:
            self._add_lane(qubit)
        v1 = self._last_vertex[qubit]
        g.set_type(v1, VertexType.Z)
        if not silent:
            g.set_phase(v1, f"rec[{len(self._rec)}]")
            self._rec.append(v1)
        else:
            g.set_phase(v1, f"m[{len(self._silent_rec)}]")
            self._silent_rec.append(v1)
        v2 = self._add_dummy(qubit)
        g.add_edge((v1, v2))

    @accepts_qubit_list
    def m(self, qubit: int, p: float = 0):
        """Measure qubit(s) in Z basis (optionally noisy)"""
        self._m(qubit, p, silent=False)

    @accepts_qubit_list
    def mx(self, qubit: int, p: float = 0):
        """Measure qubit(s) in X basis (optionally noisy)"""
        self.h(qubit)
        self.m(qubit, p=p)

    @accepts_qubit_list
    def mpp(self, pp: str | list[str]):
        # TODO: express within ZX

        if isinstance(pp, list):
            for pp_ in pp:
                self.mpp(pp_)
            return

        aux = -2
        self.r(aux)
        self.h(aux)

        components = pp.split("*")

        for c in components:
            p, i = c[0].lower(), int(c[1:])

            if p == "x":
                self.cx(aux, i)
            elif p == "z":
                self.cz(aux, i)
            elif p == "y":
                self.cy(aux, i)
            else:
                raise ValueError(f"Invalid Pauli operator: {p}")

        self.h(aux)
        self.m(aux)

    @accepts_qubit_list(num_qubits=2)
    def cnot(self, control: int, target: int):
        """Apply CNOT gate between control and target qubit(s)."""
        g = self.g
        if control not in self._last_vertex:
            self._add_lane(control)
        if target not in self._last_vertex:
            self._add_lane(target)

        lr1 = self._last_row(control)
        lr2 = self._last_row(target)
        r = max(lr1, lr2)

        v1 = self._last_vertex[control]
        v2 = self._last_vertex[target]
        g.set_type(v1, VertexType.Z)
        g.set_type(v2, VertexType.X)
        g.set_row(v1, r)
        g.set_row(v2, r)
        g.add_edge((v1, v2))

        v3 = self._add_dummy(control, int(r + 1))
        v4 = self._add_dummy(target, int(r + 1))
        g.add_edge((v1, v3))
        g.add_edge((v2, v4))

        g.scalar.add_power(1)

    cx = cnot  # alias

    @accepts_qubit_list(num_qubits=2)
    def cz(self, control: int, target: int):
        """Apply CZ gate between control and target qubit(s)."""
        g = self.g
        if control not in self._last_vertex:
            self._add_lane(control)
        if target not in self._last_vertex:
            self._add_lane(target)

        lr1 = self._last_row(control)
        lr2 = self._last_row(target)
        r = max(lr1, lr2)

        v1 = self._last_vertex[control]
        v2 = self._last_vertex[target]
        g.set_type(v1, VertexType.Z)
        g.set_type(v2, VertexType.Z)
        g.set_row(v1, r)
        g.set_row(v2, r)
        g.add_edge((v1, v2), EdgeType.HADAMARD)

        v3 = self._add_dummy(control, int(r + 1))
        v4 = self._add_dummy(target, int(r + 1))
        g.add_edge((v1, v3))
        g.add_edge((v2, v4))

        g.scalar.add_power(1)

    @accepts_qubit_list(num_qubits=2)
    def cy(self, control: int, target: int):
        self.s_dag(target)
        self.cx(control, target)
        self.s(target)

    def _error(self, qubit: int, error_type: int, phase: str):
        if qubit not in self._last_vertex:
            self._add_lane(qubit)
        g = self.g
        v1 = self._last_vertex[qubit]
        v2 = self._add_dummy(qubit)
        g.add_edge((v1, v2))

        g.set_type(v1, error_type)  # type: ignore[arg-type]
        g.set_phase(v1, phase)  # type: ignore[arg-type]

    @accepts_qubit_list
    def x_error(self, qubit: int, p: float):
        """Apply X error with probability p to qubit(s)."""
        self._key, subkey = jax.random.split(self._key)
        self.error_channels.append(Error(p, subkey))
        self._error(qubit, VertexType.X, f"e{self._num_error_bits}")
        self._num_error_bits += 1

    @accepts_qubit_list
    def z_error(self, qubit: int, p: float):
        """Apply Z error with probability p to qubit(s)."""
        self._key, subkey = jax.random.split(self._key)
        self.error_channels.append(Error(p, subkey))
        self._error(qubit, VertexType.Z, f"e{self._num_error_bits}")
        self._num_error_bits += 1

    @accepts_qubit_list
    def y_error(self, qubit: int, p: float):
        """Apply Y error with probability p to qubit(s)."""
        self._key, subkey = jax.random.split(self._key)
        self.error_channels.append(Error(p, subkey))
        self._error(qubit, VertexType.Z, f"e{self._num_error_bits}")
        self._error(qubit, VertexType.X, f"e{self._num_error_bits}")
        self._num_error_bits += 1

    @accepts_qubit_list
    def depolarize1(self, qubit: int, p: float):
        """Apply single-qubit depolarizing noise with probability p."""
        self._key, subkey = jax.random.split(self._key)
        self.error_channels.append(Depolarize1(p, subkey))
        self._error(qubit, VertexType.Z, f"e{self._num_error_bits}")
        self._error(qubit, VertexType.X, f"e{self._num_error_bits + 1}")
        self._num_error_bits += 2

    @accepts_qubit_list
    def pauli_channel_1(self, qubit: int, px: float, py: float, pz: float):
        """Apply single-qubit Pauli channel with specified error probabilities."""
        self._key, subkey = jax.random.split(self._key)
        self.error_channels.append(PauliChannel1(px, py, pz, subkey))
        self._error(qubit, VertexType.Z, f"e{self._num_error_bits}")
        self._error(qubit, VertexType.X, f"e{self._num_error_bits + 1}")
        self._num_error_bits += 2

    @accepts_qubit_list(num_qubits=2)
    def depolarize2(self, qubit_i: int, qubit_j: int, p: float):
        """Apply two-qubit depolarizing noise with probability p."""
        self._key, subkey = jax.random.split(self._key)
        self.error_channels.append(Depolarize2(p, subkey))
        self._error(qubit_i, VertexType.Z, f"e{self._num_error_bits}")
        self._error(qubit_i, VertexType.X, f"e{self._num_error_bits + 1}")
        self._error(qubit_j, VertexType.Z, f"e{self._num_error_bits + 2}")
        self._error(qubit_j, VertexType.X, f"e{self._num_error_bits + 3}")
        self._num_error_bits += 4

    @accepts_qubit_list(num_qubits=2)
    def pauli_channel_2(
        self,
        qubit_i: int,
        qubit_j: int,
        pix: float = 0,
        piy: float = 0,
        piz: float = 0,
        pxi: float = 0,
        pxx: float = 0,
        pxy: float = 0,
        pxz: float = 0,
        pyi: float = 0,
        pyx: float = 0,
        pyy: float = 0,
        pyz: float = 0,
        pzi: float = 0,
        pzx: float = 0,
        pzy: float = 0,
        pzz: float = 0,
    ):
        """Apply two-qubit Pauli channel with specified error probabilities."""
        self._key, subkey = jax.random.split(self._key)
        self.error_channels.append(
            PauliChannel2(
                pix,
                piy,
                piz,
                pxi,
                pxx,
                pxy,
                pxz,
                pyi,
                pyx,
                pyy,
                pyz,
                pzi,
                pzx,
                pzy,
                pzz,
                subkey,
            )
        )
        self._error(qubit_i, VertexType.Z, f"e{self._num_error_bits}")
        self._error(qubit_i, VertexType.X, f"e{self._num_error_bits + 1}")
        self._error(qubit_j, VertexType.Z, f"e{self._num_error_bits + 2}")
        self._error(qubit_j, VertexType.X, f"e{self._num_error_bits + 3}")
        self._num_error_bits += 4

    @accepts_qubit_list
    def i(self, qubit):
        if qubit not in self._last_vertex:
            self._add_lane(qubit)
        v = self._last_vertex[qubit]
        self.g.set_row(v, self._last_row(qubit) + 1)

    def tick(self):
        """Add a tick to the circuit."""
        if len(self._last_vertex) == 0:
            return
        r = max(self._last_row(q) for q in self._last_vertex)
        for q in self._last_vertex:
            self.g.set_row(self._last_vertex[q], r)

    def diagram(self, labels=False) -> BaseGraph:
        """Display the ZX-diagram representation of the circuit."""
        if len(self.g.vertices()) == 0:
            return self.g
        g = self.g.clone()
        max_row = max(self.g.row(v) for v in self._last_vertex.values())
        for q in self._last_vertex:
            g.set_row(self._last_vertex[q], max_row)
        zx.draw(g, labels=labels)
        return g

    def detector(self, rec: list[int], *args: Any):
        """Add a detector that checks parity of measurement records."""
        r = min(set([self.g.row(self._rec[r]) for r in rec])) - 0.5
        d_rows = set([self.g.row(d) for d in self._detectors + self._observables])
        while r in d_rows:
            r += 1
        v0 = self.g.add_vertex(
            VertexType.X, qubit=-1, row=r, phase=f"det[{len(self._detectors)}]"  # type: ignore[arg-type]
        )
        for rec_ in rec:
            self.g.add_edge((v0, self._rec[rec_]))
        self._detectors.append(v0)

    def observable_include(self, rec: list[int], idx: int):
        """Include measurement records in a logical observable."""
        idx = int(idx)

        if idx not in self._observables_dict:
            r = min(set([self.g.row(self._rec[r]) for r in rec])) - 0.5
            d_rows = set([self.g.row(d) for d in self._detectors + self._observables])
            while r in d_rows:
                r += 1
            v0 = self.g.add_vertex(
                VertexType.X, qubit=-1, row=r, phase=f"obs[{idx}]"  # type: ignore[arg-type]
            )
            self._observables_dict[idx] = v0

        v0 = self._observables_dict[idx]
        for rec_ in rec:
            self.g.add_edge((v0, self._rec[rec_]))

    def to_tensor(self) -> Any:
        """Convert circuit to tensor representation."""
        g = self.g.copy()
        g.normalize()
        return g.to_tensor()

    def to_matrix(self) -> Any:
        """Convert circuit to matrix representation."""
        g = self.g.copy()
        g.normalize()
        return g.to_matrix()

    def get_sampling_graph(self, sample_detectors: bool = False) -> BaseGraph:
        """Get a ZX graph that can be used to compute probabilities.

        This graph will be constructed as follows:

        1. Double the ZX-diagram by composing it with its adjoint.
        2. Connect all rec[i] vertices to their corresponding adjoint rec[i] vertices.
        3. Add outputs:
        (a) When sampling measurements (i.e. `sample_detectors` is False),
            add output nodes for each measurement. Detectors and observables are
            removed since they are ignored when sampling measurements.
        (b) When sampling detectors and observables (i.e. `sample_detectors` is True),
            add output nodes for each detector and observable. Only one set of detector
            and observable nodes is kept, i.e., detectors and observables are not
            composed with their adjoints.
        """

        g = self.g.copy()

        # clean up last row
        max_row = max(self.g.row(v) for v in self._last_vertex.values())
        for q in self._last_vertex:
            g.set_row(self._last_vertex[q], max_row)

        num_measurements = len(self._rec)
        outputs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
        g.set_outputs(tuple(outputs))

        g_adj = g.adjoint()
        g.compose(g_adj)

        g = g.copy()

        label_to_vertex: dict[str, list[int]] = defaultdict(list)
        annotation_to_vertex: dict[str, list[int]] = defaultdict(list)
        for v in g.vertices():
            phase_vars = g._phaseVars[v]
            if not len(phase_vars) == 1:
                continue
            phase = list(phase_vars)[0]
            if "det" in phase or "obs" in phase or "rec" in phase or "m" in phase:
                label_to_vertex[phase].append(v)
            if "det" in phase or "obs" in phase:
                annotation_to_vertex[phase].append(v)

        outputs = [0] * num_measurements if not sample_detectors else []

        # connect all rec[i] vertices to each other and add red vertex with rec[i] label
        for i in range(num_measurements):
            label = f"rec[{i}]"
            vertices = label_to_vertex[label]

            assert len(vertices) == 2
            v0, v1 = vertices
            if not g.connected(v0, v1):
                g.add_edge((v0, v1))
            g.set_phase(v0, 0)
            g.set_phase(v1, 0)

            # add outputs
            if not sample_detectors:
                v3 = g.add_vertex(VertexType.BOUNDARY, qubit=-1, row=i + 1, phase=0)
                outputs[i] = v3
                g.add_edge((v0, v3))

        # connect all m[i] vertices to each other
        for i in range(len(self._silent_rec)):
            label = f"m[{i}]"
            vertices = label_to_vertex[label]

            assert len(vertices) == 2
            v0, v1 = vertices
            if not g.connected(v0, v1):
                g.add_edge((v0, v1))
            g.set_phase(v0, 0)
            g.set_phase(v1, 0)

        if not sample_detectors:
            # sample measurements: remove detectors and observables
            for vertices in annotation_to_vertex.values():
                assert len(vertices) == 2
                for v in vertices:
                    g.remove_vertex(v)
        else:
            # sample detectors and observables:
            # keep detector and observables but remove the adjoint (duplicated)
            # annotation nodes
            for vertices in annotation_to_vertex.values():
                assert len(vertices) == 2
                g.remove_vertex(vertices.pop())

            labels = [f"det[{i}]" for i in range(len(self._detectors))] + [
                f"obs[{i}]" for i in self._observables_dict.keys()
            ]
            for label in labels:
                vs = annotation_to_vertex[label]
                assert len(vs) == 1
                v = vs[0]
                row = g.row(v)
                vb = g.add_vertex(VertexType.BOUNDARY, qubit=-2, row=row)
                g.add_edge((v, vb))
                g.set_phase(v, 0)
                outputs.append(vb)

        g.set_outputs(tuple(outputs))

        return g

    def copy(self):
        return copy.deepcopy(self)

    def without_noise(self):
        """Return a copy of the circuit with all noise removed."""
        c = self.copy()
        g = c.g
        label_to_vertex = {}
        for v in g.vertices():
            phase_vars = g._phaseVars[v]
            if not len(phase_vars) == 1:
                continue
            phase = list(phase_vars)[0]
            assert isinstance(phase, str)
            if phase.startswith("e"):
                label_to_vertex[phase] = v

        for vertex in label_to_vertex.values():
            neighbors = g.neighbors(vertex)
            assert len(neighbors) == 2
            n0, n1 = list(neighbors)
            le = g.edge(n0, vertex)
            re = g.edge(vertex, n1)
            le_type = g.edge_type(le)
            re_type = g.edge_type(re)

            if le_type == re_type:
                g.add_edge((n0, n1), EdgeType.SIMPLE)
            else:
                g.add_edge((n0, n1), EdgeType.HADAMARD)
            g.remove_vertex(vertex)
        c.error_channels = []
        return c

    def without_annotations(self):
        """Return a copy of the circuit with detectors and observables removed."""
        c = self.copy()
        g = c.g
        label_to_vertex = defaultdict(list)
        for v in g.vertices():
            phase_vars = g._phaseVars[v]
            if not len(phase_vars) == 1:
                continue
            phase = list(phase_vars)[0]
            assert isinstance(phase, str)
            label_to_vertex[phase].append(v)

        for label, vertices in label_to_vertex.items():
            if "det" in label or "obs" in label:
                for v in vertices:
                    g.remove_vertex(v)
            if "rec" in label or "m" in label:
                g.set_phase(vertices[0], 0)

        c.error_channels = []
        c._detectors = []
        c._observables_dict = {}
        c._rec = []
        return c

    def append_from_stim_program(
        self,
        stim_circuit: stim.Circuit,
        skip_annotations: bool = False,
        skip_detectors: bool = False,
        replace_s_with_t: bool = False,
    ):
        """Append gates from a Stim circuit to this circuit."""

        ignore_gates = {"qubit_coords"}

        if skip_annotations:
            ignore_gates.add("observable_include")
            ignore_gates.add("detector")
            ignore_gates.add("m")
        if skip_detectors:
            ignore_gates.add("detector")

        for instruction in stim_circuit.flattened():
            name = instruction.name.lower()
            if name in ignore_gates:
                continue

            if name == "s" and (replace_s_with_t or instruction.tag == "T"):
                name = "t"
            if name == "s_dag" and (replace_s_with_t or instruction.tag == "T"):
                name = "t_dag"

            if name == "tick":
                self.tick()
                continue
            if name == "mpp":
                # TODO: improve parsing
                args = str(instruction).split(" ")[1:]
                self.mpp(args)
                continue

            assert not isinstance(instruction, stim.CircuitRepeatBlock)
            targets = [t.value for t in instruction.targets_copy()]
            args = instruction.gate_args_copy()
            func = getattr(self, name)
            func(targets, *args)

    @staticmethod
    def from_stim_program(
        stim_circuit: stim.Circuit,
        skip_annotations: bool = False,
        skip_detectors: bool = False,
        replace_s_with_t: bool = False,
    ):
        """Create a Circuit from a Stim circuit."""
        c = Circuit()
        c.append_from_stim_program(
            stim_circuit,
            skip_annotations=skip_annotations,
            skip_detectors=skip_detectors,
            replace_s_with_t=replace_s_with_t,
        )
        return c

    @staticmethod
    def from_stim_program_text(
        stim_program_text: str,
        skip_annotations: bool = False,
        skip_detectors: bool = False,
        replace_s_with_t: bool = False,
    ):
        """Create a Circuit from a Stim program text."""
        return Circuit.from_stim_program(
            stim.Circuit(stim_program_text),
            skip_annotations,
            skip_detectors,
            replace_s_with_t,
        )

    def append_from_stim_program_text(
        self,
        stim_program_text: str,
        skip_annotations: bool = False,
        skip_detectors: bool = False,
        replace_s_with_t: bool = False,
    ):
        """Append gates from a Stim program text to this circuit."""
        self.append_from_stim_program(
            stim.Circuit(stim_program_text),
            skip_annotations,
            skip_detectors,
            replace_s_with_t,
        )

    def compile_sampler(self):
        """Compile circuit into a measurement sampler."""
        from tsim.sampler import CompiledMeasurementSampler

        return CompiledMeasurementSampler(self)

    def compile_detector_sampler(self):
        """Compile circuit into a detector sampler."""
        from tsim.sampler import CompiledDetectorSampler

        return CompiledDetectorSampler(self)

    @staticmethod
    def random(
        qubits: int,
        depth: int,
        p: float = 0.0,
        p_t: float | None = None,
        p_s: float | None = None,
        p_hsh: float | None = None,
        p_cnot: float | None = None,
        seed: int | None = None,
    ) -> "Circuit":
        """Generate a random quantum circuit with depolarizing noise.

        Args:
            qubits: Number of qubits in the circuit
            depth: Number of gate layers to generate
            p: Depolarizing noise parameter (applied after each gate)
            p_t: Probability of applying T gate (default: equal probability)
            p_s: Probability of applying S gate (default: equal probability)
            p_hsh: Probability of applying H, sqrt(X), or sqrt(Y) gates (default: equal probability)
            p_cnot: Probability of applying CNOT gate (default: equal probability)
            seed: Random seed for reproducibility (optional)

        Returns:
            A randomly generated Circuit with specified gates and noise
        """
        if seed is not None:
            random.seed(seed)

        # Set default equal probabilities if not specified
        gate_probs = []
        gate_types = []

        if p_t is not None and p_t > 0:
            gate_probs.append(p_t)
            gate_types.append("t")
        if p_s is not None and p_s > 0:
            gate_probs.append(p_s)
            gate_types.append("s")
        if p_hsh is not None and p_hsh > 0:
            gate_probs.append(p_hsh)
            gate_types.append("hsh")
        if p_cnot is not None and p_cnot > 0:
            gate_probs.append(p_cnot)
            gate_types.append("cnot")

        # If no probabilities specified, use equal probabilities for all gates
        if not gate_types:
            gate_types = ["t", "s", "hsh", "cnot"]
            gate_probs = [0.25, 0.25, 0.25, 0.25]
        else:
            # Normalize probabilities
            total = sum(gate_probs)
            gate_probs = [p / total for p in gate_probs]

        circ = Circuit()

        for q in range(qubits):
            circ.r(q)

        for _ in range(depth):
            gate_type = random.choices(gate_types, weights=gate_probs, k=1)[0]

            if gate_type == "t":
                q = random.randint(0, qubits - 1)
                circ.t(q)
                if p > 0:
                    circ.depolarize1(q, p)

            elif gate_type == "s":
                q = random.randint(0, qubits - 1)
                circ.s(q)
                if p > 0:
                    circ.depolarize1(q, p)

            elif gate_type == "hsh":
                q = random.randint(0, qubits - 1)
                gate = random.choice(["h", "sqrt_x", "sqrt_y"])
                if gate == "h":
                    circ.h(q)
                elif gate == "sqrt_x":
                    circ.sqrt_x(q)
                else:
                    circ.sqrt_y(q)
                if p > 0:
                    circ.depolarize1(q, p)

            elif gate_type == "cnot":
                if qubits < 2:
                    continue
                q1, q2 = random.sample(range(qubits), 2)
                circ.cnot(q1, q2)
                if p > 0:
                    circ.depolarize2(q1, q2, p)

        return circ

    @property
    def tcount(self):
        return zx.tcount(self.g)
