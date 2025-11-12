import copy
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import wraps
from typing import Any, Iterable

import jax
import stim
from jax import Array

import tsim.external.pyzx as zx
from tsim.channels import (
    Depolarize1,
    Depolarize2,
    Error,
    ErrorSampler,
    PauliChannel1,
    PauliChannel2,
)
from tsim.external.pyzx import EdgeType, VertexType
from tsim.external.pyzx.graph.base import BaseGraph


def accepts_qubit_list(func=None, *, num_qubits=1):
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

    def decorator(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            # Check if the first argument is an iterable (but not a string)
            if args and isinstance(args[0], Iterable) and not isinstance(args[0], str):
                qubit_list = list(args[0])
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


@dataclass
class SamplingGraphs:
    """Container for compiled ZX-graphs used in sampling."""

    graphs: list[BaseGraph]
    num_errors: int
    chars: list[str]
    error_sampler: ErrorSampler | None = None


class Circuit:
    """Quantum circuit represented as a ZX-diagram with noise support."""

    def __init__(self, key: Array | None = None):
        """Initialize an empty circuit with optional random key for noise sampling."""
        if key is None:
            key = jax.random.key(0)
        self.key = key
        self.g = zx.Graph()
        self.last_vertex: dict[int, int] = {}
        self.last_row: dict[int, int] = {}
        self.errors = []

        self.rec = []
        self.detectors = []
        self._observables_dict: dict[int, int] = {}  # idx: vertex
        self.num_error_bits = 0
        self.qubit_to_input = {}

    @property
    def observables(self) -> list[int]:
        return [self._observables_dict[i] for i in sorted(self._observables_dict)]

    def _last_row(self, qubit: int):
        return self.g.row(self.last_vertex[qubit])

    def _last_edge(self, qubit: int):
        edges = self.g.incident_edges(self.last_vertex[qubit])
        assert len(edges) == 1
        return edges[0]

    def _add_dummy(self, qubit: int, row: float | int | None = None):
        if row is None:
            row = self._last_row(qubit) + 1
        v1 = self.g.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=row)  # type: ignore[arg-type]
        self.last_vertex[qubit] = v1
        return v1

    def _add_lane(self, qubit: int):
        v1 = self.g.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=0)
        self.qubit_to_input[qubit] = v1
        v2 = self.g.add_vertex(VertexType.BOUNDARY, qubit=qubit, row=1)
        self.g.add_edge((v1, v2))
        self.last_vertex[qubit] = v2
        return v1

    @accepts_qubit_list
    def h(self, qubit: int):
        """Apply Hadamard gate to qubit(s)."""
        g = self.g
        if qubit not in self.last_vertex:
            self._add_lane(qubit)

        e = self._last_edge(qubit)
        g.set_edge_type(
            e,
            EdgeType.HADAMARD if g.edge_type(e) == EdgeType.SIMPLE else EdgeType.SIMPLE,
        )

    @accepts_qubit_list
    def x_phase(self, qubit: int, phase: Fraction):
        g = self.g
        if qubit not in self.last_vertex:
            self._add_lane(qubit)
        v1 = self.last_vertex[qubit]
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
        if qubit not in self.last_vertex:
            self._add_lane(qubit)
        v1 = self.last_vertex[qubit]
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
        # self.z_phase(qubit, Fraction(3, 2))
        # self.x_phase(qubit, Fraction(1, 2))
        # self.z_phase(qubit, Fraction(1, 2))
        self.z(qubit)
        self.h(qubit)
        self.g.scalar.add_phase(Fraction(1, 4))

    @accepts_qubit_list
    def sqrt_y_dag(self, qubit: int):
        # self.z_phase(qubit, Fraction(3, 2))
        # self.x_phase(qubit, Fraction(-1, 2))
        # self.z_phase(qubit, Fraction(1, 2))
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
        if qubit not in self.last_vertex:
            v1 = self._add_lane(qubit)
            g.set_type(v1, VertexType.X)
        else:
            r = self._last_row(qubit)
            v1 = self.last_vertex[qubit]
            g.set_type(v1, VertexType.X)
            v2 = list(g.neighbors(v1))[0]
            g.remove_edge((v1, v2))
            v3 = self._add_dummy(qubit, r + 1)

            g.add_edge((v1, v3))

    @accepts_qubit_list
    def rx(self, qubit: int):
        self.r(qubit)
        self.h(qubit)

    @accepts_qubit_list
    def mr(self, qubit: int, p: float = 0):
        """Measure and reset qubit(s) (optionally noisy)"""
        if p > 0:
            self.x_error(qubit, p)
        self.m(qubit, p=p)
        self.r(qubit)

    @accepts_qubit_list
    def m(self, qubit: int, p: float = 0):
        """Measure qubit(s) in Z basis (optionally noisy)"""
        if p > 0:
            self.x_error(qubit, p)
        g = self.g
        if qubit not in self.last_vertex:
            self._add_lane(qubit)
        v1 = self.last_vertex[qubit]
        g.set_type(v1, VertexType.Z)
        g.set_phase(v1, f"rec[{len(self.rec)}]")
        self.rec.append(v1)
        v2 = self._add_dummy(qubit)
        g.add_edge((v1, v2))

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
        if control not in self.last_vertex:
            self._add_lane(control)
        if target not in self.last_vertex:
            self._add_lane(target)

        lr1 = self._last_row(control)
        lr2 = self._last_row(target)
        r = max(lr1, lr2)

        v1 = self.last_vertex[control]
        v2 = self.last_vertex[target]
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

    cx = cnot

    @accepts_qubit_list(num_qubits=2)
    def cz(self, control: int, target: int):
        """Apply CZ gate between control and target qubit(s)."""
        g = self.g
        if control not in self.last_vertex:
            self._add_lane(control)
        if target not in self.last_vertex:
            self._add_lane(target)

        lr1 = self._last_row(control)
        lr2 = self._last_row(target)
        r = max(lr1, lr2)

        v1 = self.last_vertex[control]
        v2 = self.last_vertex[target]
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
        if qubit not in self.last_vertex:
            self._add_lane(qubit)
        g = self.g
        v1 = self.last_vertex[qubit]
        v2 = self._add_dummy(qubit)
        g.add_edge((v1, v2))

        g.set_type(v1, error_type)  # type: ignore[arg-type]
        g.set_phase(v1, phase)  # type: ignore[arg-type]

    @accepts_qubit_list
    def x_error(self, qubit: int, p: float):
        """Apply X error with probability p to qubit(s)."""
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(Error(p, subkey))
        self._error(qubit, VertexType.X, f"e{self.num_error_bits}")
        self.num_error_bits += 1

    @accepts_qubit_list
    def z_error(self, qubit: int, p: float):
        """Apply Z error with probability p to qubit(s)."""
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(Error(p, subkey))
        self._error(qubit, VertexType.Z, f"e{self.num_error_bits}")
        self.num_error_bits += 1

    @accepts_qubit_list
    def y_error(self, qubit: int, p: float):
        """Apply Y error with probability p to qubit(s)."""
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(Error(p, subkey))
        self._error(qubit, VertexType.Z, f"e{self.num_error_bits}")
        self._error(qubit, VertexType.X, f"e{self.num_error_bits}")
        self.num_error_bits += 1

    @accepts_qubit_list
    def depolarize1(self, qubit: int, p: float):
        """Apply single-qubit depolarizing noise with probability p."""
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(Depolarize1(p, subkey))
        self._error(qubit, VertexType.Z, f"e{self.num_error_bits}")
        self._error(qubit, VertexType.X, f"e{self.num_error_bits + 1}")
        self.num_error_bits += 2

    @accepts_qubit_list
    def pauli_channel_1(self, qubit: int, px: float, py: float, pz: float):
        """Apply single-qubit Pauli channel with specified error probabilities."""
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(PauliChannel1(px, py, pz, subkey))
        self._error(qubit, VertexType.Z, f"e{self.num_error_bits}")
        self._error(qubit, VertexType.X, f"e{self.num_error_bits + 1}")
        self.num_error_bits += 2

    @accepts_qubit_list(num_qubits=2)
    def depolarize2(self, qubit_i: int, qubit_j: int, p: float):
        """Apply two-qubit depolarizing noise with probability p."""
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(Depolarize2(p, subkey))
        self._error(qubit_i, VertexType.Z, f"e{self.num_error_bits}")
        self._error(qubit_i, VertexType.X, f"e{self.num_error_bits + 1}")
        self._error(qubit_j, VertexType.Z, f"e{self.num_error_bits + 2}")
        self._error(qubit_j, VertexType.X, f"e{self.num_error_bits + 3}")
        self.num_error_bits += 4

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
        self.key, subkey = jax.random.split(self.key)
        self.errors.append(
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
        self._error(qubit_i, VertexType.Z, f"e{self.num_error_bits}")
        self._error(qubit_i, VertexType.X, f"e{self.num_error_bits + 1}")
        self._error(qubit_j, VertexType.Z, f"e{self.num_error_bits + 2}")
        self._error(qubit_j, VertexType.X, f"e{self.num_error_bits + 3}")
        self.num_error_bits += 4

    @accepts_qubit_list
    def i(self, qubit):
        if qubit not in self.last_vertex:
            self._add_lane(qubit)
        v = self.last_vertex[qubit]
        self.g.set_row(v, self._last_row(qubit) + 1)

    def tick(self):
        """Add a tick to the circuit."""
        if len(self.last_vertex) == 0:
            return
        r = max(self._last_row(q) for q in self.last_vertex)
        for q in self.last_vertex:
            self.g.set_row(self.last_vertex[q], r)

    def diagram(self, labels=False):
        """Display the ZX-diagram representation of the circuit."""
        if len(self.g.vertices()) == 0:
            return
        g = self.g.copy()
        max_row = max(self.g.row(v) for v in self.last_vertex.values())
        for q in self.last_vertex:
            g.set_row(self.last_vertex[q], max_row)
        zx.draw(g, labels=labels)

    def detector(self, rec: list[int], *args: Any):
        """Add a detector that checks parity of measurement records."""
        r = min(set([self.g.row(self.rec[r]) for r in rec])) - 0.5
        d_rows = set([self.g.row(d) for d in self.detectors + self.observables])
        while r in d_rows:
            r += 1
        v0 = self.g.add_vertex(
            VertexType.X, qubit=-1, row=r, phase=f"det[{len(self.detectors)}]"  # type: ignore[arg-type]
        )
        for rec_ in rec:
            self.g.add_edge((v0, self.rec[rec_]))
        self.detectors.append(v0)

    def observable_include(self, rec: list[int], idx: int):
        """Include measurement records in a logical observable."""
        idx = int(idx)

        if idx not in self._observables_dict:
            r = min(set([self.g.row(self.rec[r]) for r in rec])) - 0.5
            d_rows = set([self.g.row(d) for d in self.detectors + self.observables])
            while r in d_rows:
                r += 1
            v0 = self.g.add_vertex(
                VertexType.X, qubit=-1, row=r, phase=f"obs[{idx}]"  # type: ignore[arg-type]
            )
            self._observables_dict[idx] = v0

        v0 = self._observables_dict[idx]
        for rec_ in rec:
            self.g.add_edge((v0, self.rec[rec_]))

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

    def compile_detector_graphs(self) -> SamplingGraphs:
        """Compile circuit into graphs for detector sampling."""
        g = self.g.copy()

        # clean up last row
        max_row = max(self.g.row(v) for v in self.last_vertex.values())
        for q in self.last_vertex:
            g.set_row(self.last_vertex[q], max_row)

        outputs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
        g.set_outputs(tuple(outputs))  # type: ignore[arg-type]

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
            if "det" in phase or "obs" in phase or "rec" in phase:
                label_to_vertex[phase].append(v)
            if "det" in phase or "obs" in phase:
                annotation_to_vertex[phase].append(v)

        # connect all rec[i] vertices to each other and remove lables
        for label, vertices in label_to_vertex.items():
            if "rec" not in label:
                continue
            assert len(vertices) == 2
            v0, v1 = vertices
            if not g.connected(v0, v1):
                g.add_edge((v0, v1))
            g.set_phase(v0, 0)
            g.set_phase(v1, 0)

        # remove the duplicated set of detectors and observables
        for vertices in annotation_to_vertex.values():
            assert len(vertices) == 2
            g.remove_vertex(vertices.pop())

        # build the list of measurement graphs
        g_obs_list = []

        # TODO: for now we assumed detectors are deterministic, so we remove them
        for label, annotations in annotation_to_vertex.items():
            if "det" in label:
                assert len(annotations) == 1
                g.remove_vertex(annotations.pop())

        for i in sorted(self._observables_dict.keys())[::-1]:
            g_obs_list.append(g.copy())
            obs_vertices = annotation_to_vertex[f"obs[{i}]"]
            assert len(obs_vertices) == 1
            v0 = obs_vertices[0]
            g.remove_vertex(v0)

        chars = [f"e{i}" for i in range(self.num_error_bits)] + [
            f"obs[{i}]" for i in sorted(self._observables_dict.keys())
        ]
        return SamplingGraphs(
            graphs=g_obs_list[::-1],
            num_errors=self.num_error_bits,
            chars=chars,
            error_sampler=ErrorSampler(self.errors),
        )

    def compile_sampling_graphs(self) -> SamplingGraphs:
        """Compile circuit into graphs for measurement sampling."""
        g = self.g.copy()

        # clean up last row
        max_row = max(self.g.row(v) for v in self.last_vertex.values())
        for q in self.last_vertex:
            g.set_row(self.last_vertex[q], max_row)

        num_measurements = len(self.rec)
        outputs = [v for v in g.vertices() if g.type(v) == VertexType.BOUNDARY]
        g.set_outputs(tuple(outputs))  # type: ignore[arg-type]

        g_adj = g.adjoint()
        g.compose(g_adj)

        g = g.copy()

        label_to_vertex: dict[str, list[int]] = defaultdict(list)
        annotation_to_vertex: dict[str, list[int]] = defaultdict(list)
        rec_to_vertex: dict[str, int] = {}
        for v in g.vertices():
            phase_vars = g._phaseVars[v]
            if not len(phase_vars) == 1:
                continue
            phase = list(phase_vars)[0]
            if "det" in phase or "obs" in phase or "rec" in phase:
                label_to_vertex[phase].append(v)
            if "det" in phase or "obs" in phase:
                annotation_to_vertex[phase].append(v)

        # connect all rec[i] vertices to each other and add red vertex with rec[i] label
        i = 0
        for label, vertices in label_to_vertex.items():
            if "rec" not in label:
                continue
            assert len(vertices) == 2
            v0, v1 = vertices
            if not g.connected(v0, v1):
                g.add_edge((v0, v1))
            g.set_phase(v0, 0)
            g.set_phase(v1, 0)
            v3 = g.add_vertex(VertexType.X, qubit=-1, row=i + 1, phase=label)  # type: ignore[arg-type]
            rec_to_vertex[label] = v3
            g.add_edge((v0, v3))
            i += 1

        # remove detectors and observables
        for vertices in annotation_to_vertex.values():
            assert len(vertices) == 2
            for v in vertices:
                g.remove_vertex(v)

        # build the list of measurement graphs
        g_obs_list = []
        for i in range(num_measurements):
            g_obs_list.append(g.copy())
            rec_vertex = rec_to_vertex[f"rec[{num_measurements -1 -i}]"]
            g.remove_vertex(rec_vertex)

        chars = [f"e{i}" for i in range(self.num_error_bits)] + [
            f"rec[{i}]" for i in range(num_measurements)
        ]
        return SamplingGraphs(
            graphs=g_obs_list[::-1],
            num_errors=self.num_error_bits,
            chars=chars,
            error_sampler=ErrorSampler(self.errors),
        )

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
        c.errors = []
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
            if "rec" in label:
                g.set_phase(vertices[0], 0)

        c.errors = []
        c.detectors = []
        c._observables_dict = {}
        c.rec = []
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

            if name == "s" and replace_s_with_t:
                name = "t"
            if name == "s_dag" and replace_s_with_t:
                name = "t_dag"

            if name == "tick":
                self.tick()
                continue
            if name == "mpp":
                # TODO: improve parsing
                args = str(instruction).split(" ")[1:]
                self.mpp(args)
                continue

            targets = [t.value for t in instruction.targets_copy()]  # type: ignore[attr-defined]
            args = instruction.gate_args_copy()  # type: ignore[attr-defined]
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
        from tsim.sampler import Sampler

        graphs = self.compile_sampling_graphs()
        return Sampler(graphs)

    def compile_detector_sampler(self):
        """Compile circuit into a detector sampler."""
        from tsim.sampler import Sampler

        graphs = self.compile_detector_graphs()
        return Sampler(graphs)
