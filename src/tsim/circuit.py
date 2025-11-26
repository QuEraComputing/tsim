from __future__ import annotations

from typing import Any, Literal, cast

import stim

import tsim.external.pyzx as zx
from tsim.dem import get_detector_error_model
from tsim.external.pyzx.graph.base import BaseGraph
from tsim.graph_util import build_sampling_graph
from tsim.parse import parse_stim_circuit


class Circuit:
    """Quantum circuit as a thin wrapper around stim.Circuit.

    Circuits are constructed like stim circuits:

        >>> circuit = Circuit('''
        ...     H 0
        ...     S[T] 0  # This is the T instruction
        ...     CNOT 0 1
        ...     M 0 1
        ... ''')

    where `S[T]` is the T instruction.
    """

    __slots__ = ("_stim_circ",)

    def __init__(self, stim_program_text: str = ""):
        """Initialize circuit from stim program text.

        Args:
            stim_program_text: Stim program text to parse. If empty, creates an
                empty circuit.
        """
        self._stim_circ = stim.Circuit(stim_program_text)

    @classmethod
    def from_stim_program(cls, stim_circuit: stim.Circuit) -> Circuit:
        """Create a Circuit from a stim.Circuit object.

        Args:
            stim_circuit: The stim circuit to wrap.

        Returns:
            A new Circuit instance.
        """
        c = cls.__new__(cls)
        c._stim_circ = stim_circuit.flattened()
        return c

    def __repr__(self) -> str:
        return f"tsim.Circuit('''\n{self._stim_circ}\n''')"

    def __str__(self) -> str:
        return str(self._stim_circ)

    def __len__(self) -> int:
        return len(self._stim_circ)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Circuit):
            return self._stim_circ == other._stim_circ
        return NotImplemented

    def __iadd__(self, other: Circuit | stim.Circuit) -> Circuit:
        if isinstance(other, Circuit):
            self._stim_circ += other._stim_circ
        else:
            self._stim_circ += other
        return self

    def __add__(self, other: Circuit | stim.Circuit) -> Circuit:
        result = Circuit.from_stim_program(self._stim_circ.copy())
        result += other
        return result

    def __imul__(self, repetitions: int) -> Circuit:
        self._stim_circ *= repetitions
        self._stim_circ = self._stim_circ.flattened()
        return self

    def __mul__(self, repetitions: int) -> Circuit:
        return Circuit.from_stim_program(self._stim_circ * repetitions)

    def __rmul__(self, repetitions: int) -> Circuit:
        return self * repetitions

    @property
    def num_measurements(self) -> int:
        """Counts the number of bits produced when sampling the circuit's measurements."""
        return self._stim_circ.num_measurements

    @property
    def num_detectors(self) -> int:
        """Counts the number of bits produced when sampling the circuit's detectors."""
        return self._stim_circ.num_detectors

    @property
    def num_observables(self) -> int:
        """Counts the number of bits produced when sampling the circuit's logical observables.

        This is one more than the largest observable index given to OBSERVABLE_INCLUDE.
        """
        return self._stim_circ.num_observables

    @property
    def num_qubits(self) -> int:
        """Counts the number of qubits used when simulating the circuit.

        This is always one more than the largest qubit index used by the circuit.
        """
        return self._stim_circ.num_qubits

    def copy(self) -> Circuit:
        """Create a copy of this circuit."""
        return Circuit.from_stim_program(self._stim_circ.copy())

    def without_noise(self) -> Circuit:
        """Return a copy of the circuit with all noise removed."""
        return Circuit.from_stim_program(self._stim_circ.without_noise())

    def without_annotations(self) -> Circuit:
        """Return a copy of the circuit with all annotations removed."""
        circ = stim.Circuit()
        for instr in self._stim_circ:
            assert not isinstance(instr, stim.CircuitRepeatBlock)
            if instr.name in ["OBSERVABLE_INCLUDE", "DETECTOR"]:
                continue
            circ.append(instr)
        return Circuit.from_stim_program(circ)

    def detector_error_model(
        self,
        *,
        decompose_errors: bool = False,
        flatten_loops: bool = False,
        allow_gauge_detectors: bool = False,
        approximate_disjoint_errors: bool = False,
        ignore_decomposition_failures: bool = False,
        block_decomposition_from_introducing_remnant_edges: bool = False,
    ) -> stim.DetectorErrorModel:
        """Returns a stim.DetectorErrorModel describing the error processes in the circuit.

        Unlike the stim.Circuit.detector_error_model() method, this method allows for non-deterministic observables
        when `allow_gauge_detectors` is set to true.

        Args:
            decompose_errors: Defaults to false. When set to true, the error analysis attempts to decompose the
                components of composite error mechanisms (such as depolarization errors) into simpler errors, and
                suggest this decomposition via `stim.target_separator()` between the components. For example, in an
                XZ surface code, single qubit depolarization has a Y error term which can be decomposed into simpler
                X and Z error terms. Decomposition fails (causing this method to throw) if it's not possible to
                decompose large errors into simple errors that affect at most two detectors.

                This is not supported by tsim and setting it to true will raise an error. The argument is present
                for compatibility with stim.
            flatten_loops: Defaults to false. When set to true, the output will not contain any `repeat` blocks.
                When set to false, the error analysis watches for loops in the circuit reaching a periodic steady
                state with respect to the detectors being introduced, the error mechanisms that affect them, and the
                locations of the logical observables. When it identifies such a steady state, it outputs a repeat
                block. This is massively more efficient than flattening for circuits that contain loops, but creates
                a more complex output.

                Irrelevant unless allow_gauge_detectors=False.
            allow_gauge_detectors: Defaults to false. When set to false, the error analysis verifies that detectors
                in the circuit are actually deterministic under noiseless execution of the circuit.

                Note that, unlike in stim, logical observables are also allowed to be non-deterministic.
            approximate_disjoint_errors: Defaults to false. When set to false, composite error mechanisms with
                disjoint components (such as `PAULI_CHANNEL_1(0.1, 0.2, 0.0)`) can cause the error analysis to throw
                exceptions (because detector error models can only contain independent error mechanisms). When set
                to true, the probabilities of the disjoint cases are instead assumed to be independent
                probabilities. For example, a ``PAULI_CHANNEL_1(0.1, 0.2, 0.0)` becomes equivalent to an
                `X_ERROR(0.1)` followed by a `Z_ERROR(0.2)`. This assumption is an approximation, but it is a good
                approximation for small probabilities.

                This argument can also be set to a probability between 0 and 1, setting a threshold below which the
                approximation is acceptable. Any error mechanisms that have a component probability above the
                threshold will cause an exception to be thrown.
            ignore_decomposition_failures: Defaults to False.
                When this is set to True, circuit errors that fail to decompose into graphlike
                detector error model errors no longer cause the conversion process to abort.
                Instead, the undecomposed error is inserted into the output. Whatever tool
                the detector error model is then given to is responsible for dealing with the
                undecomposed errors (e.g. a tool may choose to simply ignore them).

                Irrelevant unless decompose_errors=True.
            block_decomposition_from_introducing_remnant_edges: Defaults to False.
                Requires that both A B and C D be present elsewhere in the detector error model
                in order to decompose A B C D into A B ^ C D. Normally, only one of A B or C D
                needs to appear to allow this decomposition.

                Remnant edges can be a useful feature for ensuring decomposition succeeds, but
                they can also reduce the effective code distance by giving the decoder single
                edges that actually represent multiple errors in the circuit (resulting in the
                decoder making misinformed choices when decoding).

                Irrelevant unless decompose_errors=True.
        """
        return get_detector_error_model(
            self._stim_circ,
            allow_non_deterministic_observables=True,
            decompose_errors=decompose_errors,
            flatten_loops=flatten_loops,
            allow_gauge_detectors=allow_gauge_detectors,
            approximate_disjoint_errors=approximate_disjoint_errors,
            ignore_decomposition_failures=ignore_decomposition_failures,
            block_decomposition_from_introducing_remnant_edges=block_decomposition_from_introducing_remnant_edges,
        )

    def diagram(
        self, type: Literal["pyzx", "timeline-svg"] = "pyzx", labels: bool = False
    ) -> Any:
        """Display the circuit diagram.

        Args:
            type: Diagram type - "pyzx" for ZX-diagram, "timeline-svg" for stim timeline.
            labels: Whether to show vertex labels (only for pyzx type).

        Returns:
            The graph representation.
        """
        if type == "timeline-svg":
            return self._stim_circ.diagram(type="timeline-svg")

        built = parse_stim_circuit(self._stim_circ)
        g = built.graph

        if len(g.vertices()) == 0:
            return g

        g = g.clone()
        max_row = max(g.row(v) for v in built.last_vertex.values())
        for q in built.last_vertex:
            g.set_row(built.last_vertex[q], max_row)
        zx.draw(g, labels=labels)
        return g

    def to_tensor(self) -> Any:
        """Convert circuit to tensor representation."""
        built = parse_stim_circuit(self._stim_circ)
        g = built.graph.copy()
        g.normalize()
        return g.to_tensor()

    def to_matrix(self) -> Any:
        """Convert circuit to matrix representation."""
        built = parse_stim_circuit(self._stim_circ)
        g = built.graph.copy()
        g.normalize()
        return g.to_matrix()

    def tcount(self) -> int:
        """Count the number of T gates in the circuit."""
        built = parse_stim_circuit(self._stim_circ)
        return zx.tcount(built.graph)

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

        Args:
            sample_detectors: If True, sample detectors and observables instead of
                measurements.

        Returns:
            A ZX graph for sampling.
        """
        built = parse_stim_circuit(self._stim_circ)
        return build_sampling_graph(built, sample_detectors=sample_detectors)

    def compile_sampler(self, *, seed: int | None = None):
        """Compile circuit into a measurement sampler.

        Args:
            seed: Random seed for the sampler. If None, a random seed will be generated.

        Returns:
            A CompiledMeasurementSampler that can be used to sample measurements.
        """
        from tsim.sampler import CompiledMeasurementSampler

        return CompiledMeasurementSampler(self, seed=seed)

    def compile_detector_sampler(self, *, seed: int | None = None):
        """Compile circuit into a detector sampler.

        Args:
            seed: Random seed for the sampler. If None, a random seed will be generated.

        Returns:
            A CompiledDetectorSampler that can be used to sample detectors and observables.
        """
        from tsim.sampler import CompiledDetectorSampler

        return CompiledDetectorSampler(self, seed=seed)

    def cast_to_stim(self) -> stim.Circuit:
        """Return self with type cast to stim.Circuit. This is useful for passing the circuit to functions that expect a stim.Circuit."""
        return cast(stim.Circuit, self)
