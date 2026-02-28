"""Quantum circuit wrapper around stim.Circuit with non-Clifford gate support."""

from __future__ import annotations

from typing import Any, Iterable, Literal, cast, overload

import pyzx_param as zx
import stim
from pyzx_param.graph.base import BaseGraph
from pyzx_param.utils import VertexType

from tsim.core.graph import build_sampling_graph
from tsim.core.parse import parse_parametric_tag, parse_stim_circuit
from tsim.noise.dem import get_detector_error_model
from tsim.utils.diagram import render_svg
from tsim.utils.program_text import shorthand_to_stim, stim_to_shorthand


class Circuit:
    """Quantum circuit as a thin wrapper around stim.Circuit.

    Circuits are constructed like stim circuits:

            >>> circuit = Circuit('''
            ...     H 0
            ...     T 0
            ...     CNOT 0 1
            ...     M 0 1
            ... ''')
    """

    __slots__ = ("_stim_circ",)

    def __init__(self, stim_program_text: str = ""):
        """Initialize circuit from stim program text.

        Args:
            stim_program_text: Stim program text to parse. If empty, creates an
                empty circuit.

        """
        self._stim_circ = stim.Circuit(shorthand_to_stim(stim_program_text)).flattened()

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

    def append_from_stim_program_text(self, stim_program_text: str) -> None:
        """Append operations described by a STIM format program into the circuit.

        Supports the same shorthand syntax as the constructor.
        """
        self._stim_circ.append_from_stim_program_text(
            shorthand_to_stim(stim_program_text)
        )
        self._stim_circ = self._stim_circ.flattened()

    @classmethod
    def from_file(cls, filename: str) -> Circuit:
        """Create a Circuit from a file.

        Args:
            filename: The filename to read the circuit from.

        Returns:
            A new Circuit instance.

        """
        with open(filename, "r", encoding="utf-8") as f:
            stim_program_text = f.read()
        stim_circ = stim.Circuit(shorthand_to_stim(stim_program_text)).flattened()
        return cls.from_stim_program(stim_circ)

    def __repr__(self) -> str:
        """Return a string representation that can recreate the circuit."""
        return f"tsim.Circuit('''\n{str(self)}\n''')"

    def __str__(self) -> str:
        """Return the circuit as program text."""
        return stim_to_shorthand(str(self._stim_circ))

    def __len__(self) -> int:
        """Return the number of instructions in the circuit."""
        return len(self._stim_circ)

    def __eq__(self, other: object) -> bool:
        """Check equality with another circuit."""
        if isinstance(other, Circuit):
            return self._stim_circ == other._stim_circ
        return NotImplemented

    def __iadd__(self, other: Circuit | stim.Circuit) -> Circuit:
        """Append another circuit to this circuit in-place."""
        if isinstance(other, Circuit):
            self._stim_circ += other._stim_circ
        else:
            self._stim_circ += other
        return self

    def __add__(self, other: Circuit | stim.Circuit) -> Circuit:
        """Return a new circuit with another circuit appended."""
        result = Circuit.from_stim_program(self._stim_circ.copy())
        result += other
        return result

    def __imul__(self, repetitions: int) -> Circuit:
        """Repeat this circuit in-place."""
        self._stim_circ *= repetitions
        self._stim_circ = self._stim_circ.flattened()
        return self

    def __mul__(self, repetitions: int) -> Circuit:
        """Return a new circuit repeated the given number of times."""
        return Circuit.from_stim_program(self._stim_circ * repetitions)

    def __rmul__(self, repetitions: int) -> Circuit:
        """Return a new circuit repeated the given number of times."""
        return self * repetitions

    @overload
    def __getitem__(
        self,
        index_or_slice: int,
    ) -> stim.CircuitInstruction: ...

    @overload
    def __getitem__(
        self,
        index_or_slice: slice,
    ) -> Circuit: ...

    def __getitem__(
        self,
        index_or_slice: object,
    ) -> object:
        """Return copies of instructions from the circuit.

        Args:
            index_or_slice: An integer index picking out an instruction to return, or a
                slice picking out a range of instructions to return as a circuit.

        Returns:
            If the index was an integer, then an instruction from the circuit.
            If the index was a slice, then a circuit made up of the instructions in that
            slice.

        """
        if isinstance(index_or_slice, int):
            return self._stim_circ[index_or_slice]
        elif isinstance(index_or_slice, slice):
            return Circuit.from_stim_program(self._stim_circ[index_or_slice])
        else:
            raise TypeError(f"Invalid index or slice: {index_or_slice}")

    def approx_equals(
        self,
        other: object,
        *,
        atol: float,
    ) -> bool:
        """Check if a circuit is approximately equal to another circuit.

        Two circuits are approximately equal if they are equal up to slight
        perturbations of instruction arguments such as probabilities. For example,
        `X_ERROR(0.100) 0` is approximately equal to `X_ERROR(0.099)` within an absolute
        tolerance of 0.002. All other details of the circuits (such as the ordering of
        instructions and targets) must be exactly the same.

        Args:
            other: The circuit, or other object, to compare to this one.
            atol: The absolute error tolerance. The maximum amount each probability may
                have been perturbed by.

        Returns:
            True if the given object is a circuit approximately equal up to the
            receiving circuit up to the given tolerance, otherwise False.

        """
        if isinstance(other, Circuit):
            return self._stim_circ.approx_equals(other._stim_circ, atol=atol)
        elif isinstance(other, stim.Circuit):
            return self._stim_circ.approx_equals(other, atol=atol)
        else:
            return False

    def compile_m2d_converter(
        self,
        *,
        skip_reference_sample: bool = False,
    ) -> stim.CompiledMeasurementsToDetectionEventsConverter:
        """Create a measurement-to-detection-event converter for the given circuit.

        The converter can efficiently compute detection events and observable flips
        from raw measurement data.

        The converter uses a noiseless reference sample, collected from the circuit
        using stim's Tableau simulator during initialization of the converter, as a
        baseline for determining what the expected value of a detector is.

        Note that the expected behavior of gauge detectors (detectors that are not
        actually deterministic under noiseless execution) can vary depending on the
        reference sample. Stim mitigates this by always generating the same reference
        sample for a given circuit.

        Args:
            skip_reference_sample: Defaults to False. When set to True, the reference
                sample used by the converter is initialized to all-zeroes instead of
                being collected from the circuit. This should only be used if it's known
                that the all-zeroes sample is actually a possible result from the
                circuit (under noiseless execution).

        Returns:
            An initialized stim.CompiledMeasurementsToDetectionEventsConverter.

        """
        return self._stim_circ.compile_m2d_converter(
            skip_reference_sample=skip_reference_sample
        )

    @property
    def stim_circuit(self) -> stim.Circuit:
        """Return the underlying stim circuit."""
        return self._stim_circ.copy()

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

    @property
    def num_ticks(
        self,
    ) -> int:
        """Counts the number of TICK instructions executed when running the circuit.

        TICKs in loops are counted once per iteration.

        Returns:
            The number of ticks executed by the circuit.

        """
        return self._stim_circ.num_ticks

    def pop(
        self,
        index: int = -1,
    ) -> stim.CircuitInstruction:
        """Pops an operation from the end of the circuit, or at the given index.

        Args:
            index: Defaults to -1 (end of circuit). The index to pop from.

        Returns:
            The popped instruction.

        Raises:
            IndexError: The given index is outside the bounds of the circuit.

        """
        el = self._stim_circ.pop(index)
        assert not isinstance(el, stim.CircuitRepeatBlock)
        return el

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
        """Return a stim.DetectorErrorModel describing the error processes in the circuit.

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
        self,
        type: Literal[
            "pyzx",
            "pyzx-dets",
            "pyzx-meas",
            "timeline-svg",
            "timeslice-svg",
        ] = "timeline-svg",
        tick: int | range | None = None,
        filter_coords: Iterable[Iterable[float] | stim.DemTarget] = ((),),
        rows: int | None = None,
        height: float | None = None,
        width: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a diagram of the circuit, from a variety of options.

        Args:
            type: The type of diagram. Available types are:
                "pyzx": A pyzx SVG of the ZX diagram of the circuit.
                "pyzx-dets": A pyzx SVG of the ZX diagram that is used to compute
                    probabilities of detectors and observables.
                "pyzx-meas": A pyzx SVG of the ZX diagram that is used to compute
                    probabilities of measurements.
                "timeline-svg": An SVG image of the operations applied by
                    the circuit over time. Includes annotations showing the
                    measurement record index that each measurement writes
                    to, and the measurements used by detectors.
                "timeslice-svg": An SVG image of the operations applied
                    between two TICK instructions in the circuit, with the
                    operations laid out in 2d.
            tick: Required for time slice diagrams. Specifies
                which TICK instruction, or range of TICK instructions, to
                slice at. Note that the first TICK instruction in the
                circuit corresponds tick=1. The value tick=0 refers to the
                very start of the circuit.

                Passing `range(A, B)` for a detector slice will show the
                slices for ticks A through B including A but excluding B.

                Passing `range(A, B)` for a time slice will show the
                operations between tick A and tick B.
            rows: In diagrams that have multiple separate pieces, such as timeslice
                diagrams and detslice diagrams, this controls how many rows of
                pieces there will be. If not specified, a number of rows that creates
                a roughly square layout will be chosen.
            filter_coords: A list of things to include in the diagram. Different
                effects depending on the diagram.

                For detslice diagrams, the filter defaults to showing all detectors
                and no observables. When specified, each list entry can be a collection
                of floats (detectors whose coordinates start with the same numbers will
                be included), a stim.DemTarget (specifying a detector or observable
                to include), a string like "D5" or "L0" specifying a detector or
                observable to include.
            height: Optional height for the rendered diagram.
            width: Optional width for the rendered diagram.
            **kwargs: Additional keyword arguments passed to the underlying diagram renderer.

        Returns:
            An object whose `__str__` method returns the diagram, so that
            writing the diagram to a file works correctly. The returned
            object may also define methods such as `_repr_html_`, so that
            ipython notebooks recognize it can be shown using a specialized
            viewer instead of as raw text.

        """
        if type in [
            "timeline-svg",
            "timeslice-svg",
        ]:
            return render_svg(
                self._stim_circ,
                type,
                tick=tick,
                filter_coords=filter_coords,
                rows=rows,
                width=width,
                height=height,
            )
        elif type == "pyzx":
            from tsim.core.graph import scale_horizontally

            built = parse_stim_circuit(self._stim_circ, track_classical_wires=True)
            g = built.graph

            if len(g.vertices()) == 0:
                return g

            g = g.clone()
            max_row = max(g.row(v) for v in built.last_vertex.values())
            for q in built.last_vertex:
                g.set_row(built.last_vertex[q], max_row)

            for v in list(g.vertices()):
                phase_vars = g._phaseVars[v]
                if len(phase_vars) != 1:
                    continue
                phase = list(phase_vars)[0]
                if phase.startswith("det") or phase.startswith("obs"):
                    row = g.row(v)
                    vb = g.add_vertex(
                        VertexType.BOUNDARY,
                        qubit=-2,
                        row=row,
                    )
                    g.add_edge((v, vb))

            if kwargs.get("scale_horizontally", False):
                scale_horizontally(g, kwargs.pop("scale_horizontally", 1.0))
            zx.draw(g, **kwargs)
            return g
        elif type in ["pyzx-dets", "pyzx-meas"]:
            from tsim.core.graph import (
                scale_horizontally,
                squash_graph,
                transform_error_basis,
            )

            g = self.get_sampling_graph(sample_detectors=type == "pyzx-dets")
            zx.full_reduce(g, paramSafe=True)
            if kwargs.pop("transform_error_basis", True):
                g, _ = transform_error_basis(g)
            squash_graph(g)
            zx.simplify.to_rg(g)
            if kwargs.get("scale_horizontally", False):
                scale_horizontally(g, kwargs.pop("scale_horizontally", 1.0))
            zx.draw(g, **kwargs)
            return g
        else:
            return self._stim_circ.diagram(type=type, **kwargs)  # pragma: no cover

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

    def get_graph(self) -> BaseGraph:
        """Construct the ZX graph."""
        built = parse_stim_circuit(self._stim_circ)
        return built.graph

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

        Connected components whose single output is deterministically given by
        one f-variable are handled via a fast direct path (no compilation or
        autoregressive sampling).  Remaining components go through the full
        compilation pipeline.

        Args:
            seed: Random seed for the sampler. If None, a random seed will be generated.

        Returns:
            A CompiledDetectorSampler that can be used to sample detectors and observables.

        """
        from tsim.sampler import CompiledDetectorSampler

        return CompiledDetectorSampler(self, seed=seed)

    def cast_to_stim(self) -> stim.Circuit:
        """Return self with type cast to stim.Circuit. This is useful for passing the circuit to functions that expect a stim.Circuit."""
        return cast(stim.Circuit, self)  # pragma: no cover

    def inverse(self) -> Circuit:
        """Return the inverse of the circuit."""
        inv_stim_raw = self._stim_circ.inverse()

        # Stim will only invert Clifford gates (and S[T] / S_DAG[T])
        # Post-process to fix non-Clifford rotation gates stored as I[tag]
        inv_stim = stim.Circuit()
        for instr in inv_stim_raw:
            assert not isinstance(instr, stim.CircuitRepeatBlock)
            name = instr.name
            tag = instr.tag
            targets = [t.value for t in instr.targets_copy()]
            args = instr.gate_args_copy()

            if name == "I" and tag:
                result = parse_parametric_tag(tag)
                if result is not None:
                    gate_name, params = result
                    if gate_name == "U3":
                        # U3(θ, φ, λ)⁻¹ = U3(-θ, -λ, -φ)
                        theta = float(-params["theta"])
                        phi = float(-params["lambda"])
                        lam = float(-params["phi"])
                        new_tag = f"U3(theta={theta}*pi, phi={phi}*pi, lambda={lam}*pi)"
                    else:
                        theta = float(-params["theta"])
                        new_tag = f"{gate_name}(theta={theta}*pi)"
                    inv_stim.append("I", targets, args, tag=new_tag)
                    continue

            # All other instructions are correct from stim's inverse
            inv_stim.append(instr)

        return Circuit.from_stim_program(inv_stim)
