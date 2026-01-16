"""Detector error model generation for QEC decoder integration."""

from collections import defaultdict

import stim


def get_detector_error_model(
    stim_circuit: stim.Circuit,
    *,
    allow_non_deterministic_observables: bool = True,
    decompose_errors: bool = False,
    flatten_loops: bool = False,
    allow_gauge_detectors: bool = False,
    approximate_disjoint_errors: bool = False,
    ignore_decomposition_failures: bool = False,
    block_decomposition_from_introducing_remnant_edges: bool = False,
) -> stim.DetectorErrorModel:
    """Return a stim.DetectorErrorModel describing the error processes in the circuit.

    Unlike the stim.Circuit.detector_error_model() method, this method allows for non-deterministic observables
    when `allow_gauge_detectors` is set to true. This is achieved by converting logical
    observables into detectors, calling the stim.Circuit.detector_error_model(allow_gauge_detectors=True), and then
    reconverting the detectors back into observables in the detector error model.

    WARNING: If the circuit has distance one, i.e. if there are errors that only flip logical observables,
    this method will return an incorrect detector error model.

    Args:
        stim_circuit: The stim circuit to compute the detector error model for.
        allow_non_deterministic_observables: Defaults to true. When set to true, the detector error model allows for
            non-deterministic observables. This is achieved by converting logical observables into detectors, calling
            the stim.Circuit.detector_error_model(allow_gauge_detectors=True), and then reconverting the detectors
            back into observables in the detector error model.
        decompose_errors: Defaults to false. When set to true, the error analysis attempts to decompose the
            components of composite error mechanisms (such as depolarization errors) into simpler errors, and
            suggest this decomposition via `stim.target_separator()` between the components. For example, in an
            XZ surface code, single qubit depolarization has a Y error term which can be decomposed into simpler
            X and Z error terms. Decomposition fails (causing this method to throw) if it's not possible to
            decompose large errors into simple errors that affect at most two detectors.
            When allow_non_deterministic_observables is set to true, decomposition is not supported.
        flatten_loops: Defaults to false. When set to true, the output will not contain any `repeat` blocks.
            When set to false, the error analysis watches for loops in the circuit reaching a periodic steady
            state with respect to the detectors being introduced, the error mechanisms that affect them, and the
            locations of the logical observables. When it identifies such a steady state, it outputs a repeat
            block. This is massively more efficient than flattening for circuits that contain loops, but creates
            a more complex output.

            Irrelevant unless allow_non_deterministic_observables=False.
        allow_gauge_detectors: Defaults to false. When set to false, the error analysis verifies that detectors
            in the circuit are actually deterministic under noiseless execution of the circuit. When set to
            true, these detectors are instead considered to be part of degrees freedom that can be removed from
            the error model. For example, if detectors D1 and D3 both anti-commute with a reset, then the error
            model has a gauge `error(0.5) D1 D3`. When gauges are identified, one of the involved detectors is
            removed from the system using Gaussian elimination.

            Note that logical observables are still verified to be deterministic, even if this option is set.
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
    if allow_non_deterministic_observables and decompose_errors:
        raise ValueError(
            "Decomposition of error mechanisms is not supported when allowing non-deterministic observables."
        )
    obs: dict[int, list[int]] = defaultdict(list)

    if not allow_non_deterministic_observables:
        return stim_circuit.detector_error_model(
            allow_gauge_detectors=allow_gauge_detectors,
            decompose_errors=decompose_errors,
            flatten_loops=flatten_loops,
            approximate_disjoint_errors=approximate_disjoint_errors,
            ignore_decomposition_failures=ignore_decomposition_failures,
            block_decomposition_from_introducing_remnant_edges=block_decomposition_from_introducing_remnant_edges,
        )

    new_circuit = stim.Circuit()

    # NOTE: stim allows multiple OBSERVABLE_INCLUDE instruction with the same index.
    # We will combine them into a single OBSERVABLE_INCLUDE instructions and
    # push them to the end of the circuit. This requires updating the rec[] indices.

    for instruction in stim_circuit.flattened():
        assert not isinstance(instruction, stim.CircuitRepeatBlock)
        if instruction.name in [
            "M",
            "MPP",
            "MR",
            "MRX",
            "MRY",
            "MRX",
            "MX",
            "MY",
            "MZ",
        ]:
            targets = instruction.targets_copy()
            if instruction.name == "MPP":
                # MPP produces one measurement per Pauli product.
                # Products are separated by spaces; within a product, Paulis are joined by '*' (combiners).
                # num_measurements = num_non_combiner_targets - num_combiners
                num_combiners = sum(1 for t in targets if t.is_combiner)
                num_meas = len(targets) - 2 * num_combiners
            else:
                num_meas = len(targets)
            for idx in obs:
                # update measurement rec indices for the OBSERVABLE_INCLUDE instructions
                obs[idx] = [t - num_meas for t in obs[idx]]

        if instruction.name == "OBSERVABLE_INCLUDE":
            assert len(instruction.gate_args_copy()) == 1
            idx = int(instruction.gate_args_copy()[0])
            target_vals = [t.value for t in instruction.targets_copy()]
            obs[idx].extend(target_vals)
        else:
            new_circuit.append_operation(
                instruction.name,
                instruction.targets_copy(),
                instruction.gate_args_copy(),
            )

    # obs combines all OBSERVABLE_INCLUDE instructions. We now add them to the end
    # of the flattened circuit as DETECTOR instructions.
    num_detectors = stim_circuit.num_detectors
    mapping: dict[int, int] = {}
    for idx, targets in obs.items():
        new_circuit.append_operation(
            "DETECTOR",
            [stim.target_rec(t) for t in targets],
        )
        # mapping from DETECTORS (D) to logical observables (L)
        mapping[num_detectors] = idx
        num_detectors += 1

    dem = new_circuit.detector_error_model(
        allow_gauge_detectors=True,
        decompose_errors=decompose_errors,
        flatten_loops=flatten_loops,
        approximate_disjoint_errors=approximate_disjoint_errors,
        ignore_decomposition_failures=ignore_decomposition_failures,
        block_decomposition_from_introducing_remnant_edges=block_decomposition_from_introducing_remnant_edges,
    )

    new_dem = stim.DetectorErrorModel()

    for instruction in dem:
        assert not isinstance(instruction, stim.DemRepeatBlock)

        new_targets = []
        new_type = instruction.type
        for t in instruction.targets_copy():
            if (
                isinstance(t, stim.DemTarget)
                and t.is_relative_detector_id()
                and t.val in mapping
            ):
                new_targets.append(stim.target_logical_observable_id(mapping[t.val]))
                if instruction.type == "detector":
                    new_type = "logical_observable"
            else:
                new_targets.append(t)

        new_instruction = stim.DemInstruction(
            new_type,
            instruction.args_copy(),
            new_targets,
        )

        # Remove gauge statements that only affect logical observables (e.g., "error(0.5) L0").
        # These arise from non-deterministic observables and should not appear in the final DEM.
        if instruction.args_copy() == [0.5]:
            # Check if all targets are logical observables (no detectors)
            all_logical = all(
                isinstance(t, stim.DemTarget) and t.is_logical_observable_id()
                for t in new_targets
            )
            if all_logical:
                continue

        new_dem.append(new_instruction)

    if new_dem.num_observables != stim_circuit.num_observables:
        raise ValueError(
            "Failed to compute detector error model. "
            "The number of observables changed after conversion. "
            "This indicates that stim has interpreted logical observables as gauges "
            f"and removed them. Error model:\n {str(new_dem)}"
        )
    return new_dem
