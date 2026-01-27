"""Core data types for the tsim compilation and sampling pipeline.

This module defines immutable data structures that represent the different
stages of circuit compilation:

1. SamplingGraph: Result of parsing and reducing a circuit graph
2. CompiledComponent: A single compiled connected component
3. CompiledProgram: The full compiled circuit ready for sampling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from pyzx_param.graph.base import BaseGraph

    from tsim.compile.compile import CompiledScalarGraphs


@dataclass(frozen=True)
class SamplingGraph:
    """Result of the graph preparation phase for sampling.

    Contains all data structures needed for sampling. This represents a circuit
    that has been:
    1. Parsed from stim format
    2. Converted to a ZX graph
    3. Doubled (composed with adjoint)
    4. Reduced via zx.full_reduce
    5. Had its error basis transformed (Gaussian elimination: e → f)

    Attributes:
        graph: The prepared ZX graph with f-parameters on vertices.
        error_transform: Binary matrix of shape (num_f, num_e) where entry [i, j] = 1
            means f_i depends on e_j (i.e., f_i = XOR of e_j where matrix[i,j] = 1).
        channel_probs: List of probability arrays for error channels.
        num_outputs: Number of output vertices (measurements or detectors).
        num_detectors: Number of detector vertices.

    """

    graph: BaseGraph
    error_transform: np.ndarray
    channel_probs: list[np.ndarray]
    num_outputs: int
    num_detectors: int


class CompiledComponent(eqx.Module):
    """A single compiled connected component of a circuit.

    Each component is independent and can be sampled separately. The results
    are then combined according to output_indices.

    Attributes:
        output_indices: Which global output indices this component owns.
            Used to reassemble component outputs into the final result.
        f_selection: Indices into the global f_params array to select this
            component's required f-parameters. Shape: (num_f_for_component,)
        compiled_scalar_graphs: Compiled circuits for sampling. For sequential mode:
            - compiled_scalar_graphs[0]: Normalization (no outputs plugged)
            - compiled_scalar_graphs[i]: First i outputs plugged
            For joint mode:
            - compiled_scalar_graphs[0]: Normalization
            - compiled_scalar_graphs[1]: All outputs plugged

    """

    output_indices: tuple[int, ...] = eqx.field(static=True)
    f_selection: Array
    compiled_scalar_graphs: tuple[CompiledScalarGraphs, ...]


@dataclass(frozen=True)
class CompiledProgram:
    """A fully compiled circuit program ready for sampling.

    This is the result of compiling a SamplingGraph and contains everything
    needed to sample from the circuit.

    Attributes:
        components: The compiled components, sorted by number of outputs.
        output_order: Array for reordering component outputs to final order.
            final_samples = combined[:, np.argsort(output_order)]
        num_outputs: Total number of outputs across all components.
        num_f_params: Total number of f-parameters.
        num_detectors: Number of detector outputs (for detector sampling).

    """

    components: tuple[CompiledComponent, ...]
    output_order: Array
    num_outputs: int
    num_f_params: int
    num_detectors: int
