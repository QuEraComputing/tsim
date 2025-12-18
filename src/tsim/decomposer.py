from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

import tsim.external.pyzx as zx
from tsim.compile import CompiledCircuit, compile_circuit
from tsim.external.pyzx.graph.base import BaseGraph
from tsim.stabrank import find_stab

DecompositionMode = Literal["sequential", "joint"]


@dataclass
class Decomposer:
    graph: BaseGraph
    output_indices: list[int]
    f_chars: list[str]
    m_chars: list[str]
    plugged_graphs: list[BaseGraph] | None = None
    compiled_circuits: list[CompiledCircuit] | None = None
    f_selection: jax.Array | None = None

    def plug_outputs(self, outputs_to_plug: list[int]) -> list[BaseGraph]:
        """Create graphs with specified numbers of outputs plugged.

        Args:
            outputs_to_plug: List of integers specifying how many outputs to plug
                for each graph. E.g., [1, 2, 3] creates 3 graphs with 1, 2, and 3
                outputs plugged respectively.
        """
        graphs: list[BaseGraph] = []
        self.outputs_to_plug = outputs_to_plug
        num_outputs = len(self.graph.outputs())

        for num_plugged in outputs_to_plug:
            g0 = self.graph.copy()
            output_vertices = list(g0.outputs())
            effect = "0" * num_plugged + "+" * (num_outputs - num_plugged)
            g0.apply_effect(effect)
            g0.scalar.add_power(num_outputs - num_plugged)  # compensate power of trace
            for i, v in enumerate(output_vertices[:num_plugged]):
                g0.set_phase(v, self.m_chars[i])
            zx.full_reduce(g0, paramSafe=True)

            # Remove parametrized global phase terms
            g0.scalar.phasevars_halfpi = dict()
            g0.scalar.phasevars_pi_pair = []

            graphs.append(g0)

        self.plugged_graphs = graphs
        return graphs

    def decompose(self) -> None:
        """Decompose the graph into compiled circuits."""
        if self.plugged_graphs is None:
            raise ValueError("Graphs not plugged")

        graphs = self.plugged_graphs
        circuits: list[CompiledCircuit] = []
        chars = self.f_chars + self.m_chars
        num_errors = len(self.f_chars)

        power2 = 0
        for i, graph in enumerate(graphs):
            g_copy = graph.copy()
            zx.full_reduce(g_copy, paramSafe=True)
            g_copy.normalize()

            # Balance power2 of graphs to avoid over/underflow
            # TODO: this might require a more sophisticated approach for large number of T gates
            if i == 0:
                power2 = g_copy.scalar.power2
            g_copy.scalar.add_power(-power2)

            g_list = find_stab(g_copy)
            n_params = num_errors + self.outputs_to_plug[i]
            circuits.append(compile_circuit(g_list, n_params, chars))

        self.compiled_circuits = circuits
        self.f_selection = jnp.array(
            [int(f_char[1:]) for f_char in self.f_chars], dtype=jnp.int32
        )


@dataclass
class DecomposerArray:
    components: list[Decomposer]

    @property
    def output_order(self) -> jnp.ndarray:
        ord_list: list[int] = []
        for component in self.components:
            ord_list.extend(component.output_indices)
        return jnp.array(ord_list, dtype=jnp.int32)

    def decompose(self, mode: DecompositionMode = "sequential") -> None:
        """Decompose all components.

        Args:
            mode: Decomposition mode applied to each component:
                - "sequential": For sampling - [0, 1, 2, ..., n] per component
                  (includes normalization circuit for efficient chain-rule sampling)
                - "joint": For probability - [0, n] per component
        """
        for component in self.components:
            if mode == "sequential":
                # [0, 1, 2, ..., n] - includes normalization (0) for chain-rule optimization
                outputs_to_plug = list(range(len(component.graph.outputs()) + 1))
            elif mode == "joint":
                outputs_to_plug = [0, len(component.graph.outputs())]

            component.plug_outputs(outputs_to_plug)
            component.decompose()
