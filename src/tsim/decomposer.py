from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

import tsim.external.pyzx as zx
from tsim.compile import CompiledCircuit, compile_circuit
from tsim.external.pyzx.graph.base import BaseGraph
from tsim.stabrank import find_stab


@dataclass
class Decomposer:
    graph: BaseGraph
    output_indices: list[int]
    f_chars: list[str]
    m_chars: list[str]
    plugged_graphs: list[BaseGraph] | None = None
    compiled_circuits: list[CompiledCircuit] | None = None
    f_selection: jax.Array | None = None

    def plug_outputs(self, autoregressive: bool = True) -> list[BaseGraph]:
        graphs: list[BaseGraph] = []
        num_outputs = len(self.graph.outputs())
        for o in range(num_outputs) if autoregressive else [-1, num_outputs - 1]:
            g0 = self.graph.copy()
            output_vertices = list(g0.outputs())
            effect = "0" * (o + 1) + "+" * (num_outputs - o - 1)
            g0.apply_effect(effect)
            g0.scalar.add_power(
                (num_outputs - o - 1)
            )  # compensate power of the trace effect
            for i, v in enumerate(output_vertices[: o + 1]):
                g0.set_phase(v, self.m_chars[i])
            zx.full_reduce(g0, paramSafe=True)
            graphs.append(g0)

        self.plugged_graphs = graphs
        return graphs

    def decompose(self, autoregressive: bool = True) -> None:
        graphs = self.plug_outputs(autoregressive=autoregressive)
        circuits: list[CompiledCircuit] = []
        chars = self.f_chars + self.m_chars
        num_errors = len(self.f_chars)

        for i, graph in enumerate(graphs):
            g_copy = graph.copy()
            zx.full_reduce(g_copy, paramSafe=True)
            g_copy.normalize()
            g_list = find_stab(g_copy)
            if autoregressive:
                circuits.append(compile_circuit(g_list, num_errors + i + 1, chars))
            else:
                if i == 0:
                    # just normalization
                    circuits.append(compile_circuit(g_list, num_errors, chars))
                else:
                    # all outputs have variables
                    num_outsputs = len(self.graph.outputs())
                    circuits.append(
                        compile_circuit(g_list, num_errors + num_outsputs, chars)
                    )

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

    def decompose(self, autoregressive: bool = True) -> None:
        for component in self.components:
            component.decompose(autoregressive=autoregressive)
