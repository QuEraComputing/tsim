r"""Write the circuits of Gidney, Shutty and Jones's T-vs-S figure (arXiv:2409.17595, Fig. "t-vs-s").

Cultivation without an escape stage at color-code distance 3 (unitary injection, then
cultivation), under uniform depolarizing noise, built with the authors' generator. Their
circuits are Clifford proxies in the Y basis: every T gate is written as S. As the authors'
vector-simulator sampler (``VecInterceptSampler``) does, each S becomes T^k and the final
Y-basis MPP is rotated by T^k: k = 1 cultivates a T state, k = 2 an S state, k = 4 a Z state.

Usage, with the authors' code (https://github.com/Strilanc/magic-state-cultivation, ``src/``)
and its dependencies:

    uv run --no-project --with stim --with sinter --with pymatching --with pygltflib \
        --with scipy --with matplotlib python make_circuits.py PATH/TO/magic-state-cultivation/src

It writes ``circuits/``: the noisy circuits of the figure, and a noiseless one per gate in
which every detector and the observable are 0.
"""

import pathlib
import sys

import stim

sys.path.insert(0, sys.argv[1])
import gen
from cultiv import make_inject_and_cultivate_circuit

NOISE = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
POWERS = {"T": 1, "S": 2, "Z": 4}
OUT = pathlib.Path(__file__).parent / "circuits"


def with_power_of_t(circuit: stim.Circuit, k: int) -> str:
    """Return ``circuit`` as Tsim text, with each S as T^k and the Y-basis MPPs rotated."""
    lines = []
    for inst in circuit:
        assert not isinstance(inst, stim.CircuitRepeatBlock)
        targets = [t.qubit_value for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name in ("S", "S_DAG"):
            gate = "T" if inst.name == "S" else "T_DAG"
            lines += [f"{gate} {' '.join(map(str, targets))}"] * k
        elif inst.name == "MPP":
            # NOTE: one MPP per product, so that a Y product can be measured as
            # T^k (S Y S^-1) T^-k: Y rotated by k eighth-turns about Z.
            for group in inst.target_groups():
                one = stim.CircuitInstruction(
                    "MPP", _joined(group), inst.gate_args_copy()
                )
                if not all(t.is_y_target for t in group):
                    lines.append(str(one))
                    continue
                qubits = " ".join(str(t.qubit_value) for t in group)
                lines += [f"T_DAG {qubits}"] * k + [f"S {qubits}", str(one)]
                lines += [f"S_DAG {qubits}"] + [f"T {qubits}"] * k
        else:
            lines.append(str(inst))
    return "\n".join(lines) + "\n"


def _joined(group: list[stim.GateTarget]) -> list[stim.GateTarget]:
    """Return the targets of one Pauli product, joined with combiners."""
    joined = []
    for target in group:
        joined += [target, stim.target_combiner()]
    return joined[:-1]


def main() -> None:
    """Write the circuits to ``circuits/``."""
    OUT.mkdir(exist_ok=True)
    proxy = make_inject_and_cultivate_circuit(
        dcolor=3, inject_style="unitary", basis="Y"
    )
    for p in NOISE:
        noisy = gen.NoiseModel.uniform_depolarizing(
            p
        ).noisy_circuit_skipping_mpp_boundaries(proxy)
        for gate, k in POWERS.items():
            path = OUT / f"cultivate_d3_{gate}_p{p}.stim"
            path.write_text(with_power_of_t(noisy.flattened(), k))
    noiseless = {g: with_power_of_t(proxy.flattened(), k) for g, k in POWERS.items()}
    for gate, text in noiseless.items():
        (OUT / f"cultivate_d3_{gate}_noiseless.stim").write_text(text)
    print(f"wrote {len(list(OUT.glob('*.stim')))} circuits to {OUT}")


if __name__ == "__main__":
    main()
