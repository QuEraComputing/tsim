[![Docs](https://img.shields.io/badge/docs-latest-blue?style=flat&logo=mintlify)](https://bloqade.quera.com/tsim)
[![Coverage](https://img.shields.io/codecov/c/github/QuEraComputing/tsim?style=flat&logo=codecov)](https://codecov.io/gh/QuEraComputing/tsim)
[![arXiv](https://img.shields.io/badge/arXiv-2604.01059-b31b1b.svg?style=flat&logo=arxiv)](https://arxiv.org/abs/2604.01059)
[![Like This? Leave a star](https://img.shields.io/github/stars/QuEraComputing/tsim?style=flat&label=Like%20Tsim%3F%20Leave%20a%20star&color=yellow&logo=github)](https://github.com/QuEraComputing/tsim)

# Tsim

A fast sampler for noisy quantum circuits with non-Clifford gates, built on the ZX-calculus.
Tsim feels just like [Stim](https://github.com/quantumlib/Stim), but supports `T` gates, arbitrary rotations, and
other non-Clifford instructions.


## Installation

Tsim requires Python 3.11 or later. Install with pip:
```bash
pip install bloqade-tsim
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add bloqade-tsim
```

The default sampling strategy runs on the CPU. To run the stabilizer-rank strategies on an NVIDIA GPU,
install the JAX CUDA extra:
```bash
pip install "bloqade-tsim[cuda13]"   # or bloqade-tsim[cuda12]
```

### Using Tsim with coding agents

Tsim ships an [agent skill](skills/simulating-circuits-with-tsim/SKILL.md) that teaches coding
agents (Claude Code, Codex, Cursor and others) how to judge whether a circuit is feasible,
sample and postselect it, run logical error rate studies, and port published Stim circuits.
Install it with:
```bash
npx skills add QuEraComputing/tsim --skill simulating-circuits-with-tsim
```

The docs also run an MCP server that lets agents search them:
```bash
claude mcp add --transport http tsim https://bloqade.quera.com/tsim/mcp
```

## Quick Start
An introductory tutorial is available [here](https://bloqade.quera.com/tsim/tutorials/encoding_demo). Please also refer to the [documentation](https://bloqade.quera.com/tsim).

For many existing scripts, replacing `stim` with `tsim` should just work. Tsim mirrors the Stim API and supports all [Stim instructions](https://github.com/quantumlib/Stim/blob/v1.16.0/doc/gates.md), plus the [non-Clifford instructions](#supported-instructions) listed below.

```python
import tsim

c = tsim.Circuit(
    """
    RX 0
    R 1
    T 0
    PAULI_CHANNEL_1(0.1, 0.1, 0.2) 0 1
    H 0
    CNOT 0 1
    DEPOLARIZE2(0.01) 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    """
)

detector_sampler = c.compile_detector_sampler()
samples = detector_sampler.sample(shots=100)
```


## Sampling Strategies

Tsim offers two ways to simulate the non-Clifford parts of a circuit. Both start from the same ZX diagram.
Choose a strategy with the `strategy` argument of `compile_sampler` and `compile_detector_sampler`.

| Strategy | Method | Runs on | Cost grows with | Use it for |
|---|---|---|---|---|
| `"active_statevector"` (default) | Dense state vector over the active qubits of each component, as in [Clifft](https://arxiv.org/abs/2604.27058) and [SymFT](https://arxiv.org/abs/2607.28600) | CPU (numba, multi-threaded) | Number of active qubits (at most 30) | Most circuits, especially with postselection |
| `"cat5"`, `"bss"`, `"cutting"` | Stabilizer rank decomposition | CPU or GPU (JAX) | Number of non-Clifford gates | Few non-Clifford gates acting on many qubits; GPU throughput |

```python
sampler = c.compile_detector_sampler(strategy="cat5")
```

The [Tsim 1.0 blog post](https://bloqade.quera.com/blog/introducing-tsim-1-0-0/) introduces the active state vector strategy and compares its performance with other simulators.



## Architecture
<img src="docs/architecture.svg" alt="Architecture" width="100%">

A detailed description of Tsim's architecture is given in [arXiv:2604.01059](https://arxiv.org/abs/2604.01059).
Quantum programs are translated into ZX diagrams in which Pauli noise channels appear as parameterized vertices with binary variables $e_i$.
ZX simplification factors the diagram into a classical part that represents the Tanner graph and a quantum part containing the observable circuit. Both parts define a new basis of *error mechanisms* $f_i = \bigoplus_j T_{ij}\,e_j$.
Outputs that depend on the error mechanisms alone are sampled directly from the noise model.
The remaining components of the observable diagram are simulated in one of two ways:

- **Active state vector.** Each component is read off the reduced ZX diagram, without circuit extraction, as a sequence of Pauli rotations and measurements on $|0\dots0\rangle$. Tsim then simulates it on a dense state vector over its active qubits, an approach that follows [Clifft](https://arxiv.org/abs/2604.27058) ([Chase and Labib, 2026](https://arxiv.org/abs/2604.27058)) and [SymFT](https://arxiv.org/abs/2607.28600) ([Fang, Lou and Li, 2026](https://arxiv.org/abs/2607.28600)).
- **Stabilizer rank.** Each component is decomposed into a sum of Clifford terms, following [Sutcliffe and Kissinger (2024)](https://arxiv.org/abs/2403.06777), and compiled into binary JAX tensors $g_{tki}$. JIT-compiled XLA kernels contract $g_{tki}$ with batched noise configurations $f_i^{s}$ to evaluate marginal probabilities and autoregressively sample detector and observable bits.


## Differences from Stim
Tsim supports non-deterministic detectors and observables. An important consequence is that
Tsim will simulate actual detector samples, whereas Stim only reports detection flips (i.e. detection samples XORed with
a noiseless reference sample). Concretely,
```python
c = tsim.Circuit(
    """
    X 0
    M 0
    DETECTOR rec[-1]
    """
)
sampler = c.compile_detector_sampler()
samples = sampler.sample(shots=100)
print(samples)
```
will report `True` values, whereas the same circuit would result in `False` values in Stim. To reproduce the behavior of Stim, you can use the following:
```python
samples = sampler.sample(
    shots=100,
    use_detector_reference_sample=True,
    use_observable_reference_sample=True,
)
```
When set to `True`, a noiseless reference sample is computed and XORed with the
results, so that output values represent deviations from the noiseless baseline.
Note that this feature should be used carefully. If detectors or observables are not deterministic, this may lead to incorrect statistics.




## Supported Instructions

Tsim supports all [Stim instructions](https://github.com/quantumlib/Stim/blob/v1.16.0/doc/gates.md). In addition, Tsim defines the following non-Clifford instructions. Angles are given in units of π (half-turns).

| Instruction | Targets | Action |
|---|---|---|
| `T`, `T_DAG` | qubits | π/4 phase rotation around Z, and its inverse |
| `R_X(α)`, `R_Y(α)`, `R_Z(α)` | qubits | Rotation by απ around the X, Y, or Z axis |
| `U3(θ, φ, λ)` | qubits | General single-qubit unitary, each angle in units of π |
| `TPP`, `TPP_DAG` | Pauli products | exp(∓iπ/8 · P) up to global phase; `TPP Z0` is `T` |
| `R_XX(α)`, `R_YY(α)`, `R_ZZ(α)` | two distinct qubits | exp(−iαπ/2 · P) for P = XX, YY, ZZ |
| `R_PAULI(α)` | Pauli products | exp(−iαπ/2 · P) for a Pauli product of up to 64 qubits |
| `CS`, `CS_DAG` | qubit pairs | Controlled-S, diag(1, 1, 1, i), and its inverse, via a Clifford+T decomposition |
| `CCZ`, `CCX` | qubit triples | Controlled-controlled Z and Toffoli (controls first), via a Clifford+T decomposition |

```
T 0 1 2                  # T on qubits 0, 1, 2
R_Z(0.125) 2             # rotate qubit 2 around Z by π/8
U3(0.5, 0.25, 0.125) 0   # θ=π/2, φ=π/4, λ=π/8
TPP X0*Y1                # exp(-i π/8 · X0⊗Y1)
R_ZZ(0.5) 0 2            # exp(-i π/4 · Z0⊗Z2)
R_PAULI(0.3) X0*Y1*Z2    # exp(-i 0.3·π/2 · X0⊗Y1⊗Z2)
CS 0 1                   # controlled-S on qubits 0 and 1
CCX 0 1 2                # Toffoli with controls 0, 1 and target 2
```


## Upgrading from 0.1

Tsim 1.0 changes some defaults and APIs. See the [changelog](CHANGELOG.md) for the full list.

- Samplers default to `strategy="active_statevector"`. Pass `strategy="cat5"` for the previous behavior.
- With a `postselection_mask`, `CompiledDetectorSampler.sample` returns only the surviving shots, so the acceptance rate is the number of returned rows divided by `shots` (`len(result[0]) / shots` with `separate_observables=True`).
- Samples for a fixed seed differ from earlier versions.
- Tsim requires Python 3.11 or later and `stim>=1.15.0`.
- `Circuit.get_graph()` and `Circuit.get_sampling_graph()` return `zxparam.Graph`; Tsim no longer depends on `pyzx-param`.
- `Circuit.to_matrix()` is removed. `Circuit.to_tensor()` returns the circuit's channel and needs `zxparam[quimb]`.
- The `"pyzx"`, `"pyzx-dets"` and `"pyzx-meas"` diagram types are now `"zx"`, `"zx-dets"` and `"zx-meas"`.


## Publications Using Tsim

- [Simulating magic state cultivation with few Clifford terms](https://arxiv.org/abs/2509.08658) by Kwok Ho Wan and Zhenghao Zhong (2025).


## Contributing

We welcome contributions! Please see the [contributing guide](docs/contrib.md) for details on setting up your development environment, running tests, and code style guidelines.


## Citing Tsim
If you use Tsim, please consider citing the paper describing the core simulation approach:
```
@article{tsim2026,
  title={Tsim: Fast Universal Simulator for Quantum Error Correction},
  author={Haenel, Rafael and Luo, Xiuzhe and Zhao, Chen},
  journal={arXiv preprint arXiv:2604.01059},
  year={2026}
}
```

You may also cite the work by [Sutcliffe and Kissinger (2024)](https://arxiv.org/abs/2403.06777):
```
@article{sutcliffe2024fast,
  title={Fast classical simulation of quantum circuits via parametric rewriting in the ZX-calculus},
  author={Sutcliffe, Matthew and Kissinger, Aleks},
  journal={arXiv preprint arXiv:2403.06777},
  year={2024}
}
```

The active state vector strategy builds on Clifft and SymFT:
```
@article{chase2026clifft,
  title={Clifft: Fast Exact Simulation of Near-Clifford Quantum Circuits},
  author={Chase, Bradley A. and Labib, Farrokh},
  journal={arXiv preprint arXiv:2604.27058},
  year={2026}
}

@article{fang2026symft,
  title={SymFT: Universal Fault-Tolerant Quantum Circuit Simulation via Symbolic Clifford--Pauli Frames and Stabilizer Coordinates},
  author={Fang, Wang and Lou, Huazhe and Li, Riling},
  journal={arXiv preprint arXiv:2607.28600},
  year={2026}
}
```
