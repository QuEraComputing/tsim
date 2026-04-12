[![Docs](https://img.shields.io/badge/docs-latest-blue?style=flat&logo=materialformkdocs)](https://queracomputing.github.io/tsim/latest)
[![Coverage](https://img.shields.io/codecov/c/github/QuEraComputing/tsim?style=flat&logo=codecov)](https://codecov.io/gh/QuEraComputing/tsim)
[![arXiv](https://img.shields.io/badge/arXiv-2403.06777-b31b1b.svg?style=flat&logo=arxiv)](https://arxiv.org/abs/2604.01059)
[![Like This? Leave a star](https://img.shields.io/github/stars/QuEraComputing/tsim?style=flat&label=Like%20Tsim%3F%20Leave%20a%20star&color=yellow&logo=github)](https://github.com/QuEraComputing/tsim)

# Tsim

A GPU-accelerated quantum circuit sampler based on ZX-calculus stabilizer rank decomposition.
Tsim feels just like [Stim](https://github.com/quantumlib/Stim), but supports non-Clifford gates.


## Installation

Install with pip:
```bash
pip install bloqade-tsim
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add bloqade-tsim
```

If your machine has a GPU, use:
```bash
pip install "bloqade-tsim[cuda13]"
```


## Quick Start
An introductory tutorial is available [here](https://queracomputing.github.io/tsim/latest/demos/encoding_demo/). Please also refer to the [documentation](https://queracomputing.github.io/tsim/latest/).

For many existing scripts, replacing `stim` with `tsim` should just work. Tsim mirrors the Stim API and currently supports all [Stim instructions](https://github.com/quantumlib/Stim/wiki/Stim-v1.13-Gate-Reference).

Additionally, Tsim supports the instructions `T`, `T_DAG`, `R_Z`, `R_X`, `R_Y`, and `U3`.
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


## Architecture
![Architecture](docs/architecture.svg)
A detailed description of Tsim's architecture is given in [arXiv:2604.01059](https://arxiv.org/abs/2604.01059).
Quantum programs are translated into ZX diagrams in which Pauli noise channels appear as parameterized vertices with binary variables $e_i$.
ZX simplification factors the diagram into a classical part that represents the Tanner graph and a quantum part containing the observable circuit. Both parts define a new basis of *error mechanisms* $f_i = \bigoplus_j T_{ij}\,e_j$.
The observable diagram is used to compute marginal probabilities for autoregressive sampling. Here, each diagram is decomposed into a sum of Clifford terms via stabilizer rank decomposition, following [Sutcliffe and Kissinger (2024)](https://arxiv.org/abs/2403.06777), and compiled into binary JAX tensors $g_{tki}$.
At sampling time, JIT-compiled XLA kernels contract $g_{tki}$ with batched noise configurations $f_i^{s}$ to evaluate marginal probabilities and autoregressively sample detector and observable bits.

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


## Benchmarks

With GPU acceleration, Tsim can achieve sampling throughput for low-magic circuits that approaches the throughput of Stim on Clifford circuits of the same size. The figure below shows a comparison for [distillation circuits](https://arxiv.org/html/2412.15165v1) (35 and 85 qubits), [cultivation circuits](https://arxiv.org/abs/2409.17595), and rotated surface code circuits.
Tsim can be five orders of magnitude faster than [quizx](https://github.com/zxcalc/quizx).

![Benchmarks](docs/benchmarks.svg)


## Supported Instructions

Tsim supports all [Stim instructions](https://github.com/quantumlib/Stim/wiki/Stim-v1.13-Gate-Reference). In addition, Tsim defines the following non-Clifford instructions:

### `T` and `T_DAG`

The `T` gate applies a π/4 phase rotation around the Z axis, and `T_DAG` is its inverse:


```
T 0 1 2  # Apply T to qubits 0, 1, 2
T_DAG 0  # Apply T_DAG to qubit 0
```

### Rotation Gates: `R_X`, `R_Y`, `R_Z`

Rotation gates around the X, Y, and Z axes by an angle θ = α·π (where α is specified as the parameter):


```
R_X(0.11) 0  # Rotate qubit 0 around X by 0.11π
R_Y(0.25) 1  # Rotate qubit 1 around Y by π/4
R_Z(0.125) 2  # Rotate qubit 2 around Z by π/8
```

### `U3` Gate

The general single-qubit unitary with three parameters (θ, φ, λ), each specified as a multiple of π:


```
U3(0.5, 0.25, 0.125) 0  # Apply U3 with θ=π/2, φ=π/4, λ=π/8
```

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
