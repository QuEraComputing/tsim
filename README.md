# tsim

A GPU-accelerated circuit sampler via ZX-calculus stabilizer rank decomposition.
**Tsim feels just like [Stim](https://github.com/quantumlib/Stim), but can handle T-gates.**

It is based on work by [Sutcliffe and Kissinger (2025)](https://arxiv.org/abs/2403.06777).

## Installation

Tsim is not yet released on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/QuEraComputing/tsim.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/QuEraComputing/tsim.git
```

If you machine has a GPU, use:
```bash
pip install "git+https://github.com/QuEraComputing/tsim.git#egg=tsim[cuda13]"
```


## Quick Start
An introductory tutorial is available [here](https://github.com/QuEraComputing/tsim/blob/main/docs/demos/encoding_demo.ipynb).

For many existing scripts, replacing `stim` with `tsim` should just work. Tsim mirrors the Stim API and currently supports all [Stim gates](https://github.com/quantumlib/Stim/wiki/Stim-v1.9-Gate-Reference), except `CORRELATED_ERROR` and `ELSE_CORRELATED_ERROR`.

T-gates are supported via the `S[T]` and `S_DAG[T]` instructions:

```python
import tsim

c = tsim.Circuit(
    """
    RX 0
    R 1
    S[T] 0  # T gate
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

## What is Tsim?

Stim is a tool that allows sampling from Clifford+T circuits or computing amplitudes for
outputs bitstrings. It's underlying algorithm is stabilizer rank decomposition, together with ZX-calculus simplification rules.

As such, Tsim can scales to hundreds or even thousands of qubits, as long as the circuit
does not have too many T-gates.

Just like Stim, Tsim compiles circuits into measurement or detector samplers.
These samplers manage a contiguous data structure that allows for efficient sampling on CPU or GPU, following the approach described in [Sutcliffe and Kissinger (2025)](https://arxiv.org/abs/2403.06777).

Tsim supports all [Stim gates](https://github.com/quantumlib/Stim/wiki/Stim-v1.9-Gate-Reference), except `CORRELATED_ERROR` and `ELSE_CORRELATED_ERROR`.



## Roadmap

- [ ] Update PyZX dependency
- [ ] More efficient error bit sampling (combine single-bit errors)
- [ ] Efficient sampling from error distribution
- [ ] Symbolic simplification of scalars
- [ ] Support multiple samples per error configuration
- [ ] API for stabilizer rank decomposition techniques (e.g., diagram cutting)
- [ ] Benchmarks
- [ ] Simulate d=5 magic state cultivation circuit
- [ ] Add quizx-powered simulator that is not compiled, analogous to stim's `TableauSimulator`
