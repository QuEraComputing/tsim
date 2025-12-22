# tsim: fast sampling of Clifford+T circuits

`tsim` is a quantum circuit sampler designed for efficient sampling of Clifford+T circuits with Pauli noise.

`tsim` follows the `stim` API and works with `stim` circuits. It supports all `stim` [gates and noise channels](https://github.com/quantumlib/Stim/wiki/Stim-v1.9-Gate-Reference),
and, importantly, T-gates.



## Quick Start



```python
c = tsim.Circuit(
    """
    RX 0
    T 0
    DEPOLARIZE1(0.1) 0
    H 0
    M 0
    """
)
sampler = c.compile_sampler()
samples = sampler.sample(shots=100)
```

For circuits with detector and observable annotations, you can compile a detector sampler:

```python
c = tsim.Circuit(
    """
    RX 0
    R 1
    T_DAG 0
    PAULI_CHANNEL_1(0.1, 0.1, 0.2) 0 1
    H 0
    CNOT 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    """
)
detector_sampler = c.compile_detector_sampler()
samples = detector_sampler.sample(shots=100)
```

## Installation

```bash
uv add git+https://github.com/QuEraComputing/tsim.git
```

For GPU acceleration, use

```bash
uv add "git+https://github.com/QuEraComputing/tsim.git#egg=tsim[cuda13]"
```

See [Installation](install.md) for more options.

## How It Works

`tsim` uses stabilizer rank decomposition based on the ZX calculus and is built on top of `pyzx`.
Circuits are converted into ZX diagrams where noise channels are injected as parametrized Pauli vertices. For efficient sampling on
CPU and GPU, the diagram is compiled into contiguous jax arrays, following the approach described in [arXiv:2403.06777](https://arxiv.org/abs/2403.06777).
