# Installation

Tsim requires Python 3.11 or later.

## Using uv (recommended)

We recommend using [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv add bloqade-tsim
```

## Using pip

```bash
pip install bloqade-tsim
```

## NVIDIA GPUs

Each sampling strategy has its own GPU extra:

| Extra | Accelerates | Needs |
|---|---|---|
| `cuda` | `"active_statevector"` (the default), with CuPy | CUDA 13 (driver 580 or later) |
| `jax-cuda13` | `"cat5"`, `"bss"` and `"cutting"`, with JAX | CUDA 13 |
| `jax-cuda12` | `"cat5"`, `"bss"` and `"cutting"`, with JAX | CUDA 12 |

```bash
uv add "bloqade-tsim[cuda]"
# or, with both strategies on the GPU:
uv add "bloqade-tsim[cuda,jax-cuda13]"
```

The `cuda12` and `cuda13` extras of Tsim 1.0 still install the JAX extras.

### The default strategy

With the `cuda` extra, `strategy="active_statevector"` samples on the GPU
whenever CuPy sees one and every component has at most 11 active qubits;
otherwise it samples on the CPU, with a warning. The GPU computes in float32,
and a seed's samples differ from the CPU's. With a `postselection_mask`, only
the shots that pass are copied off the GPU, so heavily postselected circuits
gain the most:

```python
sampler = circuit.compile_detector_sampler()
kept = sampler.sample(
    1_000_000, append_observables=True, bit_packed=True, postselection_mask=mask
)
```

`CUDA_VISIBLE_DEVICES=""` keeps the samplers on the CPU.

### Both strategies in one process

JAX starts only when a stabilizer-rank strategy samples, and then reserves 75%
of the GPU's memory. To sample with both strategies in one process, set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before starting Python, so that JAX
allocates as it goes.

## Development Setup

If you're contributing to Tsim, clone the repository and install development dependencies:

```bash
git clone https://github.com/QuEraComputing/tsim.git
cd tsim
uv sync
```

Install pre-commit hooks to run linting checks automatically:

```bash
pre-commit install
```

This will run formatters and linters (black, isort, ruff, pyright) before each commit.
