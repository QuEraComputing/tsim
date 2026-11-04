# Installation

Tsim requires Python 3.11 or later.

## Using uv (recommended)

We recommend using [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv add bloqade-tsim
```

For GPU acceleration with CUDA:

```bash
# For CUDA 13
uv add "bloqade-tsim[cuda13]"

# For CUDA 12
uv add "bloqade-tsim[cuda12]"
```

## Using pip

```bash
pip install bloqade-tsim
```

For GPU acceleration with CUDA:

```bash
pip install "bloqade-tsim[cuda13]"
```

## Statevector sampling on NVIDIA GPUs

The `cuda13` and `cuda12` extras accelerate the stabilizer-rank strategies with
JAX. The statevector strategy has its own CUDA executor, which needs CuPy and
CUDA 13 (driver 580 or later):

```bash
pip install "bloqade-tsim[gpu]"
```

With it installed, `strategy="active_statevector"` (the default) samples on the
GPU whenever CuPy sees one and every component has at most 11 active qubits;
otherwise it samples on the CPU. The GPU computes in float32, and a seed's
samples differ from the CPU's. With a `postselection_mask`, only the shots that
pass are copied off the GPU, so heavily postselected circuits gain the most:

```python
sampler = circuit.compile_detector_sampler()
kept = sampler.sample(
    1_000_000, append_observables=True, bit_packed=True, postselection_mask=mask
)
```

`CUDA_VISIBLE_DEVICES=""` keeps the samplers on the CPU.

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
