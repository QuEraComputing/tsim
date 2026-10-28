# Installation

tsim requires Python 3.11 or later.

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

```python
sampler = circuit.compile_detector_sampler(device="cuda")
samples = sampler.sample(1_000_000, append_observables=True)
# Only the shots that pass postselection, bit-packed, without the postselected
# detectors, and the number of shots that postselection discarded:
kept, discarded = sampler.sample_postselected(1_000_000, postselection_mask=mask)
```

It simulates components of up to 11 active qubits in float32.

## Development Setup

If you're contributing to tsim, clone the repository and install development dependencies:

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
