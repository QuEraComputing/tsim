# Installation

tsim requires Python 3.10 or later.

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

### NVIDIA extras

On CUDA machines two more optional packages speed up sampling:

- [`cuStabilizer`](https://docs.nvidia.com/cuda/cuquantum/latest/custabilizer/) (part of
  `cuquantum-python`) together with `cupy` samples the error channels on the GPU instead
  of the host. tsim uses it automatically when it is installed (`channel_backend="auto"`).
- `cuda-bindings` lets tsim copy results back through pinned host memory.

```bash
# For CUDA 13
uv add "bloqade-tsim[cuda13,nvidia13]"

# For CUDA 12
uv add "bloqade-tsim[cuda12,nvidia12]"
```

## Using pip

```bash
pip install bloqade-tsim
```

For GPU acceleration with CUDA:

```bash
pip install "bloqade-tsim[cuda13]"
```

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
