# Installation

tsim requires Python 3.10 or later.

## Using uv (recommended)

We recommend using [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv add git+https://github.com/QuEraComputing/tsim.git
```

For GPU acceleration with CUDA:

```bash
# For CUDA 13
uv add "git+https://github.com/QuEraComputing/tsim.git#egg=tsim[cuda13]"

# For CUDA 12
uv add "git+https://github.com/QuEraComputing/tsim.git#egg=tsim[cuda12]"
```

## Using pip

```bash
pip install git+https://github.com/QuEraComputing/tsim.git
```

For GPU acceleration with CUDA:

```bash
pip install "git+https://github.com/QuEraComputing/tsim.git#egg=tsim[cuda13]"
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

This will run formatters and linters (black, isort, ruff, mypy) before each commit.
