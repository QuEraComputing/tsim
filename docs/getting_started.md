# Getting Started

`tsim` is a quantum circuit sampler that can efficiently sample from Clifford+T circuits with Pauli noise. It is based on ZX-calculus stabilizer rank decomposition and parametrized ZX diagrams: [arXiv:2403.06777](https://arxiv.org/abs/2403.06777).

## Quick Example



## Circuit Construction

### From Stim Programs



### Programmatic Construction

TODO: decide if we expose this.


## Sampling

### Measurement Sampling

### Detector Sampling


## Visualization



## Performance Tips

- **JIT compilation**: The first sample call might be slow due to compilation, subsequent calls are fast
- **GPU acceleration**: Install with `pip install tsim[cuda13]` for automatic GPU acceleration.
- **Batch sampling**: Especially on GPU, higher batch sizes can significantly improve performance.

## Next Steps

- See the [Contributing](contrib.md) guide to learn about the development workflow
- Check out the demos in `docs/demos/` for more examples
- Read the [arXiv paper](https://arxiv.org/abs/2403.06777) for the theoretical background
