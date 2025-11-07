# tsim

A circuit sampler based on ZX stabilizer rank decomposition. It feels like `stim` but can handle T-gates.

Circuits are represented as parametrized ZX diagrams and compiled for efficient sampling on CPU or GPU. See [arXiv:2403.06777](https://arxiv.org/abs/2403.06777).


# TODO

- [ ] Rewrite pyzx extension for parametrized diagrams and simplification rules (based on newest pyzx version)
- [ ] Detector sampling via decoding graph: only possible for errors after Ts. Need to figure out dependency, i.e. some errors will produce deterministic detector flips, others will not.
- [ ] Irreducible set of equivalent errors (combining errors like stim)
- [ ] Fp64? Exact arithmetic?
- [ ] Symbolic simplification of scalars?
- [ ] Allow taking multiple samples from a single error configuration?
- [ ] Support/ API for stab rank decomposition techniques, e.g., cutting diagrams
- [ ] Test on GPU
- [ ] Benchmarks




## Installation

```bash
uv add tsim
```
