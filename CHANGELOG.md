# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-04-07

### Fixed
- Exact scalar reduction during sum/product operations to prevent underflows/overflows of int32 on large diagrams. Unfortunately, this change comes with a 2x performance overhead, but results in more stable numerical results (#93)
- Normalization issues for circuits with arbitrary rotation gates now raise a warning instead of an error (#91)
- Parsing errors for invalid Stim circuits now raise useful exceptions (#91)

### Added
- `SPP` and `SPP_DAG` instructions — generalized S gate that phases the -1 eigenspace of Pauli product observables by i or -i. Supports multi-qubit Pauli products and inverted targets (#97)
- `MXX`, `MYY`, `MZZ` two-qubit parity measurement instructions (#96)
- `MPAD` instruction for padding the measurement record with fixed bit values (#95)


## [0.1.1] - 2026-04-01

### Added
- Improved stabilizer decomposition strategies. When compiling a sampler, you can now choose between three different strategies: `"cat5"`, `"bss"`, and `"cutting"`. The default is `"cat5"` and applies to T and arbitrary rotations; see [arxiv.org/abs/2106.07740](https://arxiv.org/abs/2106.07740) (#77)
- Sparse geometric channel sampler for noise modeling based on [this repo](https://github.com/kh428/accel-cutting-magic-state/tree/main). This significantly improves performance when the stabilizer rank is low. (#64)
- `Circuit.append` method for programmatic circuit construction (#65)
- `Circuit.is_clifford` property and automatic replacement of U3 gates with Clifford equivalents for pi/2 rotations (#69)
- Improved `pyzx` visualization. Now *doubled ZX notation* is used when using the `"pyzx"` argument in `Circuit.diagram`, which is a technically accurate depiction of the quantum circuit (#86)
- Automatic batch size selection based on available memory (#84)

### Changed

- Tsim now uses `pyzx-param==0.9.3` which fixes a bug where diagrams were not fully reduced in the absence of noise
- Tsim will now make sure that marginal probabilities are normalized and raise an error if they are not. Wrong normalization can be the result of rare underflow errors that will be addressed in a future release (#87)
- Use BLAS matmul kernel for tensor contractions (#63)
- Circuit flattening deferred to ZX graph construction time (#71)
- White background for SVG plots, which are now readable in dark mode (#85)



## [0.1.0] - 2026-01-28

### Added
- Initial release
- Clifford+T circuit simulation via stabilizer rank decomposition
- Stabilizer decomposition backend based on pyzx and the [paramzx-extension](https://github.com/mjsutcliffe99/ParamZX) by [(2025) M Sutcliffe and A Kissinger](https://arxiv.org/pdf/2403.06777)
- Support for most [Stim](https://github.com/quantumlib/Stim) instructions
- `T`, `T_DAG`, `R_Z`, `R_X`, `R_Y`, and `U3` instructions
- Arbitrary rotations gates via magic cat state decomposition from Eq. 10 of [(2021) Qassim et al.](https://arxiv.org/pdf/2106.07740)
- GPU acceleration via jax
- Documentation and tutorials
