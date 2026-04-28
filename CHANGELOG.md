# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-04-28

### Fixed
- `MR`, `MRX`, and `MRY` no longer double-count their measurement flip probability as both a pre-measurement Pauli error and a measurement-result flip.
- Fixed a bug where `M(p)` instructions incorrectly flipped the qubit, not just the measurement record. This would affect circuits where qubits were measured and not immediately reset.
- Fixed a bug where `MR !q` instructions (with measurement record inversion) produced wrong measurement results.
- Out-of-order `OBSERVABLE_INCLUDE` indices now produce the correct sampler column order and output shape. Missing indices below the maximum mentioned id appear as deterministic-zero columns, and columns are emitted in sorted logical-index order.
- `matmul_gf2` no longer silently corrupts parity for inner products with more than 255 set bits. The float32→uint8 cast in JAX saturates at 255, which previously made `% 2` always return 1 once a row-sum reached 256. The modulo is now applied on float32 before the uint8 cast.
- `CompiledDetectorSampler.sample` now raises `ValueError` when `separate_observables=True` is combined with `prepend_observables=True` or `append_observables=True` (matching Stim). Previously these combinations silently dropped observable columns. The `prepend_observables=True` + `append_observables=True` combination is now supported and returns columns in `[obs, det, obs]` order, matching Stim.
- `OBSERVABLE_INCLUDE` with Pauli targets (e.g. `OBSERVABLE_INCLUDE(0) X1`) now raises `ValueError` instead of silently producing wrong observable bits or crashing with `IndexError`. tsim only supports measurement-record targets (`rec[-k]`).
- Empty `DETECTOR` and `OBSERVABLE_INCLUDE` annotations (without targets) no longer crash the parser; they now produce zero detector/observable bits, matching Stim semantics.
- `CompiledDetectorSampler.sample` with `use_detector_reference_sample=True` or `use_observable_reference_sample=True` no longer returns fewer rows than `shots` when called with an explicit `batch_size` that exactly divides `shots`.
- Incorrect visualization of `CORRELATED_ERROR` and `ELSE_CORRELATED_ERROR` instructions in the `pyzx` diagram renderer. Previously, error vertices were rendered as classical spiders instead of bold quantum spiders.
- Sweep-bit targets (e.g. `CX sweep[0] 1`) were silently parsed as unconditional gates. The parser now raises `NotImplementedError`, since sweep parameters are not supported by Tsim.


### Added
- `TPP` and `TPP_DAG` instructions — applies exp(-i pi/8 P) or exp(+i pi/8 P) (up to global phase) for a Pauli product P, i.e., phases the -1 eigenspace of P by exp(i pi/4) or exp(-i pi/4).
- `Circuit.is_clifford` now supports `REPEAT` blocks.


## [0.1.3] - 2026-04-14

### Fixed
- `DEPOLARIZE2` channel was missing the `p_ZZ` probability term, which was always set to 0. This lead to incorrect noise models that were missing ZZ errors. (#103)
- Samplers now gracefully handle circuits with no measurements or no detectors, returning empty `(shots, 0)` arrays matching stim's behavior instead of raising an error (#106)
- `MPP, MXX, MYY, MZZ` instructions now support a bit flip probability argument (#118)

### Added
- Zoomable timeline and timeslice diagrams. `Circuit.diagram` now accepts a `zoomable` option, enabled by default, to support pan and zoom in notebooks for the `timeline-svg` and `timeslice-svg` diagram types (#116)
- `HERALDED_PAULI_CHANNEL_1` and `HERALDED_ERASE` noise channel instructions with herald bit indicating whether the noise event occurred (#107)
- `CXSWAP`, `CZSWAP`, `SWAPCX`, `SWAPCZ` two-qubit gate instructions (#105)
- `C_NXYZ`, `C_XNYZ`, `C_XYNZ`, `C_NZYX`, `C_ZNYX`, `C_ZYNX` axis-cycling gate variants with negated axes (#105)
- `H_NXY`, `H_NXZ`, `H_NYZ` Hadamard-like gate variants with negated axes (#105)
- `II` two-qubit identity instruction that acts trivially (#105)

### Changed
- `I_ERROR`, `II_ERROR`, and `QUBIT_COORDS` instructions now allocate qubit lanes instead of being silently skipped (#105)

## [0.1.2] - 2026-04-07

### Fixed
- Exact scalar reduction during sum/product operations to prevent underflows/overflows of int32 on large diagrams. Unfortunately, this change comes with a 2x performance overhead, but results in more stable numerical results (#93)
- Normalization issues for circuits with arbitrary rotation gates now raise a warning instead of an error (#91)
- Parsing errors for invalid Stim circuits now raise useful exceptions (#91)

### Added
- `SPP` and `SPP_DAG` instructions — generalized S gate that phases the -1 eigenspace of Pauli product observables by i or -i. Supports multi-qubit Pauli products and inverted targets (#97)
- `MXX`, `MYY`, `MZZ` two-qubit parity measurement instructions, delegating to existing MPP infrastructure. Also adds `II_ERROR` support (#96)
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
