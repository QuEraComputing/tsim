# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `"timeline-text"` diagram type. Renders tsim's custom gates under their
  logical names instead of the Clifford placeholders used to store them, so
  `T` prints as `T` rather than `S`. Covers `T`, `T_DAG`, `TPP`, `TPP_DAG`,
  `R_X`, `R_Y`, `R_Z`, and `U3`. `R_XX`, `R_YY`, and `R_ZZ` share `R_PAULI`'s
  metadata tag and render as `R_PAULI(alpha)`. Rendering is provided by the
  `stim-timeline-text` package. (#171)
- `strategy="active_statevector"` for `compile_sampler`, `compile_detector_sampler`
  and the sampler classes. Each non-direct component is read off the reduced ZX
  diagram, without circuit extraction, as Pauli rotations and measurements on
  `|0…0⟩`, and simulated on a dense state vector over its active qubits, so the
  cost grows as `2^W` in the active width `W`. It only samples: it cannot compute
  probabilities, and it raises `ValueError` for components it cannot read, for
  components wider than 30 active qubits, and when one shot needs more than half
  of the available memory. Batches hold at most 2^16 shots. (#203)
- `trajectory_sharing=True` for `strategy="active_statevector"`: shots with
  equal histories share one state vector, which splits at measurements, and
  shots that a postselected component detector discards stop being simulated
  there. (#198)

### Changed
- The samplers now default to `strategy="active_statevector"`; pass
  `strategy="cat5"` for the previous stabilizer-rank default.
  `CompiledStateProbs` keeps `"cat5"`. (#203)
- `strategy="active_statevector"` no longer uses JAX: it samples on the CPU
  with numba kernels, also when JAX has a GPU, and trajectory sharing is on
  by default (`trajectory_sharing=None` means True for this strategy and
  False for the others). Amplitudes are complex128. Postselected
  detectors of a component prune its shots with sharing off as well, and also
  when no direct detector is postselected. (#199)
- Without a `batch_size`, a statevector call is split into one batch per
  numba thread whenever each batch gets at least 4096 shots, and smaller
  calls run as one batch on the parallel kernels. An explicit `batch_size`
  keeps samples independent of the thread count. (#199)
- Noise sampling runs in numba kernels, so the samples for a given seed
  differ from earlier versions; `numba>=0.61` is now a dependency. (#199)
- Tsim now uses zxparam instead of pyzx-param. Circuits are built directly as quantum/classical ZX diagrams (no more composing with the adjoint), and reduction and stabilizer-rank decomposition run in zxparam.
- `Circuit.get_graph()` and `Circuit.get_sampling_graph()` return `zxparam.Graph`.
- In `Circuit.get_graph()` (and the `"zx"` diagram), every detector and observable has a classical output boundary, with ports after the qubit outputs, so the graph's tensor is the circuit's channel together with the detector and observable bits.
- `Circuit.to_tensor()` returns the doubled tensor (the circuit's channel, `U ⊗ Ū` for unitaries), with error parameters set to 0 unless bound.
- `Circuit.to_tensor()` needs quimb, which is not a Tsim runtime dependency: install `zxparam[quimb]`, otherwise it raises `ImportError`.
- `Circuit.get_sampling_graph()` is no longer doubled: qubit wires are quantum and measurement records are classical spiders. Its `to_tensor()` raises on unbound error parameters (pyzx treated them as 0); bind them by error index. With all errors bound, the tensor is the joint distribution of the outputs.
- The ZX diagram types are now `"zx"`, `"zx-dets"` and `"zx-meas"`. They return zxparam's HTML drawing object instead of a graph.
- Internals: `GraphRepresentation` (`tsim.core.instructions`) is now `GraphBuilder` in the new `tsim.core.builder`, and `parse_stim_circuit` returns it. Its bookkeeping functions became methods: `detector` is now `add_detector` and `observable_include` is now `include_observable`; `tick`, `correlated_error` and `finalize_correlated_error` keep their names; `ensure_lane`, `add_lane`, `add_quantum_boundary`, `last_col` and `last_edge` are now private. Its fields are private behind read-only properties (`rec` is now `records`). `ANNOTATION_ROW`, `HELPER_LANE` and `_error_phase` (now `error_phase`) moved to `tsim.core.builder`. `build_sampling_graph` (`tsim.core.graph`) is replaced by `GraphBuilder.build(closed=..., outputs=...)`.
- Internals moved or removed: `classify_direct` moved from `tsim.core.graph` to `tsim.compile.pipeline`; `tsim.utils.diagram.render_pyzx_d3`, `squash_graph` and `scale_horizontally` are gone, since ZX diagrams use zxparam's `draw` layout and scale options.

### Deprecated
- The `"pyzx"`, `"pyzx-dets"` and `"pyzx-meas"` diagram types. They still work as aliases for `"zx"`, `"zx-dets"` and `"zx-meas"` but emit a `DeprecationWarning`.

### Removed
- `Circuit.to_matrix()` (compare `to_tensor()` with `U ⊗ Ū` instead).
- The `pyzx-param` dependency.
- Internals: the `track_classical_wires` argument of `parse_stim_circuit`, the `tsim.compile.stabrank` module, and `ConnectedComponent`, `connected_components`, `get_params` and `evaluate_graph` from `tsim.core.graph`.

## [0.1.5] - 2026-07-21

### Added
- `CCX` (Toffoli) and `CCZ` gates in circuit text and `Circuit.append`, implemented using Clifford+T decompositions. (#150)
- `R_XX`, `R_YY`, `R_ZZ`, and `R_PAULI` parametric Pauli-product rotation gates. `R_XX(alpha) q0 q1` applies `exp(-i * alpha * pi/2 * XX)` (and likewise for `YY`/`ZZ`), while `R_PAULI(alpha) X0*Y1*Z2` applies the rotation for an arbitrary Pauli product (up to 64 qubits per instruction). The `alpha` parameter is in half-turns, matching the existing `R_X`/`R_Y`/`R_Z` convention. Duplicate target qubits are rejected at parse time. (#152)
- Global-rotation quantum error-correction tutorial covering color-code state preparation, noise, postselection, and logical-error analysis. (#156)
- Fast path in the detector sampler for components whose output is deterministically given by a single error variable. These components now skip the JAX compilation and autoregressive sampling pipeline, significantly speeding up detector sampling for surface-code circuits at low physical error rates. (#78)
- `CompiledDetectorSampler.sample` now accepts an optional `postselection_mask` argument for postselected simulations. The mask has length `num_detectors`; a shot is discarded when any masked detector fires. When a discarded shot is flagged by a direct detector, the expensive JAX autoregressive loop is skipped for that draw while still returning one row per requested shot. Discarded rows retain their direct detector columns and fill all other columns with zero; callers recover surviving shots by re-applying the mask to the returned detector columns. Masks that only target non-direct detectors, or that mask no direct detectors, fall back to the standard sampling path. Fully-direct circuits continue to use the NumPy fast path. (#157)

### Changed
- Exact-scalar reductions and device-to-host transfers now use lower-memory JAX and CUDA paths, reducing sampling memory use and transfer overhead for large detector-sampling workloads. (#149)
- `ipywidgets` is now installed as a runtime dependency so notebook circuit diagrams work in clean installations. (#179)

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
