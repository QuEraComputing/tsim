# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `strategy="active_statevector"` samples on the GPU of an Apple-silicon Mac, in
  float32, whenever every component has at most 11 active qubits (Tsim now depends
  on `pyobjc-framework-Metal` there). A seed's GPU samples differ from the CPU's and
  depend on `batch_size`. The `device` argument of `compile_sampler`,
  `compile_detector_sampler` and the sampler classes chooses where they sample:
  `"auto"` (the default) uses the GPU when it can run the circuit and otherwise
  the CPU, without a warning; `"cpu"` keeps them on the CPU; `"metal"` raises when
  the GPU cannot run the circuit. `TSIM_DEVICE=cpu` keeps `"auto"` on the CPU, and
  `SamplerStats.device` and the repr say where they sample.
- `strategy="active_statevector"` also samples on NVIDIA GPUs, through CUDA, with
  the new `cuda` extra (`pip install bloqade-tsim[cuda]`, CUDA 13). It runs the
  same circuits as Metal and behaves the same way: `device="auto"` picks it when
  CuPy sees a GPU, and `device="cuda"` raises when the GPU cannot run the circuit.
- The `jax-cuda12` and `jax-cuda13` extras, for the stabilizer-rank strategies
  on NVIDIA GPUs with JAX. The `cuda12` and `cuda13` extras of Tsim 1.0 install
  the same packages.
- The docs pages have a menu to copy a page or open it in ChatGPT or Claude, and
  the README, the install page and the agent skill point to the docs MCP server
  at `https://bloqade.quera.com/tsim/mcp`. The README and the install page also
  show how to install the agent skill with Ion.
- An agent skill, `simulating-circuits-with-tsim`, in `skills/`, that teaches
  coding agents to use Tsim: active width, sampling and postselection, QEC
  studies with decoders, and porting published circuits. Install it with
  `npx skills add QuEraComputing/tsim --skill simulating-circuits-with-tsim`;
  the docs site also serves it. The repository's development skills in
  `.claude/skills` are marked internal, so the installer does not offer them.
- `Circuit(...)` accepts a `stim.Circuit` or another `Circuit` and copies it,
  like `Circuit.from_stim_program`. Before, passing a `stim.Circuit` raised an
  `AttributeError`.
- `CS` and `CS_DAG` (controlled-S and its inverse) gates in circuit text and
  `Circuit.append`, implemented using Clifford+T decompositions.
- `active_width` and `stats` properties on `CompiledMeasurementSampler` and
  `CompiledDetectorSampler`. `active_width` is the largest number of active
  qubits that a shot simulates (0 if every output is direct, `None` under a
  stabilizer-rank strategy); `stats` is a frozen dataclass with it, the
  component widths, the action and error-bit counts, and memory estimates.
  They replace reading the private `_program` or parsing `repr(sampler)`.

### Fixed
- Compiling a circuit with many detectors and noise locations no longer needs
  memory proportional to their product. The error basis (a GF(2) elimination
  over vertices x error bits) is now built and kept sparse, so a d=25
  surface-code 15-to-1 distillation circuit (40k qubits, 115k detectors, 1.1M
  error bits) compiles with 4.5 GB instead of failing to allocate 121 GiB.
  Seeded samples are unchanged.
- `CS`, `CCZ` and `CCX` in circuit text accept a tag containing `#` and can
  share a line with a `REPEAT n {` or `}` delimiter. Repeated target qubits, as
  in `CS 0 0`, raise an error naming the gate instead of its decomposition.

### Changed
- On an Apple GPU, detector samples that return a subset or reordering of the
  outputs (`separate_observables=True`, `prepend_observables=True`, or detectors
  only) select their columns on the GPU instead of on the host, bit-packed or as
  booleans. On an M4 Pro, distance-3 cultivation postselected on every detector
  samples 1.1 to 1.5 billion shots per second in these layouts, instead of 0.06
  to 0.15 (detectors only, bit-packed, was already fast); end-to-end cultivation
  d=5 samples 156 to 221 million instead of 1 to 3 million with booleans or
  prepended observables, and 291 instead of 59 million with separate bit-packed
  observables.
- `strategy="active_statevector"` may run a component's measurements later
  where that frees no axis that a later rotation reuses and leaves more of the
  tape before its first shot-dependent action, which a GPU backend runs once for
  all shots. The samples follow the same distribution, but some circuits' seeds
  give other samples than in 1.0.
- `ChannelSampler.signature_matrix` is now a `tsim.utils.linalg.SparseGF2Matrix`
  instead of a dense uint8 array; `np.asarray` gives the dense matrix.

## [1.0.0] - 2026-11-04

Tsim 1.0 adds a new default sampling strategy, `"active_statevector"`, which
simulates each non-Clifford component on a dense state vector over its active
qubits, building on the ideas of [Clifft](https://arxiv.org/abs/2604.27058) and
[SymFT](https://arxiv.org/abs/2607.28600). Tsim now uses
[zxparam](https://pypi.org/project/zxparam/) for its ZX diagrams instead of
pyzx-param. See the
[blog post](https://bloqade.quera.com/blog/introducing-tsim-1-0-0/) for an
overview and benchmarks.

### Breaking
- The samplers default to `strategy="active_statevector"`. Pass
  `strategy="cat5"` for the previous stabilizer-rank default.
  `CompiledStateProbs` keeps `"cat5"`. (#203)
- `CompiledDetectorSampler.sample(postselection_mask=...)` returns only the
  shots that no masked detector discards, so it can return fewer than `shots`
  rows, possibly none. The acceptance rate is the number of returned rows
  divided by `shots`: `len(result) / shots`, or `len(result[0]) / shots` with
  `separate_observables=True`. It used to return all `shots` rows, with the
  discarded ones partly filled with `False`, for the caller to filter.
  Filtering the result with the mask again still works but no longer does
  anything. Code that assumes `len(result) == shots` must change, and code
  that computes the acceptance rate as the mean of the re-applied mask now
  silently gets 1.0. The rule is the same for every strategy. Without a mask,
  exactly `shots` rows are returned. `sample` warns when a masked detector is
  a constant 1, which discards every shot. (#222, #252)
- The public API is `tsim.Circuit`, `tsim.CompiledMeasurementSampler` and
  `tsim.CompiledDetectorSampler`, which follow semantic versioning. Every other
  module is internal and may change without notice, and the API reference
  documents only the public and experimental objects. `CompiledStateProbs`,
  `SteaneEncoder` and `ColorEncoder5` are importable from `tsim.experimental`,
  which may change in any release. The stabilizer rank strategies (`"cat5"`,
  `"bss"`, `"cutting"`) are provisional: they may be removed in a minor
  release, after at least one minor release in which they emit a
  `DeprecationWarning`. A seed reproduces its samples only with the same
  versions of Tsim and its dependencies and the same backend (CPU or GPU), and,
  with a stabilizer-rank strategy, the same explicit `batch_size`. (#268)
- Samples for a fixed seed differ from earlier versions under every
  strategy: noise sampling runs in numba kernels, and the noise, the components
  and the reference samples each draw from their own child of
  `np.random.SeedSequence(seed)`. (#199, #201, #211, #263)
- Python 3.10 is no longer supported; Tsim supports Python 3.11 to 3.14. (#216)
- Tsim requires `stim>=1.15.0`, the first release with instruction tags, which
  Tsim uses to store its gates. It used to declare `stim>=1.0.0`. (#255)
- Tsim requires `numpy>=2.0` and `equinox>=0.11.5`, the oldest versions it
  works with; the `cuda13` extra requires `jax>=0.7.0`, the first release with
  a `cuda13` extra.
- `Circuit.cast_to_stim()` is removed. It only cast the type and returned the
  same `tsim.Circuit`; pass the circuit directly. (#266)
- Tsim uses zxparam instead of pyzx-param (#194, #195, #205):
  - `Circuit.get_graph()` and `Circuit.get_sampling_graph()` return
    `zxparam.Graph`. Circuits are built directly as quantum/classical ZX
    diagrams, without composing with the adjoint.
  - In `Circuit.get_graph()` (and the `"zx"` diagram), every detector and
    observable has a classical output boundary, with ports after the qubit
    outputs, so the graph's tensor is the circuit's channel together with the
    detector and observable bits.
  - `Circuit.get_sampling_graph()` is no longer doubled: qubit wires are
    quantum and measurement records are classical spiders. Its `to_tensor()`
    raises on unbound error parameters (pyzx treated them as 0); bind them by
    error index.
  - `Circuit.to_tensor()` returns the doubled tensor (the circuit's channel,
    `U ⊗ Ū` for unitaries), with error parameters set to 0 unless bound. It
    needs quimb, which is not a Tsim runtime dependency: install
    `zxparam[quimb]`, otherwise it raises `ImportError`. (#246)
  - `Circuit.to_matrix()` is removed; compare `to_tensor()` with `U ⊗ Ū`
    instead.
  - The ZX diagram types `"pyzx"`, `"pyzx-dets"` and `"pyzx-meas"` are now
    `"zx"`, `"zx-dets"` and `"zx-meas"`, and return zxparam's HTML drawing
    object instead of a graph. The old names raise `ValueError`.
- `Circuit.append` raises `ValueError` when `targets`, `arg` or `tag` are
  passed with a `stim.CircuitInstruction`, `stim.CircuitRepeatBlock` or
  `stim.Circuit`. They used to be silently ignored. (#254)

### Added
- `strategy="active_statevector"` for `compile_sampler`,
  `compile_detector_sampler` and the sampler classes. Each non-direct
  component is read off the reduced ZX diagram, without circuit extraction, as
  Pauli rotations and measurements on `|0…0⟩`, and simulated on a dense state
  vector over its active qubits, so the cost grows as `2^W` in the active width
  `W`. It runs on the CPU in multi-threaded numba kernels, without JAX. It only
  samples: it cannot compute probabilities, and it raises `ValueError` for
  components it cannot read, for components wider than 30 active qubits, and
  when one shot needs more than half of the available memory. (#203, #199,
  #212, #213, #223, #224)
  - Shots with equal histories share one state vector, which splits at
    measurements, and a shot that a postselected detector discards stops being
    simulated there. (#198, #250)
  - The samples of a seed depend on neither `batch_size` nor the thread
    count: the noise and every measurement outcome are drawn from
    counter-based streams by shot index. (#199, #207, #212)
  - With `bit_packed=True`, samples usually come straight from the packed
    words the sampler writes, without building a boolean array. (#211, #212)
- `"timeline-text"` diagram type. Renders Tsim's custom gates under their
  logical names instead of the Clifford placeholders used to store them, so
  `T` prints as `T` rather than `S`. Covers `T`, `T_DAG`, `TPP`, `TPP_DAG`,
  `R_X`, `R_Y`, `R_Z`, and `U3`. `R_XX`, `R_YY`, and `R_ZZ` share `R_PAULI`'s
  metadata tag and render as `R_PAULI(alpha)`. Rendering is provided by the
  `stim-timeline-text` package. (#186 by @shalini9894; reported in #171 by
  @egcarnio)
- `Circuit.detector_error_model(decompose_errors=True)`. The argument is passed
  on to Stim, with the logical observables kept as observables, so the
  observables must be deterministic. The docstring warns that the default,
  undecomposed model loses every error mechanism that flips more than two
  detectors when it is given to a matching decoder such as PyMatching. (#234
  by @mamuncseru, #232)
- `CompiledMeasurementSampler.sample(bit_packed=True)`, as in Stim. (#253)
- Parametric tags accept positional angles in units of pi, as in the
  shorthand: `I[R_X(0.1)] 0` is `R_X(0.1) 0`, `I[U3(0.3, 0.24, 0.49)] 0` is
  `U3(0.3, 0.24, 0.49) 0`, and `SPP[R_PAULI(0.1)] X0*Y1` is
  `R_PAULI(0.1) X0*Y1`. The keyword form (`R_X(theta=0.1*pi)`) still works;
  program text is canonicalized to it. Empty angle fields, as in
  `R_X(theta=0.1*pi,)`, are now rejected in both forms. (#218; reported in
  #202 by @dobbse42)
- Tsim gate names are case-insensitive, like Stim's: `t 0`, `ccz 0 1 2`,
  `r_x(0.1) 0` and `Circuit.append("tpp", ...)` used to fail with
  `Gate not found`. (#253)

### Changed
- Compiling a sampler is faster: ZX reduction and stabilizer-rank
  decomposition run in zxparam's Rust core (#205), and channel validation,
  the noise sampling tables, the GF(2) error basis and detector placement do
  less work. (#207)
- Noise sampling runs in numba kernels and draws each fault's outcome from an
  alias table; `numba>=0.61` is now a dependency. (#199, #211)
- Outputs that are constant now take the direct-output path, skipping
  component compilation and sampling. Postselection treats them as direct
  detectors. (#201)
- Components that feed no sampled output, such as a non-Clifford block cut
  off by an `MR` that no detector reads or by a mid-circuit `R`, are no longer
  compiled or sampled. (#262)
- With `strategy="active_statevector"` and a postselection mask, the surviving
  shots are gathered in parallel, and discarded shots are no longer finished.
  (#259)
- `repr()` of a sampler shows only the statistics of the backends it uses.
  (#251)

### Removed
- The `pyzx-param` dependency. (#194)

### Fixed
- Noise channels whose total probability is positive but at most `1e-15` are
  sampled instead of silently dropped. (#267)
- Different seeds no longer give the same samples. The noise was seeded from a
  30-bit number drawn from the seed, so seeds shared it in pairs: seeds 0 to
  300,000 hold 46 such pairs, such as 37430 and 43909, and a circuit of noise
  bits alone gave identical samples for both. The stabilizer-rank strategies
  seeded JAX with `jax.random.key(seed)`, which keeps only the low 32 bits of a
  seed, so `seed` and `seed + 2**32` gave the same component samples, and seeds
  from 2^63 on raised `OverflowError`. `seed=None` now draws 128 bits instead
  of 30. (#263)
- `Circuit.stim_circuit` keeps the tag of a `stim.CircuitRepeatBlock`. (#183 by
  @perlinm)
- `Circuit.stim_circuit` no longer reorders merged Clifford-angle rotations
  such as `SPP[R_PAULI(theta=1.0*pi)] X0 Z0` or `I[U3(...)] 0 0`, which
  produced a different circuit although `is_clifford` was True. (#242, #225)
- The 64-qubit limit of `R_PAULI` applies to each Pauli product, not to the
  whole instruction, so 33 adjacent `R_ZZ` rotations, which Stim merges into
  one instruction, are no longer rejected. (#241 by @Mohith26, #239)
- `Circuit.append("R_PAULI", pauli_string, angle)` no longer drops the sign of a
  `stim.PauliString`: a `-1` sign now reverses the rotation, and a sign of `±i`
  raises `ValueError`. (#237 by @Shriraj1901, #228)
- `Circuit.append` no longer drops the sign of a `stim.PauliString` target of
  `MPP`, `SPP`, `SPP_DAG`, `OBSERVABLE_INCLUDE`, `TPP` or `TPP_DAG` either, so
  `MPP` about `-Z` flips the result and `TPP` about `-Z` is `T_DAG`. Stim itself
  drops the sign (quantumlib/Stim#1134). For all these gates and `R_PAULI`, a
  PauliString with a sign of `±i` or no Paulis raises `ValueError`. (#240)
- `Circuit.append("CCZ", targets)` and `"CCX"` reject inverted targets such as
  `stim.target_inv(0)` instead of ignoring the inversion. (#240)
- `R_PAULI` program text accepts inverted targets such as `R_PAULI(0.3) !X0*Y1`,
  and circuits print them in that form instead of
  `SPP[R_PAULI(theta=0.3*pi)] !X0*Y1`. (#244)
- Long `E`/`ELSE_CORRELATED_ERROR` chains whose links share Pauli components,
  such as a general 3-qubit Pauli channel, compile instead of failing with
  `Maximum allowed dimension exceeded`. A chain that would need more than 2^20
  outcomes raises `ValueError`. (#248)
- Program text no longer rewrites shorthand inside tags or comments, so tags
  such as `I[R_X(0.1)]`, `X[U3(0.1, 0.2, 0.3)]` or `H[my T tag]` no longer
  fail to parse or change. (#218, #202)
- `SPP` and `SPP_DAG` with a parametric tag other than `R_PAULI` (e.g.
  `SPP[R_X(theta=0.1*pi)] X0`) raise instead of acting as a plain `SPP`, and
  `Circuit.is_clifford` returns False for them. (#218, #253)
- The README's quick start links to the tutorial notebook instead of a retired
  docs page. (#238 by @Vamsi-2581; reported in #235 by @cduck)

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
- `OBSERVABLE_INCLUDE` with Pauli targets (e.g. `OBSERVABLE_INCLUDE(0) X1`) now raises `ValueError` instead of silently producing wrong observable bits or crashing with `IndexError`. Tsim only supports measurement-record targets (`rec[-k]`).
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
- Samplers now gracefully handle circuits with no measurements or no detectors, returning empty `(shots, 0)` arrays matching Stim's behavior instead of raising an error (#106)
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
