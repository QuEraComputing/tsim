# Tsim vs Clifft on the clifft-bench circuits

Everything needed to benchmark Tsim's `strategy="statevector"` sampler (one lane
per shot and with trajectory sharing) against [Clifft](https://github.com/unitaryfoundation/clifft)
on the same machine: the circuits, the harness, the last full run and the notes
of the investigations behind it. All commands run from the repository root.

| path | contents |
|---|---|
| `comparison/` | the current harness: `run_all.sh`, `run_tsim.py`, `run_clifft.py`, `report.py`, `workloads.py` |
| `circuits/` | copies of every circuit the scripts read (sources below) |
| `results/` | full runs of `run_all.sh`; new runs are written here |
| `statevector/` | the earlier five-circuit benchmark without trajectory sharing |
| `trajectory/` | the measurement scripts of the trajectory-sharing investigation |
| `notes/` | write-ups and earlier results |

## Setup

```bash
uv sync
```

Clifft is not a Tsim dependency, so it lives in its own environment.
`run_all.sh` looks for it at `/tmp/clifft-venv` (override with `CLIFFT_PYTHON`):

```bash
uv venv --python 3.12 /tmp/clifft-venv
uv pip install --python /tmp/clifft-venv clifft==0.11.0
```

Add `--no-config` to both if a user-level `uv.toml` points at a private index.

## Run the comparison

```bash
benchmarks/clifft/comparison/run_all.sh
```

This runs Clifft, then Tsim on all threads, then Tsim on one thread
(`NPROC=1 NUMBA_NUM_THREADS=1`), and writes everything to
`results/statevector-vs-clifft-<timestamp>/` (or the directory given as the
first argument). The last full run took 29 minutes on an M4 Pro.

- Before each step it waits until the summed `%CPU` of all processes stays below
  `CPU_MAX` (default 300) for three checks a minute apart. Run nothing else
  meanwhile: other benchmarks or busy apps slow every rate down.
- `SECONDS_PER` (default 10) is the timed window of every measurement.
- `environment.txt` records the Tsim commit (and whether tracked files were
  modified), the Clifft and JAX versions and the machine. `load.log` samples
  CPU use and load every 30 s. `report.md` is the summary.
- `progress.log` lists the five busiest processes before each step. Remove
  those lines before committing a run (the reference run below has them
  removed).

Single steps, for a subset of workloads:

```bash
mkdir -p /tmp/run
```
```bash
uv run python benchmarks/clifft/comparison/run_tsim.py --seconds 10 --out /tmp/run/tsim-all.jsonl distillation cultivation_d3
```
```bash
/tmp/clifft-venv/bin/python benchmarks/clifft/comparison/run_clifft.py --seconds 10 --out /tmp/run/clifft.jsonl distillation cultivation_d3
```
```bash
uv run python benchmarks/clifft/comparison/report.py /tmp/run
```

Workload names are the keys of `WORKLOADS` in `comparison/workloads.py`:
`surface_d7_r7`, `coherent_d3_r1`, `coherent_d5_r1`, `distillation`,
`cultivation_d3`, `coherent_d3_r3`, `cultivation_d5`, `coherent_d5_r5`,
`qv_q10`, `qv_q20`, `symft_msc_d7`, `symft_msc_d7_postselected`.

### What is measured

Counting follows clifft-bench: throughput is attempted shots per second, every
detector is postselected (except on Quantum Volume and `symft_msc_d7`), and the
logical count is observable 0 among accepted shots. Compilation and one warmup
call (JIT) are timed separately and excluded from the rate.

- Tsim runs each workload with one lane per shot and with each
  trajectory-sharing configuration in `Workload.shared`.
- Clifft compiles with its default HIR passes and samples with
  `sample_survivors`, with scalar execution and with packed lanes of 2048 shots
  (where they fit in its 64 MiB limit).
- `report.py` keeps each simulator's fastest configuration per thread setting.
  It also pools every run of a simulator and compares acceptance and the
  observable-0 rate between Tsim and Clifft with a two-proportion z-test,
  flagging |z| > 5.

## Reference results

`results/statevector-vs-clifft-20261013-020940-tsim-perf/` is the last full
run (2026-10-13, Apple M4 Pro, 14 cores, 48 GB, macOS 15.7.9, JAX 0.9.2,
Clifft 0.11.0). It was measured at `af98f77`, the pre-rebase twin of `b37378d`
on this branch, so it predates the noise-setup speedups `09581e4` and
`e394e84`. Every workload passes the Tsim/Clifft check. Tsim's best rate
divided by Clifft's:

| circuit | 1 thread | all threads |
|---|---:|---:|
| surface_d7_r7 | 1.63x | 0.61x |
| coherent_d3_r1 | 9.36x | 1.20x |
| coherent_d5_r1 | 1923x | 195x |
| distillation | 1.87x | 0.64x |
| cultivation_d3 | 1.41x | 0.24x |
| coherent_d3_r3 | 2.75x | 0.41x |
| cultivation_d5 | 21.1x | 5.83x |
| coherent_d5_r5 | 38.1x | 12.4x |
| qv_q10 | 23.0x | 4.57x |
| qv_q20 | 0.38x | 0.64x |
| symft_msc_d7 | 24.6x | 34.6x |
| symft_msc_d7_postselected | 338x | 205x |

## Circuits

| path | source | license |
|---|---|---|
| `circuits/clifft-bench/` | `workloads/circuits/` of [unitaryfoundation/clifft-bench](https://github.com/unitaryfoundation/clifft-bench) at `ad0647f` (suite 0.2.2), which copies them from [unitaryfoundation/clifft-paper](https://github.com/unitaryfoundation/clifft-paper) at `db7dc9f`; checksums and semantics are in clifft-bench's `manifests/workloads.v1.json` | Apache-2.0 (`LICENSE-Clifft-paper`) |
| `circuits/symft/MSC_circuit_d7_p0.0005.stim` | `benchmark/circuit/` of [haoliri0/SymFT_Test](https://github.com/haoliri0/SymFT_Test) at `9ec5790` (the same file ships in haoliri0/SOFT) | Apache-2.0 (`LICENSE-SymFT_Test`) |
| `circuits/clifft-paper/msc_d5_end2end_p1e-3.stim` | generated with clifft-paper `db7dc9f`, `magic_state_cultivation/convert_s_to_t.make_d5_end2end(0.001)` | Apache-2.0 (`LICENSE-Clifft-paper`) |

The d=7 SymFT circuit has no documented provenance. With its T gates replaced by
S gates all 344 detectors are deterministic, but as a T circuit it is not
sound: run noiselessly, 12 detectors are random, 13% of shots pass and
observable 0 is 0.47. Treat it as a performance workload only. Tsim and Clifft
still have to agree on it statistically.

The d=5 end-to-end circuit (injection, cultivation and escape to a d=15 surface
code; 463 qubits, 2744 detectors) is not part of `run_all.sh`. It was used to
measure compile time. `msc_d5_end2end_p1e-3.postselection.npy` postselects its
107 injection and cultivation detectors: those with at most 4 coordinates or
with fifth coordinate -9. The sampler returns every shot, so acceptance is
computed over the masked detectors:

```python
import time

import numpy as np

import tsim

path = "benchmarks/clifft/circuits/clifft-paper/msc_d5_end2end_p1e-3"
circuit = tsim.Circuit.from_file(f"{path}.stim")
mask = np.load(f"{path}.postselection.npy")
start = time.perf_counter()
sampler = circuit.compile_detector_sampler(strategy="statevector", seed=0)
print(f"compile {time.perf_counter() - start:.1f} s")  # 26 s at e394e84
detectors, observables = sampler.sample(
    65_536, batch_size=16_384, separate_observables=True, postselection_mask=mask
)
accepted = ~detectors[:, mask].any(axis=1)
print(f"acceptance {accepted.mean():.4f}")
```

## Earlier scripts

These run on this branch, but their docstrings and notes describe earlier
states of the sampler.

- `statevector/benchmark.py` and `statevector/benchmark_clifft.py`: the
  five-circuit Tsim and Clifft benchmark of 2026-10-10 (one lane per shot),
  checked against Clifft 0.10.0's acceptances. Results are in
  `notes/statevector_results.md`.
- `trajectory/`: `sharing_stats.py` (how much sharing the circuits allow),
  `structure.py` (what the ZX representation hands the executor),
  `breakdown.py` (where the time goes with one lane per shot) and
  `benchmark_sharing.py` (one lane vs shared). Pass workload names before
  options, for example `sharing_stats.py distillation --lanes 4096`.

## Notes

- `notes/trajectory_sharing_investigation.md`: the trajectory-sharing
  investigation (JAX prototype, before the numba executor).
- `notes/statevector_results.md`: the 2026-10-10 same-machine comparison
  without trajectory sharing, with active widths and logical-error checks.
- `notes/clifft_bench_tsim_gpu_on_cpu.md`: how to run clifft-bench's own
  `experiments/tsim-gpu` harness on the JAX CPU backend against a local Tsim
  checkout. These are the stabilizer-rank strategies (`cat5`, `cutting`), as of
  2026-10-01.
