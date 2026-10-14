# Benchmarking tsim on CPU with clifft-bench

How to benchmark this tsim checkout on the JAX **CPU** backend (no GPU) with
[clifft-bench](https://github.com/unitaryfoundation/clifft-bench)'s `experiments/tsim-gpu`
harness. The harness uses the same eight QEC circuits and the same counting
semantics as clifft-bench's published H100 tsim run.

These are development numbers: the harness is patched for CPU and run with
`--allow-dirty`, so results cannot be finalized into clifft-bench's `results/`.
They are not comparable to the published H100 series except as a rough reference.

## 1. Optimal configuration for this machine

Calibrated on 2026-09-30 with a full batch-size / shots-per-call sweep (section 3).

| | |
|---|---|
| Machine | Apple M4 Pro (10 performance + 4 efficiency cores), 48 GB RAM, macOS 15.7 |
| Software | tsim `3dacc1a`, JAX/jaxlib 0.11.1 (CPU), NumPy 2.5.2, Python 3.12.11 |
| Harness | clifft-bench `ad0647f` + the CPU patch below |

Only the five circuits that compile within the harness's 120 s limit are run:

| Workload | Strategy | Shots per call | Batch size | Rate (attempted shots/s) | H100 reference |
|---|---|---:|---:|---:|---:|
| `surface-code-d7-r7-p1e-3` | cat5 | 262,144 | n/a (direct path) | 1.52M | 1.91M |
| `coherent-surface-d3-r1-p1e-3-rz2e-2` | cat5 | 262,144 | n/a (direct path) | 95.8M | 41.8M |
| `coherent-surface-d5-r1-p1e-3-rz2e-2` | cat5 | 262,144 | n/a (direct path) | 66.6M | 24.7M |
| `distillation-color-code-85q-p5e-2` | cat5 | 65,536 | 8,192 | 603k | 1.39M |
| `msc-d3-inject-cultivate-p1e-3` | cutting | 4,096 | 64 | 567 | 87k |

Notes:

- **Direct path**: the three Clifford-only circuits use tsim's direct CPU path
  (`execution_path=direct-cpu`, no compiled components). `batch_size` has no
  effect there; only shots per public `sample()` call matters, and larger calls
  are faster (surface d7/r7: 734k/s at 8,192 → 1.49M/s at 65,536 → 1.52M/s at
  262,144). These rows compare host CPUs, not CPU vs GPU, because the H100 run
  also used the CPU for them.
- **Distillation**: rates are flat for batch 2,048–8,192 (560k–600k/s) and drop
  sharply at larger batches (16,384: 310k/s; 65,536: 87k/s). Shots per call also
  matters: batch 8,192 with 8,192-shot calls measured only 98k/s.
- **Cultivation d3**: rates are nearly flat (440–475/s) from batch 128 to 1,024,
  and batch 64 is slightly best. Every batch shape pays its own ~30 s JIT. The
  4,096-shot call keeps each call near 7 s.
- **tsim's default batch size is a poor choice on CPU.** It is sized from free
  RAM (`CompiledDetectorSampler._estimate_batch_size`): about 1,400 for
  cultivation d3, which gave the slowest rate (335/s), and about 235k for
  distillation, far past the peak.
- **Strategies**: `cutting` for cultivation (`cat5` and `bss` fail to compile in
  time), `cat5` for everything else.
- **Excluded**: cultivation d5 (`cutting`) and coherent d3/r3 and d5/r5 (`cat5`)
  exceed the 120 s compile limit, as they did on the H100 host. They used 5–8 GiB
  of RSS while compiling.
- **Threads**: nothing is pinned (macOS has no CPU affinity). BLAS/OpenMP helper
  threads are 1, but XLA's CPU backend uses all cores for the component circuits
  (distillation, cultivation). The direct-path circuits run on roughly one core.

## 2. Run the optimal configuration

### One-time setup

clifft-bench must be checked out **next to** this repo, because the patch points
its tsim dependency at `../../../tsim`:

```bash
cd "$(git rev-parse --show-toplevel)/.."     # parent of the tsim checkout
git clone https://github.com/unitaryfoundation/clifft-bench.git
cd clifft-bench
git switch -c local-tsim ad0647fbb99cf33dda212c8ddd4267266f205683
git apply <<'PATCH'
diff --git a/experiments/tsim-gpu/pyproject.toml b/experiments/tsim-gpu/pyproject.toml
--- a/experiments/tsim-gpu/pyproject.toml
+++ b/experiments/tsim-gpu/pyproject.toml
@@ -14,7 +14,7 @@ cuda13 = ["jax[cuda13]"]
 test = ["pytest>=8"]

 [tool.uv.sources]
-bloqade-tsim = { git = "https://github.com/QuEraComputing/tsim.git", rev = "44f64ba79c91e6ada77d6b609595ad0a3d767c11" }
+bloqade-tsim = { path = "../../../tsim", editable = true }

 [tool.uv]
 conflicts = [[{ extra = "cuda12" }, { extra = "cuda13" }]]
diff --git a/experiments/tsim-gpu/tests/test_tsim_gpu.py b/experiments/tsim-gpu/tests/test_tsim_gpu.py
--- a/experiments/tsim-gpu/tests/test_tsim_gpu.py
+++ b/experiments/tsim-gpu/tests/test_tsim_gpu.py
@@ -203,8 +203,7 @@ def test_real_worker_compiles_once_and_separates_final_samples(tmp_path, monkeyp
     assert calls == [1]
     assert len(result["samples"]) == 2
     assert len(result["tuning"]) == (1 if recovery else 2)
-    if recovery:
-        assert result["batch_candidates"] == [8]
+    assert result["batch_candidates"] == ([8] if recovery else [8, 32])
     assert all(c["warmup_seconds"] > 0 for c in result["tuning"])
     assert result["runtime_metadata"]["compiled_phase_dtypes"]

diff --git a/experiments/tsim-gpu/tsim_gpu/common.py b/experiments/tsim-gpu/tsim_gpu/common.py
--- a/experiments/tsim-gpu/tsim_gpu/common.py
+++ b/experiments/tsim-gpu/tsim_gpu/common.py
@@ -11,6 +11,9 @@ from datetime import UTC, datetime
 from pathlib import Path

 ROOT = Path(__file__).resolve().parents[3]
+# NOTE: local-tsim branch. CLIFFT_TSIM_PLATFORM=cpu runs on the JAX CPU backend
+# (development numbers only), and the local editable Tsim checkout is accepted.
+PLATFORM = os.environ.get("CLIFFT_TSIM_PLATFORM", "cuda")
 EXPERIMENT = Path(__file__).resolve().parents[1]
 TSIM_COMMIT = "44f64ba79c91e6ada77d6b609595ad0a3d767c11"
 # The paper's successful circuits first; old failures remain in the experiment.
@@ -27,7 +30,7 @@ WORKLOADS = (
 ENVIRONMENT = {
     "PYTHONHASHSEED": "0",
     "XLA_FLAGS": "",
-    "JAX_PLATFORMS": "cuda",  # Failure to initialize CUDA must be fatal.
+    "JAX_PLATFORMS": PLATFORM,  # Failure to initialize CUDA must be fatal.
     "JAX_ENABLE_COMPILATION_CACHE": "false",  # Cold JIT timing is reproducible.
     "OMP_NUM_THREADS": "1",
     "OPENBLAS_NUM_THREADS": "1",
@@ -92,6 +95,11 @@ def workloads() -> dict:


 def require_gpu(jax) -> list:
+    if PLATFORM == "cpu":
+        devices = jax.devices()
+        if any(device.platform != "cpu" for device in devices):
+            raise RuntimeError("CLIFFT_TSIM_PLATFORM=cpu initialized a non-CPU device")
+        return devices[:1]
     try:
         devices = jax.devices()
     except Exception as error:
@@ -119,11 +127,20 @@ def environment_metadata(*, allow_dirty: bool) -> dict:
         raise ValueError("Commit the runner/configuration before collection (or use --allow-dirty)")
     distribution = importlib.metadata.distribution("bloqade-tsim")
     direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
-    if direct_url.get("vcs_info", {}).get("commit_id") != TSIM_COMMIT:
+    local_tsim = direct_url.get("url", "").startswith("file://")
+    if local_tsim:
+        tsim_path = Path(direct_url["url"].removeprefix("file://"))
+        direct_url["local_commit"] = subprocess.check_output(
+            ["git", "rev-parse", "HEAD"], text=True, cwd=tsim_path
+        ).strip()
+        direct_url["local_dirty"] = bool(
+            subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=tsim_path)
+        )
+    elif direct_url.get("vcs_info", {}).get("commit_id") != TSIM_COMMIT:
         raise ValueError(
             "Tsim installation does not match the pinned source commit; uv sync --locked"
         )
-    if distribution.version != "0.1.5":
+    if distribution.version != "0.1.5" and not local_tsim:
         raise ValueError("unexpected Tsim version")
     cpuinfo = Path("/proc/cpuinfo")
     cpu_model = platform.processor()
@@ -160,7 +177,9 @@ def environment_metadata(*, allow_dirty: bool) -> dict:
         "jax_devices": [
             {"id": d.id, "kind": d.device_kind, "platform": d.platform} for d in devices
         ],
-        "nvidia_smi": command(
+        "nvidia_smi": None
+        if PLATFORM == "cpu"
+        else command(
             [
                 "nvidia-smi",
                 "--query-gpu=uuid,name,driver_version,memory.total",
diff --git a/experiments/tsim-gpu/tsim_gpu/worker.py b/experiments/tsim-gpu/tsim_gpu/worker.py
--- a/experiments/tsim-gpu/tsim_gpu/worker.py
+++ b/experiments/tsim-gpu/tsim_gpu/worker.py
@@ -157,9 +157,9 @@ def run(request: dict, checkpoint: Path) -> dict:
         # Skip shapes exceeding Tsim's own conservative device-memory estimate.
         # This is an estimate, not a promise; the controller also monitors RSS.
         estimated_capacity = int(sampler._estimate_batch_size())
-        candidates = sorted(
-            set(min(b, shots) for b in options["batch_sizes"]) | {min(shots, estimated_capacity)}
-        )
+        # NOTE: local-tsim branch. Only the requested batch sizes are tried; Tsim's
+        # memory-based estimate is no longer added as a candidate (it was slower on CPU).
+        candidates = sorted(set(min(b, shots) for b in options["batch_sizes"]))
         if "recovery_batch_sizes" in request:
             # Do not reintroduce the automatic large shape that killed tuning.
             candidates = request["recovery_batch_sizes"]
PATCH
cd experiments/tsim-gpu
uv sync --no-config --python 3.12 --extra test
.venv/bin/python -c "import tsim, jax; print(tsim.__file__, jax.devices())"
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q     # expect 26 passed
```

What the patch does:

- Installs this tsim checkout in editable mode, so tsim edits take effect without
  reinstalling.
- Adds an opt-in `CLIFFT_TSIM_PLATFORM=cpu` that runs workers with
  `JAX_PLATFORMS=cpu` and accepts the CPU device.
- Skips `nvidia-smi`.
- Records the local tsim commit and dirty flag instead of requiring the pinned
  commit.
- Tries **only** the batch sizes given with `--batch-sizes`. Upstream also adds
  tsim's memory-based estimate (`min(shots per call, _estimate_batch_size())`)
  as a candidate. On this CPU that size was 1.7× slower for cultivation and 7×
  slower for distillation, and it cost an extra JIT per run. The estimate is still
  used to skip candidates that would not fit (`memory-estimate-skip`).

Without the environment variable, the harness otherwise behaves as upstream.

`uv sync` notes:

- `--no-config` ignores a global `~/.config/uv/uv.toml` (for example, private
  package indexes that return 401).
- Do not pass `--locked`, because the patch changes the tsim source.
- Do not install the `cuda12`/`cuda13` extras.
- The `tsim.__file__` check must point into this repo's `src/tsim`.

### Run (~4 min fast, ~15 min full)

Run from `clifft-bench/experiments/tsim-gpu`. The output directory must be
outside the clifft-bench checkout:

```bash
export CLIFFT_TSIM_PLATFORM=cpu
OUT="$(cd ../../.. && pwd)/clifft-bench-results/optimal-$(date +%Y%m%d-%H%M%S)"
COMMON=(--host-label 'local / Apple M4 Pro 14c 48GB / JAX CPU' --allow-dirty)
# Fast mode: 1 warm probe, 3 x 10 s final runs. Set MEASURE=() for the harness
# defaults (3 probes, 5 x 30 s final runs, ~16 min).
MEASURE=(--probe-repetitions 1 --sample-seconds 10 --repetitions 3)

# Direct-path circuits: cat5, 262,144 shots per call (batch_size has no effect).
caffeinate -i .venv/bin/python -m tsim_gpu "$OUT/direct" "${COMMON[@]}" "${MEASURE[@]}" \
  --strategies cat5 --shots-per-call 262144 \
  --workload surface-code-d7-r7-p1e-3 \
  --workload coherent-surface-d3-r1-p1e-3-rz2e-2 \
  --workload coherent-surface-d5-r1-p1e-3-rz2e-2

# Distillation: cat5, batch 8,192, 65,536 shots per call (the default).
caffeinate -i .venv/bin/python -m tsim_gpu "$OUT/distillation" "${COMMON[@]}" "${MEASURE[@]}" \
  --strategies cat5 --batch-sizes 8192 \
  --workload distillation-color-code-85q-p5e-2

# Cultivation d3: cutting, batch 64, 4,096 shots per call.
caffeinate -i .venv/bin/python -m tsim_gpu "$OUT/cultivation-d3" "${COMMON[@]}" "${MEASURE[@]}" \
  --strategies cutting --batch-sizes 64 --shots-per-call 4096 --call-timeout-seconds 300 \
  --workload msc-d3-inject-cultivate-p1e-3
```

Each circuit gets one compile, a warmup per batch shape, warm probes to pick the
batch, and then the final runs. Those are **3 × ≥10 s** in fast mode and
**5 × ≥30 s** with the harness defaults. The final runs take 80% of the full run's
time, and in the full run they agreed to within 2.4–3.5% (max − min over median).
That is why fast mode shortens them. The rate is
attempted shots (including postselection discards) per second of
`CompiledDetectorSampler.sample` plus the NumPy count reduction. It excludes
compilation and JIT warmup.

JIT never enters the timed numbers. For each batch size, the worker first makes
two untimed, discarded `sample()` calls with the same shots per call: one with
postselection short-circuiting disabled, so the JAX path compiles even if every
shot is discarded early, and one on the normal path. Their time is recorded as
`warmup_seconds`. tsim pads the last partial batch to the full batch size, so
each batch size is a single JAX shape. Probes and final runs reuse the same
sampler in the same process.

### Read the results

```bash
.venv/bin/python - "$OUT" <<'EOF'
import glob, json, sys
for path in sorted(glob.glob(f"{sys.argv[1]}/*/raw/*/*/attempt-*/result.json")):
    r = json.load(open(path))
    workload, strategy = path.split("/")[-4:-2]
    if r["status"] != "success":
        print(f"{workload:40s} {strategy:8s} {r['status']}: {r['error']['message']}")
        continue
    samples = r["samples"]
    attempted = sum(s["attempted_shots"] for s in samples)
    accepted = sum(s["accepted_shots"] for s in samples)
    errors = sum(s["logical_errors"] for s in samples)
    print(
        f"{workload:40s} {strategy:8s} batch={r['selected_batch_size']:<7}"
        f" {r['summary']['median_attempted_shots_per_second']:>14,.0f} shots/s"
        f"  (MAD {r['summary']['mad_attempted_shots_per_second']:,.0f})"
        f"  accept={accepted / attempted:.4f}"
        f"  LER={errors / accepted if accepted else 0:.2e}"
        f"  compile={r['compile_seconds']:.1f}s"
    )
EOF
```

Measured wall times for exactly these commands, with the patch above (2026-10-01):

| Circuit | Fast mode | Full mode | What dominates |
|---|---:|---:|---|
| surface d7/r7 | 37 s | 160 s | final runs (30 s / 150 s) |
| coherent d3/r1 | 40 s | 188 s | final runs, plus per-call checkpoint writes |
| coherent d5/r1 | 39 s | 178 s | final runs |
| distillation | 38 s | 161 s | final runs |
| cultivation d3 | 98 s | 231 s | 8 s tsim compile + ~25 s JIT warmup + probes + final runs |
| **Total, incl. process startup** | **248 s** | **921 s** | |

Before the patch dropped tsim's batch estimate from the candidates, the same
commands took 323 s and 979 s. Cultivation alone lost about 60 s of tuning.

Rates from three runs of exactly these commands (attempted shots/s):

| Circuit | Full, idle | Full, load avg 6–11 | Fast, load avg 6–10 |
|---|---:|---:|---:|
| surface d7/r7 | 1.56M | 1.49M | 1.36M |
| coherent d3/r1 | 98.3M | 86.2M | 87.7M |
| coherent d5/r1 | 68.4M | 60.3M | 58.1M |
| distillation | 629k | 521k | 574k |
| cultivation d3 | 577 | 495 | 519 |

The idle full run is within about 4% of section 1. Background load (here,
browser and IDE processes) costs 5–20% and dominates the fast/full difference.
Under the same load, fast and full mode agree to within ±10% in both
directions.

Compare against the table in section 1. As a correctness check, acceptance and
logical-error rates should agree with the H100 run within statistical error:

| Workload | Acceptance | Logical-error rate |
|---|---:|---:|
| surface d7/r7 | 0.0879 | 0 |
| coherent d3/r1 | 0.9743 | ~1.7e-8 |
| coherent d5/r1 | 0.9287 | 0 |
| distillation | 0.0798 | 0.422 |
| cultivation d3 | ~0.686 | ~8e-7 (expect 0 in a short run) |

Raw evidence is in `OUT/STEP/raw/WORKLOAD/STRATEGY/attempt-NNN/`
(`request.json`, `result.json`, `worker.log`). `OUT/STEP/summary.json` holds the
per-circuit selection.

## 3. Recalibrate shots per call and batch size (~20 min)

Rerun this after changes that could move the optimum: tsim compile/sampling
changes, JAX upgrades, or a different machine. It scans the knob that matters
for each circuit type, keeping the strategies from section 1:

```bash
export CLIFFT_TSIM_PLATFORM=cpu
OUT="$(cd ../../.. && pwd)/clifft-bench-results/sweep-$(date +%Y%m%d-%H%M%S)"
COMMON=(--host-label 'local / Apple M4 Pro 14c 48GB / JAX CPU' --allow-dirty)
DIRECT=(--workload surface-code-d7-r7-p1e-3
        --workload coherent-surface-d3-r1-p1e-3-rz2e-2
        --workload coherent-surface-d5-r1-p1e-3-rz2e-2)

# Direct path: scan shots per call (batch_size has no effect). 3 x 10 s final runs each.
for shots in 8192 65536 262144; do
  caffeinate -i .venv/bin/python -m tsim_gpu "$OUT/direct-shots-$shots" "${COMMON[@]}" \
    --strategies cat5 --shots-per-call "$shots" --sample-seconds 10 --repetitions 3 "${DIRECT[@]}"
done

# Distillation: power-of-two batch scan at 65,536 shots per call.
caffeinate -i .venv/bin/python -m tsim_gpu "$OUT/distillation" "${COMMON[@]}" \
  --strategies cat5 --prepare-budget-seconds 3600 \
  --batch-sizes 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 \
  --workload distillation-color-code-85q-p5e-2

# Cultivation d3: batch scan at 4,096 shots per call (~30 s JIT per batch shape).
caffeinate -i .venv/bin/python -m tsim_gpu "$OUT/cultivation-d3" "${COMMON[@]}" \
  --strategies cutting --shots-per-call 4096 --call-timeout-seconds 300 \
  --prepare-budget-seconds 3600 --batch-sizes 64 128 256 512 1024 2048 4096 \
  --workload msc-d3-inject-cultivate-p1e-3
```

Measured wall time on 2026-10-01 was 1,196 s (with the patch):

| Step | Time | Notes |
|---|---:|---|
| `direct-shots-8192` | 233 s | many tiny calls: per-call checkpoint writes add ~75 s for coherent d3/r1 |
| `direct-shots-65536` | 145 s | |
| `direct-shots-262144` | 119 s | |
| `distillation` | 227 s | 11 candidates, 72 s tuning, 150 s final |
| `cultivation-d3` | 465 s | 5 candidates × ~57 s (~30 s JIT each), 168 s final |

That sweep reproduced the optimum:
- **Direct circuits:** 262,144 shots per call was fastest for all three.
- **Distillation:** the plateau was 2,048–8,192, with 8,192 selected (619k/s).
- **Cultivation d3:** batch 64 was selected (596/s).

Why these settings:

- `--prepare-budget-seconds 3600` is needed because every batch candidate pays
  its own JIT and probes. That would exceed the default 600 s for cultivation.
- `--call-timeout-seconds 300` and 4,096 shots per call keep a cultivation call
  (~500 shots/s) inside the deadline. The default 65,536 shots per call would
  take about 130 s and be killed at the 120 s limit.
- Candidates above tsim's memory-based estimate are skipped before JIT and
  recorded as `memory-estimate-skip`. For cultivation d3, that is 2,048 and 4,096.
- Stay below about 750k shots per call for surface d7/r7. Larger calls exceed the
  harness's default 256 MiB output limit unless `--max-output-mib` is raised.

To look at the scan, print every probe:

```bash
.venv/bin/python - "$OUT" <<'EOF'
import glob, json, sys
for path in sorted(glob.glob(f"{sys.argv[1]}/*/raw/*/*/attempt-*/result.json")):
    r = json.load(open(path))
    request = json.load(open(path.replace("result.json", "request.json")))
    step, _, workload, strategy = path[len(sys.argv[1]) + 1 :].split("/")[:4]
    shots = request["options"]["shots_per_call"]
    print(f"\n{step} | {workload} | {strategy} | shots/call={shots} | {r['status']}")
    for t in r.get("tuning", []):
        rate = t.get("median_attempted_shots_per_second")
        warmup = t.get("warmup_seconds")
        print(
            f"  batch={t['batch_size']:>7} {t['status']:22s}"
            f" probe={rate or 0:>14,.0f}/s warmup={warmup or 0:.1f}s"
        )
    if r["status"] == "success":
        print(f"  selected {r['selected_batch_size']}: final"
              f" {r['summary']['median_attempted_shots_per_second']:,.0f}/s")
EOF
```

For the direct circuits, compare the final rates across the `direct-shots-*`
steps. For distillation and cultivation, the harness picks the batch with the
highest probe median; check that it sits on a plateau rather than on a noisy
single point. Then update the table in section 1 and the flags in section 2.

The probes are short (3 × 1 s). Batches within a few percent of each other are
effectively tied, so prefer the smaller batch on a flat plateau.

To retry the excluded circuits, use `--workload msc-d5-inject-cultivate-p1e-3`
with `--strategies cutting`, and the coherent d3/r3 and d5/r5 workloads with
`--strategies cat5`. Raise `--compile-timeout-seconds` above 120. They will not
succeed at the default limit.

## Operating notes

- **One run at a time, on an idle machine.** Concurrent runs, including other
  benchmarks, compete for the CPU and distort each other's timings. Check that
  `uptime` reports a load average well below the core count before starting.
  With an unrelated single-core benchmark running and a load average of ~13,
  every rate dropped 13–21% even though each run looked internally consistent
  (small MAD). Keep the Mac on power; `caffeinate -i` prevents idle sleep,
  but closing the lid on battery still sleeps.
- **Output directories.** Use a new output directory for every distinct set of
  flags. Rerunning an *identical* command on the same directory resumes it and
  reuses completed results. Changing flags, harness code, tsim installation
  identity, or the host label fails with "resume fingerprint changed".
- **Stopping.** Workers run in their own process group. After Ctrl-C or killing
  the controller, check with `ps -Ao pid,command | grep tsim_gpu.worker` and
  run `pkill -f tsim_gpu.worker` if needed.
- **Failures are data.** `timeout`, `budget-exhausted`, `memory-limit`, and
  `memory-estimate-skip` are recorded resource limits, not zero rates.
- **Finalizing.** `python -m tsim_gpu.finalize` rejects `--allow-dirty` runs by
  design. Keep these results local.
