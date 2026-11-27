---
name: testing-tsim-correctness
description: Chooses the oracle and writes statistical tests that show a Tsim change samples the right distribution. Use when adding or changing a gate, noise channel, sampler option, scheduler, plan or kernel in Tsim, when fixing a wrong-results bug, or when asked whether Tsim's output can be trusted for a circuit.
metadata:
  internal: true
---

# Testing Tsim's correctness

Performance work is only worth anything if the samples stay right. Every change to
`src/tsim` that can change samples needs a test against an independent oracle.

## Pick the oracle

| Situation | Oracle | Where |
|---|---|---|
| A component, plan or kernel of `active_statevector` | dense state vector | `test/helpers/statevector.py`: `plan_distribution`, `ops_distribution`, `reference_rotate`, `reference_measure`, `assert_distributions_close` |
| Whole circuit, few qubits, few T gates | `strategy="cat5"`, or `VecSampler` | `test/integration/test_statevector_sampler.py` shows both |
| Clifford circuit or Clifford stand-in | Stim | compile with Stim and Tsim; pass the reference-sample flags to Tsim |
| Single gates | unitaries | `test/helpers/gate_matrices.py`, `test/integration/test_gate_unitaries.py` |
| Random coverage | `gen_stim_circuit` | `test/helpers/gen.py` |

Exact probabilities beat sampled references: compare samples to an exact
distribution with `assert_matches_distribution`, and two sampled sets with
`assert_counts_match` (both 5σ per outcome). Run `cat5` only on circuits with a
handful of T gates; its memory grows exponentially.

## Invariants to test when you touch them

- Same seed gives the same samples for any thread count and any `batch_size` on the
  CPU. On a GPU the samples depend on `batch_size`; same seed and same `batch_size`
  give the same samples.
- Bit-packed and unpacked outputs agree. Every `sample()` call draws new shots, so
  compare two fresh samplers with the same seed, not two calls on one sampler.
- Postselection returns only the accepted rows; compare accepted-row distributions.
- Detectors and observables are raw parities. The reference-sample flags give Stim's
  flips; use them only for detectors that are deterministic without noise.
- Direct outputs never enter a state vector (`sampler.stats.num_direct_outputs`).

## Writing the test

- For a bug: reproduce it in the smallest circuit you can, write the test, and show
  it fails before the fix. Run the new test file against `origin/main` in a
  separate worktree; do not `git stash` while other jobs run in this checkout.
- Fix the seed and choose the shot count for the tolerance; a flaky statistical
  test is a broken test.
- Stim merges consecutive identical instructions into one with several targets, so
  write circuits that exercise both single- and multi-target paths.
- GPU tests: `test/conftest.py` keeps every test outside
  `test/integration/test_metal_sampler.py` on the CPU. In the Metal tests compile the
  GPU side with `device="metal"`, so a silent fallback cannot make the test pass on the
  CPU, and the reference with `device="cpu"`; compare them with `assert_counts_match`
  (GPU samples never equal CPU samples bit for bit).
- Keep circuits small (a few qubits, tens of thousands of shots) so the suite stays
  near 90 s. Long randomized campaigns belong in a job, not in the suite:
  `uv run tsim-dev job start fuzz --mem 4G -- <command>`.

## Before you report

`uv run tsim-dev check` must pass. Say which oracle you compared against, the
circuits, the shot counts, and anything you could not cover.
