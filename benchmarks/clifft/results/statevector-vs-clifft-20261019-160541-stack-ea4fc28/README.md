# Run of 2026-10-19 (Apple M4 Pro, 14 cores)

- `clifft*.jsonl`, `tsim-*.jsonl`, `report.md`: `run_all.sh` at tsim-perf ea4fc28c with
  Clifft 0.11.0 and its default passes.
- `clifft-allpasses.jsonl`: Clifft with `run_clifft.py --passes all` (adds
  `ActiveWidthSchedulePass`); `clifft-all-large.jsonl`: the same with calls of 40 Tsim
  batches (`scripts/run_clifft_large.py`). `report-allpasses.md` compares them.
- `tsim-all-large.jsonl`: Tsim at ea4fc28c with calls of 40 batches
  (`scripts/run_tsim_large.py`).
- `base-*.jsonl`: `run_tsim.py` on trajectory-sharing 35db8b6, the base of QuEraComputing/tsim#199.
- `tsim-perf-e89ee4f0/`: `run_tsim.py` and the 40-batch calls on tsim-perf e89ee4f0
  (numba executor, no JAX), and `default-batch/`: `scripts/bench_default.py` with
  `batch_size=None` before (ea4fc28c) and after (e89ee4f0). `report-pr199.md` holds the
  tables of the PR description (`scripts/pr_tables.py`).
