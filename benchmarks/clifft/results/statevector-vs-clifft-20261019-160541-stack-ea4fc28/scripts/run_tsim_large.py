"""Time Tsim on all threads with calls of 40 batches (so every thread has work)."""

import json
import sys

sys.path.insert(0, "benchmarks/clifft/comparison")
from run_tsim import measure  # noqa: E402
from workloads import WORKLOADS  # noqa: E402

BATCHES_PER_CALL = 40
SKIP = {"qv_q20", "coherent_d3_r1", "coherent_d5_r1", "surface_d7_r7"}

out_path, seconds = sys.argv[1], float(sys.argv[2])
for name, workload in WORKLOADS.items():
    if name in SKIP:
        continue
    configurations = [("one lane per shot", False, workload.batch)]
    configurations += [("shared", True, workload.shared[0][0])] if workload.shared else []
    for label, sharing, batch in configurations:
        shots = BATCHES_PER_CALL * batch
        record = measure(name, label, sharing, batch, shots, seconds)
        with open(out_path, "a") as out:
            out.write(json.dumps(record) + "\n")
        print(
            f"{name:26s} {label:18s} batch {batch:>7,} shots/call {shots:>10,} "
            f"{record['shots_per_second']:>14,.0f} shots/s "
            f"cores {record['cores_busy']:.1f}",
            flush=True,
        )
