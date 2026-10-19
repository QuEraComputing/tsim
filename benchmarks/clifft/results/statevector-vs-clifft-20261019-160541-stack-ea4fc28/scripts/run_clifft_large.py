"""Time Clifft on all threads with the same call sizes as run_tsim_large.py."""

import json
import sys
import time

import clifft

sys.path.insert(0, "benchmarks/clifft/comparison")
from run_clifft import measure, pass_manager  # noqa: E402
from workloads import WORKLOADS  # noqa: E402

from run_tsim_large_sizes import SIZES  # noqa: E402

out_path, seconds = sys.argv[1], float(sys.argv[2])
# NOTE: Clifft already keeps every core busy on SymFT with its small calls.
for name, shots in SIZES.items():
    if name.startswith("symft"):
        continue
    workload = WORKLOADS[name]
    hir = clifft.trace(clifft.parse(workload.path.read_text()))
    pass_manager("all").run(hir)
    mask = [int(workload.postselect)] * int(hir.num_detectors)
    program = clifft.lower(hir, postselection_mask=mask)
    for batch_size in (1, 2048):
        try:
            record = measure(program, shots, seconds, "auto", batch_size, 12345)
        except ValueError as error:
            print(f"{name} batch={batch_size}: {error}")
            continue
        record.update(simulator="clifft", workload=name, threads="all",
                      batch=batch_size, shots_per_call=shots, passes="all",
                      width=int(program.peak_active_width))
        with open(out_path, "a") as out:
            out.write(json.dumps(record) + "\n")
        print(f"{name:26s} batch {batch_size:>5} shots/call {shots:>10,} "
              f"{record['shots_per_second']:>14,.0f} shots/s "
              f"cores {record['cores_busy']:.1f}", flush=True)
