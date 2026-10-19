"""Time sample(shots) with the default batch size, sharing on and off."""

import json
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
from workloads import WORKLOADS  # noqa: E402

import tsim  # noqa: E402

out_path, seconds = sys.argv[2], float(sys.argv[3])
for name in sys.argv[4:]:
    workload = WORKLOADS[name]
    circuit = tsim.Circuit.from_file(str(workload.path))
    mask = np.ones(circuit.num_detectors, dtype=bool) if workload.postselect else None
    for sharing in (False, True):
        sampler = circuit.compile_detector_sampler(
            strategy="active_statevector", seed=1, trajectory_sharing=sharing
        )
        sampler.sample(workload.shots, postselection_mask=mask)
        cpu = time.process_time()
        start, calls = time.perf_counter(), 0
        while time.perf_counter() - start < seconds:
            sampler.sample(workload.shots, postselection_mask=mask)
            calls += 1
        elapsed = time.perf_counter() - start
        record = {
            "workload": name, "sharing": sharing, "shots": workload.shots,
            "shots_per_second": calls * workload.shots / elapsed,
            "cores": (time.process_time() - cpu) / elapsed,
        }
        with open(out_path, "a") as out:
            out.write(json.dumps(record) + "\n")
        print(f"{name:26s} sharing={sharing!s:5s} {record['shots_per_second']:>14,.0f} "
              f"shots/s cores {record['cores']:.1f}", flush=True)
