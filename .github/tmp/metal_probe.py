"""TEMPORARY: print what the runner's Metal device supports."""

import subprocess

import numpy as np

from tsim.statevector.metal import runtime as rt

M = rt.metal()
d = rt.device()
print("name:", d.name(), "| low power:", d.isLowPower(), "| headless:", d.isHeadless())
print("unified memory:", d.hasUnifiedMemory(), "| registryID:", d.registryID())
for fam in [f"Apple{i}" for i in range(1, 11)] + ["Mac1", "Mac2", "Common1", "Common2", "Common3", "Metal3", "Metal4"]:
    const = getattr(M, f"MTLGPUFamily{fam}", None)
    if const is not None:
        print(f"  supportsFamily {fam}: {d.supportsFamily_(const)}")
try:
    probe = rt.Library(rt._PROBE).pipeline("lanes")
    print("threadExecutionWidth:", int(probe.threadExecutionWidth()))
    for threads in rt.LANE_PROBE_THREADS:
        out = rt.buffer(4 * threads, np.uint32)
        enc = rt.Encoder()
        enc.dispatch(probe, [out], 4, threads)
        rt.wait(enc.commit())
        print(f"  lanes in order at {threads} threads:", rt.lanes_in_order(out.array, threads))
except Exception as error:  # noqa: BLE001
    print("probe failed:", type(error).__name__, error)
print(subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout)
