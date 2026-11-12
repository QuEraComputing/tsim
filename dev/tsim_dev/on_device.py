"""Run tsim-bench's worker with its samplers compiled for one device.

    python on_device.py DEVICE WORKER [ARGS...]

The worker compiles with Tsim's default device ("auto", which uses a GPU when it
can). This runs it with ``device=DEVICE`` on ``Circuit.compile_sampler`` and
``Circuit.compile_detector_sampler``, where Tsim has that argument (a Tsim
without it samples on the CPU only). It runs in the environment of the checkout
under test, so it uses only the standard library.
"""

from __future__ import annotations

import functools
import inspect
import runpy
import sys


def main() -> None:
    """Run the worker of ``sys.argv[2]`` with the arguments after it."""
    device, worker, *argv = sys.argv[1:]
    # NOTE: not as "__main__", so that the worker's main does not run yet.
    namespace = runpy.run_path(worker, run_name="tsim_bench_worker")
    run = namespace["main"]
    imported = namespace["phase_import"]

    # NOTE: the same name as the worker's, since profiles find the phases by name.
    def phase_import():
        tsim = imported()
        for name in ("compile_sampler", "compile_detector_sampler"):
            method = getattr(tsim.Circuit, name)
            if "device" in inspect.signature(method).parameters:
                setattr(
                    tsim.Circuit, name, functools.partialmethod(method, device=device)
                )
        return tsim

    run.__globals__["phase_import"] = phase_import
    sys.argv = [worker, *argv]
    run()


if __name__ == "__main__":
    main()
