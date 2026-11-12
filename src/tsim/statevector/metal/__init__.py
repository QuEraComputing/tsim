"""The statevector executor on an Apple GPU, through Metal.

The noise sampler (``noise``) and every component's tape (``codegen``, written
by ``tsim.statevector.gpu.tape``) run as Metal kernels that PyObjC compiles at
runtime (``runtime``); ``MetalShotSampler`` (``sampler``) replaces
``ShotSampler`` behind the compiled samplers of ``device="auto"`` or
``"metal"`` whenever PyObjC sees an Apple GPU and every component fits. Amplitudes are complex64. It needs Apple silicon,
where Tsim depends on ``pyobjc-framework-Metal``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsim.statevector.gpu.tape import MAX_WIDTH, RESOLVABLE, rarest_branch, supports
from tsim.statevector.metal import runtime

if TYPE_CHECKING:
    import numpy as np

    from tsim.noise.channels import ChannelSampler
    from tsim.statevector.metal.sampler import MetalShotSampler
    from tsim.statevector.shots import ShotLayout, ShotSampler


def shot_sampler(
    host: ShotSampler,
    channels: ChannelSampler,
    num_detectors: int,
    seed: np.random.SeedSequence,
    *,
    required: bool,
) -> ShotSampler | MetalShotSampler:
    """Return a Metal shot sampler of ``host``'s program, or ``host`` where Metal cannot run it.

    Without ``required`` (``device="auto"``) it returns ``host`` silently; with it
    (``device="metal"``) it raises instead. A program without shot words has
    nothing to sample on a GPU and keeps ``host`` either way.

    Raises:
        RuntimeError: With ``required``, if this process cannot use an Apple GPU:
            no Apple GPU, no PyObjC's Metal, or forked after its parent used the GPU.
        ValueError: With ``required``, if the GPU executor cannot run the program:
            a component wider than ``MAX_WIDTH``, a rotation by an angle too small
            for float32 amplitudes (``RESOLVABLE``), or noise beyond the GPU
            tables' limits.
        MemoryError: With ``required``, if the GPU runs out of memory.

    """
    reason = runtime.unavailable()
    if reason is not None:
        if required:
            raise RuntimeError(f"Cannot sample on Metal: {reason}.")
        return host
    if host.layout.num_words == 0:
        return host
    reason = _unsupported(host.layout)
    if reason is not None:
        if required:
            raise ValueError(f"Cannot sample on Metal: {reason}.")
        return host
    from tsim.statevector.metal.sampler import MetalShotSampler

    try:
        return MetalShotSampler(host, channels, num_detectors, seed, required=required)
    except MemoryError:
        if required:
            raise
    except ValueError as error:
        # NOTE: noise beyond the GPU tables' limits (channels of more than 127
        # outcomes, outcomes that flip more than 255 words).
        if required:
            raise ValueError(f"Cannot sample on Metal: {error}.") from error
    return host


def _unsupported(layout: ShotLayout) -> str | None:
    """Return why the GPU executor cannot run ``layout``, or ``None`` if it can."""
    if supports(layout):
        return None
    width = max(c.width for c in layout.components)
    if width > MAX_WIDTH:
        return (
            f"a component needs {width} active qubits, more than the "
            f"{MAX_WIDTH} that the GPU executor supports"
        )
    return (
        f"a rotation has a branch of weight {rarest_branch(layout):.1e}, "
        f"below the {RESOLVABLE:.0e} that float32 amplitudes resolve"
    )
