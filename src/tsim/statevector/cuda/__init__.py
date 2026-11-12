"""The statevector executor on a CUDA GPU.

The noise sampler (``noise``) and every component's tape (``codegen``, written
by ``tsim.statevector.gpu.tape``) run as CUDA kernels that CuPy compiles with
NVRTC (``runtime``); ``CudaShotSampler`` (``sampler``) replaces ``ShotSampler``
behind the compiled samplers of ``device="auto"`` or ``"cuda"`` whenever CuPy
sees a GPU and every component fits. Amplitudes are complex64. It needs CuPy
and an NVIDIA GPU (``pip install bloqade-tsim[cuda]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsim.statevector.cuda import runtime
from tsim.statevector.gpu.tape import unsupported

if TYPE_CHECKING:
    import numpy as np

    from tsim.noise.channels import ChannelSampler
    from tsim.statevector.cuda.sampler import CudaShotSampler
    from tsim.statevector.shots import ShotSampler


def shot_sampler(
    host: ShotSampler,
    channels: ChannelSampler,
    num_detectors: int,
    seed: np.random.SeedSequence,
    *,
    required: bool,
) -> ShotSampler | CudaShotSampler:
    """Return a CUDA shot sampler of ``host``'s program, or ``host`` where CUDA cannot run it.

    Without ``required`` (``device="auto"``) it returns ``host`` silently; with it
    (``device="cuda"``) it raises instead. A program without shot words (one
    with no outputs; direct outputs are shot-word bits too) has nothing to
    sample on a GPU and keeps ``host`` either way.

    Raises:
        RuntimeError: With ``required``, if this process cannot use a CUDA GPU:
            no CuPy, no device, or forked after its parent used the GPU.
        ValueError: With ``required``, if the GPU executor cannot run the program:
            a component wider than ``MAX_WIDTH``, a rotation by an angle too small
            for float32 amplitudes (``RESOLVABLE``), or noise beyond
            the GPU tables' limits.
        MemoryError: With ``required``, if the GPU runs out of memory.

    """
    reason = runtime.unavailable()
    if reason is not None:
        if required:
            raise RuntimeError(f"Cannot sample on CUDA: {reason}.")
        return host
    if host.layout.num_words == 0:
        return host
    reason = unsupported(host.layout)
    if reason is not None:
        if required:
            raise ValueError(f"Cannot sample on CUDA: {reason}.")
        return host
    from tsim.statevector.cuda.sampler import CudaShotSampler

    try:
        return CudaShotSampler(host, channels, num_detectors, seed, required=required)
    except ValueError as error:
        # NOTE: noise beyond the GPU tables' limits (channels of more than 127
        # outcomes, outcomes that flip more than 255 words).
        if required:
            raise ValueError(f"Cannot sample on CUDA: {error}.") from error
    except Exception as error:
        if not runtime.out_of_memory(error):
            raise
        if required:
            raise MemoryError(f"Cannot sample on CUDA: {error}") from error
    return host
