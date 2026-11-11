"""The statevector executor on a CUDA GPU.

The noise sampler (``noise``) and every component's tape (``codegen``) run as
CUDA kernels that CuPy compiles with NVRTC (``runtime``), from the host code
that the GPU backends share (``tsim.statevector.gpu``); ``CudaShotSampler``
(``sampler``) replaces ``ShotSampler`` behind the compiled samplers whenever
CuPy sees a GPU and every component fits. Amplitudes are complex64. It needs
CuPy and an NVIDIA GPU (``pip install bloqade-tsim[cuda]``).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from tsim.statevector.cuda import runtime
from tsim.statevector.cuda.runtime import available
from tsim.statevector.gpu import FORKED
from tsim.statevector.gpu.tape import MAX_WIDTH, supports

if TYPE_CHECKING:
    import numpy as np

    from tsim.noise.channels import ChannelSampler
    from tsim.statevector.cuda.sampler import CudaShotSampler
    from tsim.statevector.shots import ShotSampler

__all__ = ["available", "shot_sampler"]

# NOTE: the warnings point at the caller of compile_sampler.
_STACKLEVEL = 5


def shot_sampler(
    host: ShotSampler,
    channels: ChannelSampler,
    num_detectors: int,
    seed: np.random.SeedSequence,
) -> ShotSampler | CudaShotSampler:
    """Return a CUDA shot sampler of ``host``'s program when one can run it, else ``host``.

    It warns when a CUDA GPU is there but cannot be used: in a process forked
    after its parent used the GPU, for a component wider than ``MAX_WIDTH``, for
    noise beyond the GPU tables' limits, and when the sampler cannot be set up
    for lack of GPU memory.
    """
    if runtime.forked_after_init():
        warnings.warn(FORKED, stacklevel=_STACKLEVEL)
        return host
    if not runtime.available() or host.layout.num_words == 0:
        return host
    if not supports(host.layout):
        width = max(c.width for c in host.layout.components)
        warnings.warn(
            f"A CUDA GPU is available, but a component needs {width} active "
            f"qubits, more than the {MAX_WIDTH} that the GPU executor supports; "
            "sampling on the CPU.",
            stacklevel=_STACKLEVEL,
        )
        return host
    from tsim.statevector.cuda.sampler import CudaShotSampler

    try:
        return CudaShotSampler(host, channels, num_detectors, seed)
    except ValueError as error:
        # NOTE: noise beyond the GPU tables' limits (channels of more than 127
        # outcomes, outcomes that flip more than 255 words).
        warnings.warn(
            f"The GPU cannot sample this circuit ({error}); sampling on the CPU.",
            stacklevel=_STACKLEVEL,
        )
    except Exception as error:
        if not runtime.out_of_memory(error):
            raise
        warnings.warn(
            "The GPU is out of memory; sampling on the CPU.", stacklevel=_STACKLEVEL
        )
    return host
