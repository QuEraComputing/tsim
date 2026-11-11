"""The statevector executor on an Apple GPU, through Metal.

The noise sampler (``noise``) and every component's tape (``codegen``, written
by ``tsim.statevector.gpu.tape``) run as Metal kernels that PyObjC compiles at
runtime (``runtime``); ``MetalShotSampler`` (``sampler``) replaces
``ShotSampler`` behind the compiled samplers whenever PyObjC sees an Apple GPU
and every component fits. Amplitudes are complex64. It needs Apple silicon,
where Tsim depends on ``pyobjc-framework-Metal``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from tsim.statevector.gpu import FORKED
from tsim.statevector.gpu.tape import MAX_WIDTH, supports
from tsim.statevector.metal import runtime

if TYPE_CHECKING:
    import numpy as np

    from tsim.noise.channels import ChannelSampler
    from tsim.statevector.metal.sampler import MetalShotSampler
    from tsim.statevector.shots import ShotSampler

# NOTE: the warnings point at the caller of compile_sampler.
_STACKLEVEL = 5


def shot_sampler(
    host: ShotSampler,
    channels: ChannelSampler,
    num_detectors: int,
    seed: np.random.SeedSequence,
) -> ShotSampler | MetalShotSampler:
    """Return a Metal shot sampler of ``host``'s program when one can run it, else ``host``.

    It warns when an Apple GPU is there but cannot be used: in a process forked
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
            f"An Apple GPU is available, but a component needs {width} active "
            f"qubits, more than the {MAX_WIDTH} that the GPU executor supports; "
            "sampling on the CPU.",
            stacklevel=_STACKLEVEL,
        )
        return host
    from tsim.statevector.metal.sampler import MetalShotSampler

    try:
        return MetalShotSampler(host, channels, num_detectors, seed)
    except MemoryError:
        warnings.warn(
            "The GPU is out of memory; sampling on the CPU.", stacklevel=_STACKLEVEL
        )
    except ValueError as error:
        # NOTE: noise beyond the GPU tables' limits (channels of more than 127
        # outcomes, outcomes that flip more than 255 words).
        warnings.warn(
            f"The GPU cannot sample this circuit ({error}); sampling on the CPU.",
            stacklevel=_STACKLEVEL,
        )
    return host
