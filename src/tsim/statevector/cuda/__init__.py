"""The statevector executor on a CUDA GPU (``device="cuda"``).

The noise sampler (``noise``) and every component's tape (``codegen``) run as
CUDA kernels that CuPy compiles with NVRTC (``runtime``); ``CudaShotSampler``
(``sampler``) replaces ``ShotSampler`` behind the compiled samplers. Amplitudes
are complex64. It needs CuPy and an NVIDIA GPU (``pip install
bloqade-tsim[gpu]``).
"""

from tsim.statevector.cuda.runtime import available

__all__ = ["available"]
