"""The statevector executor on an Apple GPU, through Metal.

The noise sampler (``noise``) and every component's tape (``codegen``, written
by ``tsim.statevector.gpu.tape``) run as Metal kernels that PyObjC compiles at
runtime (``runtime``); ``MetalShotSampler`` (``sampler``) replaces
``ShotSampler`` behind the compiled samplers whenever PyObjC sees an Apple GPU
and every component fits. Amplitudes are complex64. It needs
``pyobjc-framework-Metal`` (``pip install bloqade-tsim[metal]``) and Apple
silicon.
"""
