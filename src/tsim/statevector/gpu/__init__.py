"""What the statevector executor's GPU backends share.

A GPU backend runs the noise sampler and every component's tape as kernels of
its own language. The host side is common to them: the noise tables that the
noise kernels read (``noise``), the tape generator, which writes a component's
tape in any of their languages (``tape``), and the selection of outputs from
the rows that the kernels return (``rows``).
"""

FORKED = (
    "This process was forked after its parent used the GPU, so it samples on the "
    "CPU; start it with the 'spawn' or 'forkserver' method to sample on the GPU."
)
"""The warning of a GPU sampler in a process forked after its parent used the GPU."""
