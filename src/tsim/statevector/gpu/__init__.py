"""What the statevector executor's GPU backends share.

A GPU backend runs the noise sampler and every component's tape as kernels of
its own language. The host side is common to them: the noise tables that the
noise kernels read (``noise``), the tape generator, which writes a component's
tape in any of their languages (``tape``), and the selection of outputs from
the rows that the kernels return (``rows``).
"""
