"""Statevector sampling of Tsim's quantum components (``strategy="active_statevector"``).

A non-direct component of the reduced sampling graph is read as rotations and
measurements in the frame of the initial ``|0…0⟩`` state (``pbc``),
reordered to keep the dense array small (``schedule``), planned on a fixed
number of array axes (``plan``), lowered to JAX arrays (``program``) and run
lane by lane (``execute``). As in Clifft [1], the dense array only spans the
active qubits.

References:
    [1] B. A. Chase, F. Labib, "Clifft: Fast Exact Simulation of Near-Clifford
        Quantum Circuits", arXiv:2604.27058 (2026).

"""
