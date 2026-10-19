"""Statevector sampling of Tsim's quantum components (``strategy="statevector"``).

A non-direct component of the reduced sampling graph is read, without circuit
extraction, as rotations and measurements in the frame of the initial
``|0…0⟩`` state (``heisenberg``), reordered to keep the dense array
small (``schedule``), planned on a fixed number of array axes (``plan``),
lowered to JAX arrays (``program``) and interpreted lane by lane (``execute``).
"""
