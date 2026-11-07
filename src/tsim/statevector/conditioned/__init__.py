"""Conditioned statevector sampling (``strategy="conditioned_statevector"``).

Tsim's ``active_statevector`` backend compiles one action tape per component
that serves every noise key and every outcome branch, so its width is set by
the worst case. This backend keeps the same pieces (a Clifford frame, a dense
register of *active* virtual qubits, every other virtual qubit in ``|0>``) but
plans each lane with its concrete values:

- rotation signs come from the lane's noise key;
- T-like layers are not applied before a measurement but *measured through*:
  the measurement of ``M`` behind the layer ``Q`` is the measurement of
  ``O = Q^dag M Q``, a Clifford involution whose projection is built directly
  in a register no wider than the result (``lane.measure_through``);
- after each measurement, Z-type stabilizers of the register are moved back
  into the frame (``lane.compress``).

Modules:
    paulis: Pauli rows (X/Z bit words and a power of i) and Clifford gates on them.
    gauss: Z4 Gauss sums, and one amplitude of ``O|0...0>``.
    lane: A lane's state and the operations on it.
    plan: The static step plan of a component (which rotations each
        measurement applies, which it measures through).
    component: Compilation of a component, its noise key and its outputs.
    runner: Runs a chunk of shots: lanes by key, split at random outcomes.
"""
