"""
Statevector simulatorfor stim circuits.

Based on code from:
Gidney, C., Jones, C., & Shutty, N. (2024). "Magic state cultivation: growing
T states as cheap as CNOT gates." arXiv:2409.17595


Code dataset: https://doi.org/10.5281/zenodo.13777072
Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

Modifications:
- Added missing cases in `do_stim_instruction`.
- Removed lattice surgery functionality.
"""

import random
from typing import (
    Dict,
    Any,
    Tuple,
    Union,
    Iterable,
    Literal,
    List,
    Optional,
    Sequence,
    Callable,
    cast,
)

import numpy as np
import stim


class VecSim:
    """A quantum state vector simulator.

    Qubits are added to the state using methods like `do_qalloc_x`.
    Qubits are operated on using methods like `do_h`.
    Qubits are removed using methods like `do_mz_discard`.
    """

    def __init__(self):
        # External qubit key to internal simulator index.
        self.q2i: Dict[Any, int] = {}
        # Internal simulator qubit index to external qubit key.
        self.i2q: Dict[int, Any] = {}
        # The state vector, stored as numpy tensor.
        self.state: np.ndarray = np.zeros(shape=(2, 2), dtype=np.complex64)
        self.state[0, 0] = 1
        # Workspace for implementing operations without allocating each time.
        self._buffer: np.ndarray = np.zeros(shape=(2, 2), dtype=np.complex64)
        # Recorded measurement results.
        self.m_record: Dict[Any, bool] = {}
        # Storage for instructions like `accumulator_bit_xor` and `accumulator_bit_save`.
        self._accumulator_bit = False
        # Used for giving arbitrary names to measurements.
        self.next_anon_key = 0

        # Randomness override configuration.
        self.grounded_qubits = set()

        # Error injection configuration.
        self._measurements_to_flip = set()
        self._next_error_mechanism = 0
        self.included_error_mechanisms = set()

    def clear(self):
        self.q2i = {}
        self.i2q = {}
        self.state = np.zeros(shape=(2, 2), dtype=np.complex64)
        self._buffer = np.zeros(shape=(2, 2), dtype=np.complex64)
        self.state[0, 0] = 1
        self.m_record = {}
        self.next_anon_key = 0

    def copy(self) -> "VecSim":
        s = VecSim()
        s.q2i = dict(self.q2i)
        s.i2q = dict(self.i2q)
        s.state = np.copy(self.state)
        s._buffer = np.copy(self._buffer)
        s.grounded_qubits = set(self.grounded_qubits)
        s._measurements_to_flip = set(self._measurements_to_flip)
        s._next_error_mechanism = self._next_error_mechanism
        s.included_error_mechanisms = set(self.included_error_mechanisms)
        s._accumulator_bit = self._accumulator_bit
        s.m_record = dict(self.m_record)
        s.next_anon_key = self.next_anon_key
        return s

    def normalized_state(
        self, *, order: Optional[Callable[[Any], Any]] = None
    ) -> np.ndarray:
        """Returns the internal state as a unit vector.

        Args:
            order: Determines which qubit gets mapped to which axis of the output numpy array.
                Qubit names with larger order keys, according to this function, are assigned to
                larger axis indices.
        """
        s = self.state[self.state_slicer({})]
        if order is not None:
            qs = list(self.q2i.keys())
            q2s = sorted(qs, key=order)
            i2s = [qs.index(q) for q in q2s]
            actual_order = sorted(self.q2i.keys(), key=lambda k: self.q2i[k])
            desired_order = sorted(self.q2i.keys(), key=order)
            i2s = [actual_order.index(q) for q in desired_order]
            s = np.transpose(s, i2s)
        return s / np.linalg.norm(s)

    def state_str(self, *, order: Optional[Callable[[Any], Any]] = None) -> Any:
        if order is None and all(
            isinstance(q, (complex, float, int)) for q in self.q2i
        ):
            order = lambda c: (c.real, c.imag)

        qs = sorted(self.q2i.keys(), key=order)[::-1]
        s = self.normalized_state(order=qs.index).flatten()
        if abs(s[0]) > 1e-6:
            s /= s[0]
        magnitudes = list(np.abs(s).round(4))
        polars = list((np.angle(s) * 180 / np.pi).round(4))
        result = [f"state {qs} {{"]
        n = len(magnitudes).bit_length() - 1
        result.append(f'    {"".rjust(n, " ")}  {"mag".rjust(10)} {"angle".rjust(10)}')
        for k, (m, p) in enumerate(zip(magnitudes, polars)):
            if m == int(m):
                m = int(m)
            if p == int(p):
                p = int(p)
                if p == -180:
                    p = 180
            m = str(m).rjust(10)
            p = str(p).rjust(10)
            result.append(f'    {bin(k)[2:].rjust(n, "0")}: {m} {p}')
        result.append("}")
        return "\n".join(result)

    def state_slicer(self, qs: Dict[Any, bool]) -> Tuple[Union[int, slice], ...]:
        """Returns a value that can be used to slice into a subset of the state.

        Args:
            qs: The subset to slice into is identified by specifying values for some qubits.
                For example, the part of the state vector where qubit 'A' is ON.
        """
        mask: List[Union[slice, int]] = [slice(None)] * len(self.state.shape)
        for k in range(len(self.q2i), len(mask)):
            mask[k] = 0
        for a, b in qs.items():
            i = self.q2i[a]
            assert mask[i] == slice(None)
            mask[i] = int(b)
        return tuple(mask)

    def do_qalloc_z(self, q: Any) -> None:
        """Allocates a new qubit, initializing it into the |0> state."""
        assert q not in self.q2i, f"{q} already allocated"
        i = len(self.q2i)
        assert len(self.q2i) < 20
        self.q2i[q] = i
        self.i2q[i] = q
        if len(self.q2i) > len(self.state.shape):
            old_state = self.state
            self.state = np.zeros(shape=(2,) * len(self.q2i), dtype=np.complex64)
            self._buffer = np.zeros(shape=(2,) * len(self.q2i), dtype=np.complex64)
            m: List[Union[slice, int]] = [slice(None)] * len(old_state.shape)
            m += [0] * (len(self.q2i) - len(old_state.shape))
            self.state[tuple(m)] = old_state
        self.do_rz(q)

    def do_qalloc_y(self, q: Any) -> None:
        """Allocates a new qubit, initializing it into the |i> state."""
        self.do_qalloc_z(q)
        self.do_h_yz(q)

    def do_qalloc_x(self, q: Any) -> None:
        """Allocates a new qubit, initializing it into the |+> state."""
        self.do_qalloc_z(q)
        self.do_h(q)

    def do_qalloc_p(self, q: Any, p: Literal["X", "Y", "Z"]) -> None:
        if p == "X":
            self.do_qalloc_x(q)
        elif p == "Y":
            self.do_qalloc_y(q)
        elif p == "Z":
            self.do_qalloc_z(q)
        else:
            raise NotImplementedError(f"{p=}")

    def _do_obs_qubits_to_z(self, obs: Dict[complex, Literal["X", "Y", "Z"]]):
        """Rotates given qubits so that each of their given axes gets swapped with their Z axis."""
        for q, b in obs.items():
            if b == "X":
                self.do_h(q)
            elif b == "Y":
                self.do_h_yz(q)
            elif b == "Z":
                pass
            else:
                raise NotImplementedError(f"{obs=}")

    def peek_obs(
        self, obs: Dict[Any, Literal["X", "Y", "Z"]], *, sign: int = +1
    ) -> float:
        s = self.copy()
        s._do_obs_qubits_to_z(obs)
        root, *rest = obs.keys()
        for q in rest:
            s.do_cx(q, root)
        r = s.peek_z(root)
        if sign == -1:
            r = -r
        elif sign == +1:
            pass
        else:
            raise NotImplementedError(f"{sign=}")
        return r

    def do_measure_obs(
        self,
        obs: Dict[Any, Literal["X", "Y", "Z"]],
        *,
        sign: int = +1,
        key: Optional[Any] = None,
        prefer_result: bool | None = None,
    ) -> bool:
        self._do_obs_qubits_to_z(obs)
        root, *rest = obs.keys()
        for q in rest:
            self.do_cx(q, root)
        r = self.do_mz(root, key=key, prefer_result=prefer_result)
        if sign == -1:
            r = not r
        elif sign == +1:
            pass
        else:
            raise NotImplementedError(f"{sign=}")
        for q in rest:
            self.do_cx(q, root)
        self._do_obs_qubits_to_z(obs)
        return r

    def do_t_obs(
        self, obs: Dict[Any, Literal["X", "Y", "Z"]], *, sign: int = +1
    ) -> None:
        """Applies a T gate to the given Pauli product observable."""
        self._do_obs_qubits_to_z(obs)
        root, *rest = obs.keys()
        for q in rest:
            self.do_cx(q, root)
        if sign == -1:
            self.do_x(root)
        self.do_t(root)
        if sign == -1:
            self.do_x(root)
        for q in rest:
            self.do_cx(q, root)
        self._do_obs_qubits_to_z(obs)

    def do_s_obs(self, obs: dict[Any, Literal["X", "Y", "Z"]], sign: int = +1) -> None:
        self._do_obs_qubits_to_z(obs)
        root, *rest = obs.keys()
        for q in rest:
            self.do_cx(q, root)
        if sign == -1:
            self.do_x(root)
        self.do_s(root)
        if sign == -1:
            self.do_x(root)
        for q in rest:
            self.do_cx(q, root)
        self._do_obs_qubits_to_z(obs)

    def do_z(self, a: Any) -> None:
        self.state[self.state_slicer({a: True})] *= -1

    def do_y(self, a: Any) -> None:
        self.do_x(a)
        self.do_z(a)

    def do_x(self, q: Any) -> None:
        f = self.state_slicer({q: False})
        t = self.state_slicer({q: True})
        self._buffer[f] = self.state[f]
        self.state[f] = self.state[t]
        self.state[t] = self._buffer[f]

    def do_h(self, q: Any) -> None:
        f = self.state_slicer({q: False})
        t = self.state_slicer({q: True})
        self._buffer[f] = self.state[f]
        self.state[f] += self.state[t]
        self.state[t] *= -1
        self.state[t] += self._buffer[f]

    def do_h_yz(self, q: Any) -> None:
        self.do_s_dag(q)
        self.do_h(q)
        self.do_s(q)

    def do_h_xy(self, q: Any) -> None:
        self.do_t_dag(q)
        self.do_x(q)
        self.do_t(q)

    def do_t(self, q: Any) -> None:
        self.state[self.state_slicer({q: True})] *= (1 + 1j) / np.sqrt(2)

    def do_t_dag(self, q: Any) -> None:
        self.state[self.state_slicer({q: True})] *= (1 - 1j) / np.sqrt(2)

    def do_s(self, a: Any) -> None:
        self.state[self.state_slicer({a: True})] *= 1j

    def do_multi_phase(self, qubits: Iterable[Any], phase: complex) -> None:
        root, *rest = qubits
        for q in rest:
            self.do_cx(q, root)
        self.state[self.state_slicer({root: True})] *= phase
        for q in rest:
            self.do_cx(q, root)

    def do_s_dag(self, a: Any) -> None:
        self.state[self.state_slicer({a: True})] *= -1j

    def do_cx(self, a: Any, b: Any) -> None:
        tf = self.state_slicer({a: True, b: False})
        tt = self.state_slicer({a: True, b: True})
        self._buffer[tf] = self.state[tf]
        self.state[tf] = self.state[tt]
        self.state[tt] = self._buffer[tf]

    def do_cy(self, a: Any, b: Any) -> None:
        self.do_s_dag(b)
        self.do_cx(a, b)
        self.do_s(b)

    def do_xcy(self, a: Any, b: Any) -> None:
        self.do_h_yz(b)
        self.do_cx(b, a)
        self.do_h_yz(b)

    def do_cz(self, a: Any, b: Any) -> None:
        tt = self.state_slicer({a: True, b: True})
        self.state[tt] *= -1

    def do_cs(self, a: Any, b: Any) -> None:
        tt = self.state_slicer({a: True, b: True})
        self.state[tt] *= 1j

    def do_ccz(self, a: Any, b: Any, c: Any) -> None:
        ttt = self.state_slicer({a: True, b: True, c: True})
        self.state[ttt] *= -1

    def do_swap(self, a: Any, b: Any) -> None:
        if (a in self.q2i) != (b in self.q2i):
            if a in self.q2i:
                i = self.q2i.pop(a)
                self.q2i[b] = i
                self.i2q[i] = b
            else:
                i = self.q2i.pop(b)
                self.q2i[a] = i
                self.i2q[i] = a
            return

        tf = self.state_slicer({a: True, b: False})
        ft = self.state_slicer({a: False, b: True})
        self._buffer[tf] = self.state[tf]
        self.state[tf] = self.state[ft]
        self.state[ft] = self._buffer[tf]

    def do_mxx(
        self,
        a: Any,
        b: Any,
        *,
        key: Optional[Any] = None,
        prefer_result: bool | None = None,
    ) -> bool:
        if a in self.grounded_qubits or b in self.grounded_qubits:
            prefer_result = False
        self.do_cx(a, b)
        r = self.do_mx(a, key=key, prefer_result=prefer_result)
        self.do_cx(a, b)
        return r

    def do_myy(
        self,
        a: Any,
        b: Any,
        *,
        key: Optional[Any] = None,
        prefer_result: bool | None = None,
    ) -> bool:
        if a in self.grounded_qubits or b in self.grounded_qubits:
            prefer_result = False
        self.do_cy(a, b)
        r = self.do_my(a, key=key, prefer_result=prefer_result)
        self.do_cy(a, b)
        return r

    def do_mzz(
        self,
        a: Any,
        b: Any,
        *,
        key: Optional[Any] = None,
        prefer_result: bool | None = None,
    ) -> bool:
        if a in self.grounded_qubits or b in self.grounded_qubits:
            prefer_result = False
        self.do_cx(a, b)
        r = self.do_mz(b, key=key, prefer_result=prefer_result)
        self.do_cx(a, b)
        return r

    def peek_x(self, q: Any) -> float:
        s = self.copy()
        s.do_h(q)
        return s.peek_z(q)

    def peek_y(self, q: Any) -> float:
        s = self.copy()
        s.do_h_yz(q)
        return s.peek_z(q)

    def peek_z(self, q: Any) -> float:
        f = self.state_slicer({q: False})
        t = self.state_slicer({q: True})
        weight_f = np.linalg.norm(self.state[f]) ** 2
        weight_t = np.linalg.norm(self.state[t]) ** 2
        return 1 - 2 * weight_t / (weight_t + weight_f)

    def peek_p(self, q: Any, p: Literal["X", "Y", "Z"]) -> float:
        if p == "X":
            return self.peek_x(q)
        elif p == "Y":
            return self.peek_y(q)
        elif p == "Z":
            return self.peek_z(q)
        else:
            raise NotImplementedError(f"{p=}")

    def do_mz(
        self, q: Any, *, key: Optional[Any] = None, prefer_result: bool | None = None
    ) -> bool:
        if q in self.grounded_qubits:
            prefer_result = False
        f = self.state_slicer({q: False})
        t = self.state_slicer({q: True})
        weight_f = np.linalg.norm(self.state[f]) ** 2
        weight_t = np.linalg.norm(self.state[t]) ** 2
        p = weight_t / (weight_t + weight_f)
        result = random.random() < p
        if prefer_result is not None and 0.001 < p < 0.999:
            result = prefer_result
        if result:
            self.state[f] = 0
            w = weight_t
        else:
            self.state[t] = 0
            w = weight_f
        if not (0.001 < w < 1000):
            self.state /= np.sqrt(w)
        return self._record_measurement(key, result)

    def do_mrz(
        self, q: Any, *, key: Optional[Any] = None, prefer_result: bool | None = None
    ) -> bool:
        r = self.do_mz(q, key=key, prefer_result=prefer_result)
        if r:
            self.do_x(q)
        return r

    def do_rz(self, q: Any) -> None:
        self.do_mrz(q, key=None)

    def do_mx(
        self, q: Any, *, key: Optional[Any] = None, prefer_result: bool | None = None
    ) -> bool:
        self.do_h(q)
        r = self.do_mz(q, key=key, prefer_result=prefer_result)
        self.do_h(q)
        return r

    def do_paulis(self, paulis: dict[Any, Literal["X", "Y", "Z"]]) -> None:
        for t, p in paulis.items():
            if p == "_" or p == "I":
                pass
            elif p == "X":
                self.do_x(t)
            elif p == "Y":
                self.do_y(t)
            elif p == "Z":
                self.do_z(t)
            else:
                raise NotImplementedError(f"{p=} {t=}")

    def do_pauli_string(self, *, paulis: str, targets: Sequence[Any]) -> None:
        assert len(paulis) == len(targets)
        for p, t in zip(paulis, targets):
            if p == "_" or p == "I":
                pass
            elif p == "X":
                self.do_x(t)
            elif p == "Y":
                self.do_y(t)
            elif p == "Z":
                self.do_z(t)
            else:
                raise NotImplementedError(f"{p=} {t=}")

    def do_pauli_dot(
        self,
        *,
        paulis: Sequence[str],
        controls: Sequence[bool],
        targets: Sequence[Any],
    ):
        assert len(paulis) == len(controls)
        for p, c in zip(paulis, controls, strict=True):
            if c:
                self.do_pauli_string(paulis=p, targets=targets)

    def do_ry(self, q: Any) -> None:
        self.do_rz(q)
        self.do_h_yz(q)

    def do_mry(
        self, q: Any, *, key: Optional[Any] = None, prefer_result: bool | None = None
    ) -> bool:
        self.do_h_yz(q)
        r = self.do_mrz(q, key=key, prefer_result=prefer_result)
        self.do_h_yz(q)
        return r

    def do_mx_discard(
        self, q: Any, *, key: Optional[Any] = None, prefer_result: bool | None = None
    ) -> bool:
        self.do_h(q)
        return self.do_mz_discard(q, key=key, prefer_result=prefer_result)

    def do_my_discard(
        self, q: Any, *, key: Optional[Any] = None, prefer_result: bool | None = None
    ) -> bool:
        self.do_h_yz(q)
        return self.do_mz_discard(q, key=key, prefer_result=prefer_result)

    def do_my(
        self, q: Any, *, key: Optional[Any] = None, prefer_result: bool | None = None
    ) -> bool:
        self.do_h_yz(q)
        r = self.do_mz(q, key=key, prefer_result=prefer_result)
        self.do_h_yz(q)
        return r

    def do_mrx(
        self, q: Any, *, key: Optional[Any] = None, prefer_result: bool | None = None
    ) -> bool:
        self.do_h(q)
        r = self.do_mrz(q, key=key, prefer_result=prefer_result)
        self.do_h(q)
        return r

    def do_rx(self, q: Any) -> None:
        self.do_rz(q)
        self.do_h(q)

    def do_mp_discard(
        self,
        q: Any,
        p: Literal["X", "Y", "Z"],
        *,
        prefer_result: bool | None = None,
        key: Any = None,
    ) -> bool:
        if p == "X":
            return self.do_mx_discard(q, key=key, prefer_result=prefer_result)
        elif p == "Y":
            return self.do_my_discard(q, key=key, prefer_result=prefer_result)
        elif p == "Z":
            return self.do_mz_discard(q, key=key, prefer_result=prefer_result)
        else:
            raise NotImplementedError(f"{p=}")

    def do_mz_discard(
        self, q: Any, *, key: Optional[Any] = None, prefer_result: bool | None = None
    ) -> bool:
        r = self.do_mrz(q, key=key, prefer_result=prefer_result)

        i = self.q2i[q]
        n = len(self.q2i) - 1
        if i < n:
            other = self.i2q[n]
            self.do_swap(q, other)
            self.q2i[other] = i
            self.i2q[i] = other
        del self.i2q[n]
        del self.q2i[q]
        return r

    def _record_measurement(self, key: Any, b: bool) -> bool:
        if key is not None:
            if key in self._measurements_to_flip:
                b = not b
            self.m_record[key] = b
        return b

    def do_stim_instruction(
        self,
        inst: stim.CircuitInstruction,
        *,
        sweep_bits: dict[int, bool],
        out_measurements: list[bool],
        out_detectors: list[bool],
        out_observables: list[bool],
    ):
        if inst.name == "QUBIT_COORDS":
            pass
        elif inst.name == "SHIFT_COORDS":
            pass
        elif inst.name == "TICK":
            pass
        elif inst.name == "DETECTOR":
            b = False
            for q in inst.targets_copy():
                assert q.is_measurement_record_target
                assert -len(out_measurements) <= q.value < 0
                b ^= out_measurements[q.value]
            out_detectors.append(b)
        elif inst.name == "OBSERVABLE_INCLUDE":
            (index,) = inst.gate_args_copy()
            index = round(index)
            while index >= len(out_observables):
                out_observables.append(False)
            for q in inst.targets_copy():
                assert q.is_measurement_record_target
                assert -len(out_measurements) <= q.value < 0
                out_observables[index] ^= out_measurements[q.value]
        elif inst.name == "MPP":
            ps = inst.gate_args_copy()
            if ps:
                (p,) = ps
            else:
                p = 0

            for terms in inst.target_groups():
                obs: dict[Any, Literal["X", "Y", "Z"]] = {}
                flipped = False
                for t in terms:
                    flipped ^= t.is_inverted_result_target
                    obs[t.qubit_value] = cast(Literal["X", "Y", "Z"], t.pauli_type)
                out_measurements.append(self.do_measure_obs(obs) ^ flipped)
                if random.random() < p:
                    out_measurements[-1] ^= True

        elif inst.name == "RX":
            for q in inst.targets_copy():
                self.do_rx(q.qubit_value)
        elif inst.name == "R":
            for q in inst.targets_copy():
                self.do_rz(q.qubit_value)
        elif inst.name == "X":
            for q in inst.targets_copy():
                self.do_x(q.qubit_value)
        elif inst.name == "Y":
            for q in inst.targets_copy():
                self.do_y(q.qubit_value)
        elif inst.name == "Z":
            for q in inst.targets_copy():
                self.do_z(q.qubit_value)
        elif inst.name == "S":
            for q in inst.targets_copy():
                self.do_s(q.qubit_value)
        elif inst.name == "S_DAG":
            for q in inst.targets_copy():
                self.do_s_dag(q.qubit_value)
        elif inst.name == "H":
            for q in inst.targets_copy():
                self.do_h(q.qubit_value)
        elif inst.name == "M":
            ps = inst.gate_args_copy()
            if ps:
                (p,) = ps
            else:
                p = 0
            for q in inst.targets_copy():
                out_measurements.append(
                    self.do_mz(q.qubit_value) ^ q.is_inverted_result_target
                )
                if random.random() < p:
                    out_measurements[-1] ^= True
        elif inst.name == "MX":
            ps = inst.gate_args_copy()
            if ps:
                (p,) = ps
            else:
                p = 0
            for q in inst.targets_copy():
                out_measurements.append(
                    self.do_mx(q.qubit_value) ^ q.is_inverted_result_target
                )
                if random.random() < p:
                    out_measurements[-1] ^= True
        elif inst.name == "X_ERROR":
            (p,) = inst.gate_args_copy()
            for q in inst.targets_copy():
                if random.random() < p:
                    self.do_x(q.qubit_value)
        elif inst.name == "Z_ERROR":
            (p,) = inst.gate_args_copy()
            for q in inst.targets_copy():
                if random.random() < p:
                    self.do_z(q.qubit_value)
        elif inst.name == "DEPOLARIZE1":
            (p,) = inst.gate_args_copy()
            for q in inst.targets_copy():
                if random.random() < p:
                    v = random.randrange(3)
                    if v == 0:
                        self.do_x(q.qubit_value)
                    elif v == 1:
                        self.do_y(q.qubit_value)
                    else:
                        self.do_z(q.qubit_value)
        elif inst.name == "DEPOLARIZE2":
            (p,) = inst.gate_args_copy()
            ts = inst.targets_copy()
            for k in range(0, len(ts), 2):
                q1 = ts[k].qubit_value
                q2 = ts[k + 1].qubit_value
                if random.random() < p:
                    v = random.randrange(1, 16)
                    v1 = v & 3
                    v2 = v >> 2
                    if v1 == 1:
                        self.do_x(q1)
                    elif v1 == 2:
                        self.do_y(q1)
                    elif v1 == 3:
                        self.do_z(q1)
                    if v2 == 1:
                        self.do_x(q2)
                    elif v2 == 2:
                        self.do_y(q2)
                    elif v2 == 3:
                        self.do_z(q2)
        elif inst.name == "CX":
            ts = inst.targets_copy()
            for k in range(0, len(ts), 2):
                t1, t2 = ts[k], ts[k + 1]
                if t1.is_measurement_record_target:
                    if out_measurements[t1.value]:
                        self.do_x(t2.qubit_value)
                elif t1.is_sweep_bit_target:
                    if sweep_bits[t1.value]:
                        self.do_z(t2.qubit_value)
                else:
                    self.do_cx(t1.qubit_value, t2.qubit_value)
        elif inst.name == "CZ":
            ts = inst.targets_copy()
            for k in range(0, len(ts), 2):
                t1, t2 = ts[k], ts[k + 1]
                if t1.is_measurement_record_target:
                    if out_measurements[t1.value]:
                        self.do_z(t2.qubit_value)
                elif t2.is_measurement_record_target:
                    if out_measurements[t2.value]:
                        self.do_z(t1.qubit_value)
                elif t1.is_sweep_bit_target:
                    if sweep_bits[t1.value]:
                        self.do_z(t2.qubit_value)
                elif t2.is_sweep_bit_target:
                    if sweep_bits[t2.value]:
                        self.do_z(t1.qubit_value)
                else:
                    self.do_cz(t1.qubit_value, t2.qubit_value)
        else:
            raise NotImplementedError(f"{inst=}")
