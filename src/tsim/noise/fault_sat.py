"""MaxSAT search for the smallest sets of faults that cause a logical error.

Stim's ``shortest_error_sat_problem`` writes a Clifford circuit's distance as a
MaxSAT problem: a variable per error, XOR constraints that the detectors stay
quiet and an observable flips, and a soft clause per error. Here everything but
the statevector components is such a constraint as well: the direct detectors
cancel, and the rotation signs ``s`` and output flips ``o`` of a set are XORs
of its faults. Whether ``(s, o)`` can flip an observable undetected is only
known by walking the tape, so it is added lazily:

1. Solve for the fewest faults whose direct detectors cancel and that flip a
   rotation sign or an observable.
2. Walk the tape for the solution's signs ``s*``. If the solution is a logical
   error, it has the fewest faults: every cut so far only removed sets that
   cannot be one.
3. Otherwise add the cut "``s = s*`` implies that ``o`` is one of the output
   flips that the distribution of ``s*`` lets through", and solve again.

Each distinct ``s*`` costs one round, so the rounds stay few when few distinct
signs occur below the distance.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from pysat.card import ITotalizer
from pysat.solvers import Solver

from tsim.noise.faults import (
    MIN_PROBABILITY,
    LogicalError,
    _distinct,
    _Search,
)

if TYPE_CHECKING:
    from tsim.circuit import Circuit


def _bits(words: np.ndarray, width: int) -> np.ndarray:
    """Return rows of uint64 words as rows of ``width`` bits."""
    return np.unpackbits(words.view(np.uint8), axis=1, bitorder="little")[:, :width]


class SatSearch:
    """The MaxSAT form of a circuit's fault search."""

    def __init__(self, circuit: Circuit, *, log: Callable[[str], None] | None = None):
        """Build the constraints of ``circuit``; ``log`` receives progress lines."""
        start = time.monotonic()
        self._search = search = _Search(circuit)
        self._log = log or (lambda _: None)
        n = len(search._events)
        num_o = int(sum(len(s) for s in search._slots) + len(search._direct))
        num_s = sum(len(b.sign_rows) for b in search._branches)
        self._d = np.unpackbits(search._d.view(np.uint8), axis=1, bitorder="little")
        self._sbits = _bits(search._s, num_s)
        self._obits = _bits(search._o, num_o)
        det = _bits(search._det_mask[None, :], num_o)[0].astype(bool)
        self._det_slots = np.flatnonzero(det)
        self._obs_slots = np.flatnonzero(~det)

        self._clauses: list[list[int]] = []
        self._xors: list[list[int]] = []
        self._next = n + 1
        # NOTE: an XOR constraint ``vs`` says that the XOR of its variables is 0.
        for column in np.flatnonzero(self._d.any(axis=0)):
            self._xors.append([int(i) + 1 for i in np.flatnonzero(self._d[:, column])])
        self._s_var = [self._xor(np.flatnonzero(col)) for col in self._sbits.T]
        self._o_var = [self._xor(np.flatnonzero(col)) for col in self._obits.T]
        nontrivial = [v for v in self._s_var if v]
        nontrivial += [self._o_var[k] for k in self._obs_slots if self._o_var[k]]
        self._clauses.append(nontrivial)
        self._cuts: dict[bytes, int] = {}
        self.rounds = 0
        self.walks = 0
        self.solver_seconds = 0.0
        self._log(
            f"{n} faults, {self._next - 1} variables, {len(self._xors)} XOR "
            f"constraints, built in {time.monotonic() - start:.1f} s"
        )

    def _var(self) -> int:
        self._next += 1
        return self._next - 1

    def _xor(self, faults: np.ndarray) -> int:
        """Return a variable equal to the XOR of the faults' variables, or 0."""
        if len(faults) == 0:
            return 0
        out = self._var()
        self._xors.append([out, *(int(i) + 1 for i in faults)])
        return out

    def _chain(self, xor: list[int]) -> list[list[int]]:
        """Return CNF clauses for the XOR constraint ``xor``, with new variables."""
        clauses = []
        acc = xor[0]
        for x in xor[1:]:
            out = self._var()
            # NOTE: out = acc XOR x, as four clauses.
            clauses += [
                [-out, acc, x],
                [-out, -acc, -x],
                [out, -acc, x],
                [out, acc, -x],
            ]
            acc = out
        clauses.append([-acc])
        return clauses

    def shortest(
        self, max_weight: int, *, every: bool, solver: str = "cryptominisat"
    ) -> list[LogicalError]:
        """Return the logical errors with the fewest faults, up to ``max_weight``.

        With ``every``, return all of them; otherwise the first one found. For
        ``k = 1, 2, …`` the solver looks for sets of at most ``k`` faults; once
        none is left, every logical error has more than ``k``.
        """
        search = self._search
        n = len(search._events)
        found: list[LogicalError] = []
        counter = ITotalizer(
            lits=list(range(1, n + 1)), ubound=max_weight, top_id=self._next
        )
        self._next = counter.top_id + 1
        with _Solver(solver) as sat:
            self._sat = sat
            if solver == "cryptominisat":
                for xor in self._xors:
                    sat.add_xor(xor)
            else:
                for xor in self._xors:
                    sat.add_clauses(self._chain(xor))
            sat.add_clauses(self._clauses + counter.cnf.clauses)
            for k in range(1, max_weight + 1):
                # NOTE: rhs[k] is true when more than k faults are chosen.
                at_most = [-counter.rhs[k]] if k < len(counter.rhs) else []
                while True:
                    self.rounds += 1
                    start = time.monotonic()
                    satisfiable = sat.solve(assumptions=at_most)
                    self.solver_seconds += time.monotonic() - start
                    if not satisfiable:
                        break
                    chosen = sat.true_among(n)
                    error = self._check(chosen)
                    if error is None:
                        continue
                    found.append(error)
                    self._log(
                        f"round {self.rounds}: logical error of {len(chosen)} faults"
                    )
                    if not every:
                        return found
                    sat.add_clause([-(i + 1) for i in chosen])
                self._log(f"no more sets of {k} faults after {self.rounds} rounds")
                if found:
                    return found
        return found

    def _check(self, chosen: list[int]) -> LogicalError | None:
        """Return the logical error of ``chosen``, or cut it away and return None."""
        search = self._search
        events = search._events
        if not _distinct([{e[0] for e in events[j]} for j in chosen]):
            # NOTE: two circuit errors of one channel exclude each other, and so
            # do they in every larger set.
            self._sat.add_clause([-(i + 1) for i in chosen])
            return None
        row_s = np.bitwise_xor.reduce(search._s[chosen], axis=0)
        key = row_s.tobytes()
        signs = row_s[None, :]
        reps = np.bitwise_xor.reduce(search._bits[chosen], axis=0)[None, :]
        distributions = [b.outcomes(reps[:, b.f_selection]) for b in search._branches]
        self.walks += 1
        undetected, flip = search._probabilities(
            signs, distributions, np.array([chosen])
        )
        if flip[0] > MIN_PROBABILITY:
            return LogicalError(
                tuple(search._fault(j) for j in chosen),
                float(undetected[0]),
                float(flip[0]),
            )
        if key in self._cuts:
            # NOTE: the cut let this set through, but its chance is below
            # MIN_PROBABILITY.
            self._sat.add_clause([-(i + 1) for i in chosen])
            return None
        self._cuts[key] = self.rounds
        self._sat.add_clause(self._cut(row_s, search._table(signs, distributions)))
        return None

    def _cut(self, row_s: np.ndarray, table: tuple) -> list[int]:
        """Return the clause "signs are not ``row_s``, or a row of ``table`` holds".

        Adds a variable per row of the table, which implies that the output
        flips have its detector deviations and can flip an observable.
        """
        num_s = self._sbits.shape[1]
        s = _bits(row_s[None, :], num_s)[0]
        clause = []
        for k, v in enumerate(self._s_var):
            if v:
                clause.append(-v if s[k] else v)
            elif s[k]:
                # NOTE: no fault flips this sign, so no set has these signs.
                return []
        _, masked, _, _, multi, obs, _ = table
        num_o = self._obits.shape[1]
        for row, is_multi, value in zip(
            _bits(masked, num_o), multi, _bits(obs, num_o), strict=True
        ):
            z = self._var()
            for k in self._det_slots.tolist():
                v = self._o_var[k]
                want = bool(row[k])
                if v == 0:
                    if want:
                        break
                    continue
                self._sat.add_clause([-z, v if want else -v])
            else:
                if not is_multi:
                    # NOTE: the observables must not all keep the row's value.
                    differ = []
                    for k in self._obs_slots.tolist():
                        v = self._o_var[k]
                        if v:
                            differ.append(-v if value[k] else v)
                        elif value[k]:
                            differ = None
                            break
                    if differ is not None:
                        self._sat.add_clause([-z, *differ])
                clause.append(z)
                continue
            self._sat.add_clause([-z])
        return clause


def shortest_errors_sat(
    circuit: Circuit,
    *,
    max_weight: int,
    every: bool = True,
    log: Callable[[str], None] | None = None,
) -> list[LogicalError]:
    """Return the smallest logical errors, as ``shortest_errors``, with MaxSAT."""
    return SatSearch(circuit, log=log).shortest(max_weight, every=every)


class _Solver:
    """An incremental SAT solver: pysat's (CNF only) or CryptoMiniSat's."""

    def __init__(self, name: str) -> None:
        self._name = name
        if name == "cryptominisat":
            import pycryptosat

            self._cms = pycryptosat.Solver()
            self._model: tuple = ()
        else:
            self._pysat = Solver(name=name)

    def __enter__(self) -> _Solver:
        return self

    def __exit__(self, *_: object) -> None:
        if self._name != "cryptominisat":
            self._pysat.delete()

    def add_clause(self, clause: list[int]) -> None:
        if self._name == "cryptominisat":
            self._cms.add_clause(clause)
        else:
            self._pysat.add_clause(clause)

    def add_clauses(self, clauses: list[list[int]]) -> None:
        if self._name == "cryptominisat":
            self._cms.add_clauses(clauses)
        else:
            self._pysat.append_formula(clauses)

    def add_xor(self, xor: list[int]) -> None:
        self._cms.add_xor_clause(xor, False)

    def solve(self, assumptions: list[int]) -> bool:
        if self._name == "cryptominisat":
            satisfiable, self._model = self._cms.solve(assumptions)
            return bool(satisfiable)
        return bool(self._pysat.solve(assumptions=assumptions))

    def true_among(self, n: int) -> list[int]:
        """Return the indices ``i < n`` whose variable ``i + 1`` is true."""
        if self._name == "cryptominisat":
            return [i for i in range(n) if self._model[i + 1]]
        model = self._pysat.get_model()
        assert model is not None
        return [i for i in range(n) if model[i] > 0]
