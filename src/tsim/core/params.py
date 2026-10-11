"""Integer parameter IDs for error bits, error-basis bits and output bits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

ParamKind = Literal["e", "f", "m"]
"""Error bit, error-basis bit or output bit."""


@dataclass(frozen=True)
class ParamLayout:
    """Integer IDs for Tsim's parameters, fixed by the number of error bits.

    Error bit ``j`` has ID ``j``, so error IDs are error indices. Above them,
    error-basis and output bits are interleaved: basis bit ``j`` has ID
    ``num_errors + 2j`` and output bit ``i`` has ID ``num_errors + 2i + 1``. An
    ID therefore means the same thing at every pipeline stage, whatever the
    numbers of basis and output bits; names exist only for drawing.
    """

    num_errors: int

    def error(self, j: int) -> int:
        """Return the ID of error bit ``j``."""
        return j

    def basis(self, j: int) -> int:
        """Return the ID of error-basis bit ``j``."""
        return self.num_errors + 2 * j

    def output(self, i: int) -> int:
        """Return the ID of output bit ``i``."""
        return self.num_errors + 2 * i + 1

    def kind(self, param: int) -> tuple[ParamKind, int]:
        """Return the kind of the bit with ID ``param`` and its index."""
        if param < self.num_errors:
            return "e", param
        index, is_output = divmod(param - self.num_errors, 2)
        return ("m" if is_output else "f"), index

    def basis_index(self, param: int) -> int:
        """Return ``j`` for the error-basis bit whose ID is ``param``."""
        kind, j = self.kind(param)
        if kind != "f":
            raise ValueError(f"parameter {param} is not an error-basis bit")
        return j

    def name(self, param: int) -> str:
        """Return the display name ``e{j}``, ``f{j}`` or ``m{i}`` of ``param``."""
        kind, index = self.kind(param)
        return f"{kind}{index}"

    def names(self, params: Iterable[int]) -> dict[int, str]:
        """Return the display names of ``params``, by ID."""
        return {p: self.name(p) for p in params}
