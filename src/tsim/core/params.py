"""Integer parameter IDs for error bits, error-basis bits and output bits."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ParamLayout:
    """Separate ID ranges for Tsim's parameters.

    Error bit ``j`` has ID ``j``, error-basis bit ``j`` has ID ``num_errors + j``
    and output bit ``i`` has ID ``num_errors + num_basis + i``. An ID means the
    same thing at every pipeline stage; names exist only for drawing. Each count
    is fixed at its own stage, and the layout is extended with
    ``dataclasses.replace``.
    """

    num_errors: int
    num_basis: int = 0
    num_outputs: int = 0

    def error(self, j: int) -> int:
        """Return the ID of error bit ``j``."""
        return j

    def basis(self, j: int) -> int:
        """Return the ID of error-basis bit ``j``."""
        return self.num_errors + j

    def output(self, i: int) -> int:
        """Return the ID of output bit ``i``."""
        return self.num_errors + self.num_basis + i

    def basis_index(self, param: int) -> int:
        """Return ``j`` for the error-basis bit whose ID is ``param``."""
        j = param - self.num_errors
        if not 0 <= j < self.num_basis:
            raise ValueError(f"parameter {param} is not an error-basis bit")
        return j

    def names(self) -> Mapping[int, str]:
        """Return the display names ``e{j}``, ``f{j}`` and ``m{i}`` by ID."""
        return _Names(self)


class _Names(Mapping[int, str]):
    """Display names of a ``ParamLayout``, computed on lookup."""

    def __init__(self, layout: ParamLayout) -> None:
        self._layout = layout

    def __getitem__(self, param: int) -> str:
        layout = self._layout
        if 0 <= param < layout.num_errors:
            return f"e{param}"
        basis = param - layout.num_errors
        if 0 <= basis < layout.num_basis:
            return f"f{basis}"
        output = basis - layout.num_basis
        if 0 <= output < layout.num_outputs:
            return f"m{output}"
        raise KeyError(param)

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))

    def __len__(self) -> int:
        layout = self._layout
        return layout.num_errors + layout.num_basis + layout.num_outputs
