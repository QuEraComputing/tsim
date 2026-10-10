import dataclasses
from fractions import Fraction

import pytest
from zx_param import Phase

from tsim.core.params import ParamLayout


def test_ranges_are_disjoint_and_consecutive():
    layout = ParamLayout(num_errors=3, num_basis=2, num_outputs=4)
    assert [layout.error(j) for j in range(3)] == [0, 1, 2]
    assert [layout.basis(j) for j in range(2)] == [3, 4]
    assert [layout.output(i) for i in range(4)] == [5, 6, 7, 8]


def test_basis_index_inverts_basis():
    layout = ParamLayout(num_errors=3, num_basis=2, num_outputs=4)
    assert layout.basis_index(layout.basis(1)) == 1
    with pytest.raises(ValueError, match="basis"):
        layout.basis_index(layout.error(0))
    with pytest.raises(ValueError, match="basis"):
        layout.basis_index(layout.output(0))


def test_names_cover_every_id():
    names = ParamLayout(num_errors=2, num_basis=1, num_outputs=2).names()
    assert dict(names) == {0: "e0", 1: "e1", 2: "f0", 3: "m0", 4: "m1"}
    assert 5 not in names and -1 not in names


def test_names_drive_zx_param_formatting():
    names = ParamLayout(num_errors=2, num_basis=1).names()
    assert Phase(Fraction(0), frozenset({1, 2})).format(names) == "(e1⊕f0)π"


def test_layout_is_extended_by_replace():
    layout = ParamLayout(num_errors=3)
    assert dataclasses.replace(layout, num_basis=2).basis(1) == 4
