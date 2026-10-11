from fractions import Fraction

import pytest
from zxparam import Phase

from tsim.core.params import ParamLayout


def test_errors_keep_their_index_and_basis_and_outputs_interleave():
    layout = ParamLayout(num_errors=3)
    assert [layout.error(j) for j in range(3)] == [0, 1, 2]
    assert [layout.basis(j) for j in range(2)] == [3, 5]
    assert [layout.output(i) for i in range(3)] == [4, 6, 8]


def test_kind_inverts_every_id():
    layout = ParamLayout(num_errors=3)
    assert layout.kind(layout.error(2)) == ("e", 2)
    assert layout.kind(layout.basis(4)) == ("f", 4)
    assert layout.kind(layout.output(5)) == ("m", 5)


def test_basis_index_inverts_basis():
    layout = ParamLayout(num_errors=3)
    assert layout.basis_index(layout.basis(7)) == 7
    with pytest.raises(ValueError, match="basis"):
        layout.basis_index(layout.error(0))
    with pytest.raises(ValueError, match="basis"):
        layout.basis_index(layout.output(0))


def test_names_name_the_given_ids():
    names = ParamLayout(num_errors=2).names([0, 1, 2, 3, 5])
    assert names == {0: "e0", 1: "e1", 2: "f0", 3: "m0", 5: "m1"}


def test_names_drive_zxparam_formatting():
    layout = ParamLayout(num_errors=2)
    phase = Phase(Fraction(0), frozenset({1, 2}))
    assert phase.format(layout.names(phase.parameters)) == "(e1⊕f0)π"
