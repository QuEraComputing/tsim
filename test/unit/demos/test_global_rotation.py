"""Smoke tests for the Fig. 4a global-rotation demo utilities."""

import sys
from pathlib import Path

import numpy as np

_DEMOS = Path(__file__).resolve().parents[3] / "docs" / "demos"
sys.path.insert(0, str(_DEMOS))

from utils.global_rotation import (  # noqa: E402
    CODE_7_PLUS_L,
    build_systems,
    compute_curve,
    verify_encoding_flows,
    verify_steane_plus_l_flows,
)

KEY_ANGLES = np.array([-180, -90, -45, 0, 45, 90, 180], dtype=float)


def test_steane_flow_counts():
    assert len(verify_encoding_flows(CODE_7_PLUS_L)) == 7
    assert len(verify_steane_plus_l_flows()) == 7


def test_fig4a_qualitative_curves():
    systems = {s.label: s for s in build_systems()}
    curves = {label: compute_curve(systems[label], KEY_ANGLES) for label in systems}

    xl_2d = curves["2D colour ($[[7,1,3]]$)"].logical_x
    xl_3d = curves["3D colour ($[[15,1,3]]$)"].logical_x
    xl_u = curves["Unentangled"].logical_x
    stab_2d = curves["2D colour ($[[7,1,3]]$)"].stabilizer_abs
    stab_3d = curves["3D colour ($[[15,1,3]]$)"].stabilizer_abs

    assert abs(xl_2d[3] - 1.0) < 0.02
    assert abs(xl_3d[3] - 1.0) < 0.02
    assert abs(xl_3d[4] - 0.707) < 0.05
    assert abs(xl_u[0] + 1.0) < 0.02
    assert abs(xl_u[3] - 1.0) < 0.02
    assert stab_2d[5] > 0.9
    assert stab_3d[5] > 0.9
    assert stab_3d[4] > 0.9
    assert xl_u[5] < 0.1
