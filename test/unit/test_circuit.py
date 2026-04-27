from typing import Any, Literal
from unittest.mock import patch

import numpy as np
import pytest
import stim

from tsim.circuit import Circuit


def unitaries_equal_up_to_global_phase(
    u1: np.ndarray, u2: np.ndarray[Any, Any]
) -> bool:
    product = u1 @ u2.conj().T
    # If u1 = e^(i*phi) * u2, then product = e^(i*phi) * I
    phase = product[0, 0]
    expected = phase * np.eye(u1.shape[0])
    return np.allclose(product, expected)


@pytest.mark.parametrize(
    "stim_gate",
    [
        "QUBIT_COORDS(0, 0)",
        # Pauli gates
        "I",
        "I_ERROR",
        "X",
        "Y",
        "Z",
        # Single-qubit Clifford gates
        "C_NXYZ",
        "C_NZYX",
        "C_XNYZ",
        "C_XYNZ",
        "C_XYZ",
        "C_ZNYX",
        "C_ZYNX",
        "C_ZYX",
        "H",
        "H_NXY",
        "H_NXZ",
        "H_NYZ",
        "H_XY",
        "H_XZ",
        "H_YZ",
        "S",
        "SQRT_X",
        "SQRT_X_DAG",
        "SQRT_Y",
        "SQRT_Y_DAG",
        "SQRT_Z",
        "SQRT_Z_DAG",
        "S_DAG",
    ],
)
def test_single_qubit_gate(stim_gate: str):
    c = Circuit(f"{stim_gate} 0")
    stim_c = stim.Circuit(f"{stim_gate} 0")
    stim_c_matrix = stim_c.to_tableau().to_unitary_matrix(endian="big")
    assert unitaries_equal_up_to_global_phase(c.to_matrix(), stim_c_matrix)


def test_t_gate():
    c = Circuit("S[T] 0")
    t_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    assert unitaries_equal_up_to_global_phase(c.to_matrix(), t_matrix)


def test_t_gate_shorthand():
    """Test that T shorthand is equivalent to S[T]."""
    c1 = Circuit("T 0")
    c2 = Circuit("S[T] 0")
    assert c1._stim_circ == c2._stim_circ


def test_t_dag_gate():
    c = Circuit("S_DAG[T] 0")
    t_dag_matrix = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])
    assert unitaries_equal_up_to_global_phase(c.to_matrix(), t_dag_matrix)


def test_t_dag_gate_shorthand():
    """Test that T_DAG shorthand is equivalent to S_DAG[T]."""
    c1 = Circuit("T_DAG 0")
    c2 = Circuit("S_DAG[T] 0")
    assert c1._stim_circ == c2._stim_circ


def test_tpp_shorthand():
    """Test that TPP shorthand is equivalent to SPP[T]."""
    c1 = Circuit("TPP X0*Z1")
    c2 = Circuit("SPP[T] X0*Z1")
    assert c1._stim_circ == c2._stim_circ


def test_tpp_dag_shorthand():
    """Test that TPP_DAG shorthand is equivalent to SPP_DAG[T]."""
    c1 = Circuit("TPP_DAG Y0")
    c2 = Circuit("SPP_DAG[T] Y0")
    assert c1._stim_circ == c2._stim_circ


def test_tpp_append():
    """Test that circuit.append('TPP', ...) works correctly."""
    import stim

    c = Circuit()
    c.append("TPP", [stim.target_x(0), stim.target_combiner(), stim.target_z(1)])
    c2 = Circuit("TPP X0*Z1")
    assert c._stim_circ == c2._stim_circ


def test_tpp_dag_append():
    """Test that circuit.append('TPP_DAG', ...) works correctly."""
    import stim

    c = Circuit()
    c.append("TPP_DAG", [stim.target_y(0)])
    c2 = Circuit("TPP_DAG Y0")
    assert c._stim_circ == c2._stim_circ


def test_rotation_gate_shorthand():
    """Test that R_Z(angle) shorthand is converted correctly."""
    c1 = Circuit("R_Z(0.25) 0")
    c2 = Circuit("I[R_Z(theta=0.25*pi)] 0")
    assert c1._stim_circ == c2._stim_circ

    c1 = Circuit("R_X(-0.5) 0")
    c2 = Circuit("I[R_X(theta=-0.5*pi)] 0")
    assert c1._stim_circ == c2._stim_circ

    c1 = Circuit("R_Y(0.333) 0")
    c2 = Circuit("I[R_Y(theta=0.333*pi)] 0")
    assert c1._stim_circ == c2._stim_circ


def test_u3_gate_shorthand():
    """Test that U3(theta, phi, lambda) shorthand is converted correctly."""
    c1 = Circuit("U3(0.3, 0.24, 0.49) 0")
    c2 = Circuit("I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 0")
    assert c1._stim_circ == c2._stim_circ


@pytest.mark.parametrize(
    "stim_gate",
    [
        "CNOT",
        "CX",
        "CXSWAP",
        "CY",
        "CZ",
        "CZSWAP",
        "ISWAP",
        "ISWAP_DAG",
        "SQRT_XX",
        "SQRT_XX_DAG",
        "SQRT_YY",
        "SQRT_YY_DAG",
        "SQRT_ZZ",
        "SQRT_ZZ_DAG",
        "SWAP",
        "SWAPCX",
        "SWAPCZ",
        "XCX",
        "XCY",
        "XCZ",
        "YCX",
        "YCY",
        "YCZ",
        "ZCX",
        "ZCY",
        "ZCZ",
        "II",
        "II_ERROR",
    ],
)
def test_two_qubit_gate(stim_gate: str):
    c = Circuit(f"{stim_gate} 0 1")
    stim_c = stim.Circuit(f"{stim_gate} 0 1")
    stim_c_matrix = stim_c.to_tableau().to_unitary_matrix(endian="big")
    assert unitaries_equal_up_to_global_phase(c.to_matrix(), stim_c_matrix)


def test_num_measurements():
    c = Circuit()
    assert c.num_measurements == 0

    c = Circuit("M 0")
    assert c.num_measurements == 1

    c = Circuit("M 0 1 2")
    assert c.num_measurements == 3


def test_num_detectors():
    c = Circuit()
    assert c.num_detectors == 0

    c = Circuit("M 0\nDETECTOR rec[-1]")
    assert c.num_detectors == 1

    c = Circuit("M 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]")
    assert c.num_detectors == 2


def test_num_observables():
    c = Circuit("M 0")
    assert c.num_observables == 0

    c = Circuit("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]")
    assert c.num_observables == 1

    c = Circuit("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(2) rec[-2]")
    assert c.num_observables == 3

    c = Circuit(
        "M 0 1 2\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"
        "OBSERVABLE_INCLUDE(2) rec[-2]\n"
        "OBSERVABLE_INCLUDE(5) rec[-1] rec[-2]"
    )
    assert c.num_observables == 6


def test_num_qubits():
    c = Circuit()
    assert c.num_qubits == 0

    c = Circuit("H 0")
    assert c.num_qubits == 1

    c = Circuit("H 0\nX 5")
    assert c.num_qubits == 6

    c = Circuit("H 0\nX 5\nCNOT 2 3")
    assert c.num_qubits == 6


def test_from_stim_program():
    stim_circ = stim.Circuit("H 0\nCNOT 0 1\nM 0 1")
    c = Circuit.from_stim_program(stim_circ)
    assert c._stim_circ == stim_circ


def test_from_stim_program_text():
    c = Circuit("H 0\nCNOT 0 1\nM 0 1")
    assert c._stim_circ == stim.Circuit("H 0\nCNOT 0 1\nM 0 1")


def test_circuit_copy():
    c1 = Circuit("H 0\nCNOT 0 1")
    c2 = c1.copy()
    assert c1 == c2
    assert c1 is not c2


def test_circuit_add():
    c1 = Circuit("H 0")
    c2 = Circuit("CNOT 0 1")
    c3 = c1 + c2
    assert c3._stim_circ == c1._stim_circ + c2._stim_circ


def test_circuit_iadd():
    c1 = Circuit("H 0")
    c2 = Circuit("CNOT 0 1")

    c1_stim = c1._stim_circ.copy()
    c2_stim = c2._stim_circ.copy()

    c1 += c2
    assert c1._stim_circ == c1_stim + c2_stim


def test_circuit_mul():
    c1 = Circuit("H 0")
    c1_stim = c1._stim_circ.copy()
    c2 = c1 * 3
    assert c2._stim_circ == c1_stim * 3


def test_circuit_without_noise():
    c = Circuit("H 0\nDEPOLARIZE1(0.01) 0\nM 0")
    c_clean = c.without_noise()
    assert c_clean._stim_circ == c._stim_circ.without_noise()


def test_circuit_without_annotations():
    c = Circuit("H 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nDETECTOR rec[-1]\nM 0")
    c_clean = c.without_annotations()
    assert c_clean._stim_circ == stim.Circuit("H 0\nM 0")


def test_without_annotations_repeat_block():
    c = Circuit("H 0")
    block = stim.CircuitRepeatBlock(
        3, stim.Circuit("CNOT 0 1\nM 0\nDETECTOR rec[-1]\nM 0")
    )
    c.append(block)
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)

    c_clean = c.without_annotations()
    # structure should be preserved
    assert len(c_clean) == 2
    inst = c_clean[1]
    assert isinstance(inst, stim.CircuitRepeatBlock)
    assert inst.repeat_count == 3
    # annotations should be stripped inside the repeat block too
    assert c_clean.flattened() == c.flattened().without_annotations()


def test_circuit_eq():
    c1 = Circuit("H 0")
    c2 = Circuit("H 0")
    c3 = Circuit("X 0")
    assert c1 == c2
    assert c1 != c3


def test_from_file_preprocesses_shorthand(tmp_path):
    path = tmp_path / "prog.stim"
    path.write_text("T 0\nT_DAG 1\nR_Z(0.25) 0\n", encoding="utf-8")

    loaded = Circuit.from_file(path)
    expected = Circuit("T 0\nT_DAG 1\nR_Z(0.25) 0\n")

    assert loaded._stim_circ == expected._stim_circ


def test_append_from_stim_program_text():
    c = Circuit("H 0")
    c.append_from_stim_program_text("CNOT 0 1\nM 0 1")
    expected = Circuit("H 0\nCNOT 0 1\nM 0 1")
    assert c == expected


def test_append_from_stim_program_text_t_gate():
    c = Circuit("H 0")
    c.append_from_stim_program_text("T 0")
    expected = Circuit("H 0\nT 0")
    assert c._stim_circ == expected._stim_circ


def test_append_from_stim_program_text_empty():
    c = Circuit("H 0")
    c.append_from_stim_program_text("")
    expected = Circuit("H 0")
    assert c == expected


def test_circuit_repr():
    """Test that __repr__ returns a string that can recreate the circuit."""
    c = Circuit("H 0\nCNOT 0 1")
    repr_str = repr(c)
    assert repr_str.startswith("tsim.Circuit('''")
    assert repr_str.endswith("''')")
    # The repr should contain the circuit content
    assert "H 0" in repr_str


def test_circuit_str():
    c = Circuit("H 0\nCNOT 0 1")
    str_repr = str(c)
    assert "H 0" in str_repr
    assert "CX 0 1" in str_repr or "CNOT 0 1" in str_repr


def test_circuit_str_empty():
    c = Circuit()
    assert str(c) == ""


def test_circuit_len_empty():
    """Test length of empty circuit."""
    c = Circuit()
    assert len(c) == 0


def test_circuit_len():
    """Test length of circuit with instructions."""
    c = Circuit("H 0\nCNOT 0 1\nM 0 1")
    assert len(c) == 3


def test_circuit_imul():
    """Test in-place multiplication."""
    c = Circuit("H 0")
    c *= 3
    assert c.flattened() == Circuit("H 0\nH 0\nH 0")


def test_circuit_imul_zero():
    """Test in-place multiplication by zero."""
    c = Circuit("H 0")
    c *= 0
    assert len(c) == 0


def test_circuit_rmul():
    """Test right multiplication (n * circuit)."""
    c = Circuit("H 0")
    result = 3 * c
    assert result.flattened() == Circuit("H 0\nH 0\nH 0")


def test_circuit_getitem_int():
    c = Circuit("H 0\nX 1\nCNOT 0 1")
    instr = c[0]
    assert isinstance(instr, stim.CircuitInstruction)
    assert instr.name == "H"


def test_get_item_type_error():
    c = Circuit("H 0\nX 1\nCNOT 0 1")
    with pytest.raises(TypeError):
        c[None]  # type: ignore


def test_circuit_getitem_negative_int():
    c = Circuit("H 0\nX 1\nCNOT 0 1")
    instr = c[-1]
    assert isinstance(instr, stim.CircuitInstruction)
    assert instr.name == "CX"


def test_circuit_getitem_slice():
    c = Circuit("H 0\nX 1\nCNOT 0 1\nM 0 1")
    sliced = c[1:3]
    assert isinstance(sliced, Circuit)
    assert len(sliced) == 2


def test_approx_equals_identical_circuits():
    c1 = Circuit("DEPOLARIZE1(0.01) 0")
    c2 = Circuit("DEPOLARIZE1(0.01) 0")
    assert c1.approx_equals(c2, atol=0.001)


def test_approx_equals_tsim_stim_circuits():
    c1 = Circuit("DEPOLARIZE1(0.01) 0")
    c2 = stim.Circuit("DEPOLARIZE1(0.01) 0")
    assert c1.approx_equals(c2, atol=0.001)


def test_approx_equals_within_tolerance():
    c1 = Circuit()
    c1._stim_circ = stim.Circuit("DEPOLARIZE1(0.010) 0")
    c2 = Circuit()
    c2._stim_circ = stim.Circuit("DEPOLARIZE1(0.011) 0")
    assert c1.approx_equals(c2, atol=0.01)


def test_approx_equals_outside_tolerance():
    c1 = Circuit()
    c1._stim_circ = stim.Circuit("DEPOLARIZE1(0.01) 0")
    c2 = Circuit()
    c2._stim_circ = stim.Circuit("DEPOLARIZE1(0.05) 0")
    assert not c1.approx_equals(c2, atol=0.001)


def test_approx_equals_with_non_circuit():
    c = Circuit("H 0")
    assert not c.approx_equals("not a circuit", atol=0.01)
    assert not c.approx_equals(42, atol=0.01)


def test_stim_circuit_property():
    """Test stim_circuit property returns a copy."""
    c = Circuit("H 0\nCNOT 0 1")
    stim_c = c.stim_circuit
    assert isinstance(stim_c, stim.Circuit)
    assert stim_c == c._stim_circ
    stim_c.append("X", [0], [])
    assert stim_c != c._stim_circ


def test_num_ticks_empty():
    c = Circuit("H 0")
    assert c.num_ticks == 0


def test_num_ticks():
    c = Circuit("H 0\nTICK\nCNOT 0 1\nTICK\nM 0")
    assert c.num_ticks == 2


def test_pop_last():
    c = Circuit("H 0\nX 1\nCNOT 0 1")
    instr = c.pop()
    assert instr.name == "CX"
    assert len(c) == 2


def test_pop_index():
    c = Circuit("H 0\nX 1\nCNOT 0 1")
    instr = c.pop(0)
    assert instr.name == "H"
    assert len(c) == 2


def test_pop_index_error():
    c = Circuit("H 0")
    with pytest.raises(IndexError):
        c.pop(5)


def test_circuit_iadd_with_stim_circuit():
    c = Circuit("H 0")
    stim_c = stim.Circuit("CNOT 0 1")
    c += stim_c
    expected = Circuit("H 0\nCNOT 0 1")
    assert c == expected


def test_circuit_add_with_stim_circuit():
    c = Circuit("H 0")
    stim_c = stim.Circuit("CNOT 0 1")
    result = c + stim_c
    expected = Circuit("H 0\nCNOT 0 1")
    assert result == expected


def test_compile_m2d_converter():
    c = Circuit("H 0\nM 0\nDETECTOR rec[-1]")
    converter = c.compile_m2d_converter()
    assert isinstance(converter, stim.CompiledMeasurementsToDetectionEventsConverter)


def test_compile_m2d_converter_skip_reference():
    c = Circuit("M 0\nDETECTOR rec[-1]")
    converter = c.compile_m2d_converter(skip_reference_sample=True)
    assert isinstance(converter, stim.CompiledMeasurementsToDetectionEventsConverter)


def test_tcount_no_t_gates():
    c = Circuit("H 0\nCNOT 0 1")
    assert c.tcount() == 0


def test_tcount_with_t_gates():
    c = Circuit("H 0\nT 0\nT 1\nT 0")
    assert c.tcount() == 3


def test_is_clifford_with_stim_gates():
    c = Circuit("H 0\nCNOT 0 1\nM 0 1\nDETECTOR rec[-1]")
    assert c.is_clifford


def test_is_clifford_with_half_pi_parametric_gates():
    c = Circuit("R_Z(0.5) 0\nR_X(-1.5) 0\nU3(0.5, -1.0, 1.5) 0")
    assert c.is_clifford


def test_is_clifford_rejects_t_gate():
    c = Circuit("T 0")
    assert not c.is_clifford


def test_is_clifford_rejects_tpp():
    c = Circuit("TPP Z0")
    assert not c.is_clifford


def test_is_clifford_rejects_tpp_dag():
    c = Circuit("TPP_DAG X0*Y1")
    assert not c.is_clifford


def _assert_stim_tsim_samples_match(program: str, n_shots: int = 16) -> None:
    """Assert stim and tsim produce the same samples for a deterministic circuit.

    Samples the program ``n_shots`` times with stim. All rows must be identical
    (otherwise the test circuit is not deterministic and the test is invalid).
    Then samples with tsim's compiled sampler and verifies exact agreement.
    """
    stim_samples = stim.Circuit(program).compile_sampler().sample(n_shots)
    assert np.all(stim_samples == stim_samples[0]), (
        f"Test circuit is not deterministic under stim:\n{program}\n"
        f"samples:\n{stim_samples}"
    )

    tsim_samples = (
        Circuit(program).compile_sampler(seed=0).sample(n_shots, batch_size=n_shots)
    )
    np.testing.assert_array_equal(tsim_samples, stim_samples)


@pytest.mark.parametrize(
    "instruction,inverse",
    [
        ("SPP Z0", "SPP_DAG Z0"),
        ("SPP X0", "SPP_DAG X0"),
        ("SPP Y0", "SPP_DAG Y0"),
        ("SPP_DAG Z0", "SPP Z0"),
        ("SPP_DAG X0", "SPP X0"),
        ("SPP X0*Z1", "SPP_DAG X0*Z1"),
        ("SPP_DAG Y0*Y1", "SPP Y0*Y1"),
        ("SPP !X0", "SPP X0"),  # ! toggles dagger: SPP !X0 == SPP_DAG X0
        ("SPP !Z0*X1", "SPP Z0*X1"),
        ("SPP X0*X0", ""),  # cancels to identity, no inverse needed
        ("SPP X0*Y1*X0", "SPP_DAG Y1"),  # reduces to SPP Y1
        ("SPP_DAG Z0*Z0", ""),  # cancels to identity
    ],
)
def test_spp_stim_tsim_equivalence(instruction: str, inverse: str):
    """SPP followed by its inverse should give identity; stim and tsim must agree."""
    program = f"R 0 1\n{instruction}\n{inverse}\nM 0 1"
    _assert_stim_tsim_samples_match(program)


@pytest.mark.parametrize(
    "instruction",
    [
        "MPP Z0",  # +1 eigenstate: |0⟩ → 0
        "MPP !Z0",  # flipped → 1
        "MPP Z1",
        "MPP Z0*Z1",  # +1 eigenstate: |00⟩ → 0
        "MPP !Z0*Z1",
        "MPP Z0 Z1",  # two products, both deterministic
        "MPP X0*X0",  # identity → 0
        "MPP !X0*X0",  # identity flipped → 1
        "MPP Y1*Y1",  # identity → 0
        "MPP Z0*X0*Z0*X0",  # -I → 1
        "MPP !Z0*X0*Z0*X0",  # -I flipped → 0
    ],
)
def test_mpp_stim_tsim_equivalence(instruction: str):
    """MPP measurement on |00⟩ should match between stim and tsim."""
    program = f"R 0 1\n{instruction}"
    _assert_stim_tsim_samples_match(program)


def test_is_clifford_rejects_non_clifford_rotation():
    c = Circuit("H 0\nR_Z(0.25) 0\nCNOT 0 1")
    assert not c.is_clifford


def test_is_clifford_rejects_non_clifford_u3():
    c = Circuit("U3(0.5, 0.25, 1.0) 0")
    assert not c.is_clifford


def test_stim_circuit_repeat_block_preserves_non_clifford():
    """REPEAT blocks containing non-Clifford gates round-trip through stim_circuit."""
    c = Circuit("REPEAT 2 {\n    T 0\n}")
    stim_c = c.stim_circuit
    # T is encoded internally as S with tag "T"; the block structure is preserved.
    assert len(stim_c) == 1
    block = stim_c[0]
    assert isinstance(block, stim.CircuitRepeatBlock)
    assert block.repeat_count == 2


def test_get_graph():
    """Test get_graph returns a ZX graph."""
    c = Circuit("H 0\nCNOT 0 1")
    g = c.get_graph()
    # Check it's a pyzx graph-like object
    assert hasattr(g, "vertices")
    assert hasattr(g, "edges")


def test_get_sampling_graph_measurements():
    """Test get_sampling_graph for measurements."""
    c = Circuit("H 0\nM 0")
    g = c.get_sampling_graph(sample_detectors=False)
    assert hasattr(g, "vertices")


def test_get_sampling_graph_detectors():
    """Test get_sampling_graph for detectors."""
    c = Circuit("H 0\nM 0\nDETECTOR rec[-1]")
    g = c.get_sampling_graph(sample_detectors=True)
    assert hasattr(g, "vertices")


def test_to_tensor():
    c = Circuit("H 0")
    tensor = c.to_tensor()
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (2, 2)


def test_detector_error_model_basic():
    """Test detector_error_model returns a DEM."""
    c = Circuit("H 0\nDEPOLARIZE1(0.01) 0\nM 0\nDETECTOR rec[-1]")
    dem = c.detector_error_model(allow_gauge_detectors=True)
    assert isinstance(dem, stim.DetectorErrorModel)


def test_inverse_r_z():
    """Test inverse of R_Z gate."""
    c = Circuit("R_Z(0.25) 0")
    c_inv = c.inverse()
    assert unitaries_equal_up_to_global_phase((c + c_inv).to_matrix(), np.eye(2))


def test_inverse_simple():
    c = Circuit("H 0\nS 0")
    c_inv = c.inverse()
    assert isinstance(c_inv, Circuit)
    assert len(c_inv) == len(c)


def test_inverse_identity():
    c = Circuit("H 0")
    c_inv = c.inverse()
    assert unitaries_equal_up_to_global_phase((c + c_inv).to_matrix(), np.eye(2))


def test_inverse_r_x():
    c = Circuit("R_X(0.3) 0")
    c_inv = c.inverse()
    assert unitaries_equal_up_to_global_phase((c + c_inv).to_matrix(), np.eye(2))


def test_inverse_r_y():
    c = Circuit("R_Y(-0.5) 0")
    c_inv = c.inverse()
    assert unitaries_equal_up_to_global_phase((c + c_inv).to_matrix(), np.eye(2))


def test_inverse_u3():
    c = Circuit("U3(0.3, 0.24, 0.49) 0")
    c_inv = c.inverse()
    assert unitaries_equal_up_to_global_phase((c + c_inv).to_matrix(), np.eye(2))


def test_inverse_t_gate():
    c = Circuit("T 0")
    c_inv = c.inverse()
    assert unitaries_equal_up_to_global_phase((c + c_inv).to_matrix(), np.eye(2))


def test_inverse_t_dag_gate():
    c = Circuit("T_DAG 0")
    c_inv = c.inverse()
    assert unitaries_equal_up_to_global_phase((c + c_inv).to_matrix(), np.eye(2))


def test_inverse_tpp():
    c = Circuit("TPP Z0")
    c_inv = c.inverse()
    assert unitaries_equal_up_to_global_phase((c + c_inv).to_matrix(), np.eye(2))


def test_inverse_tpp_dag():
    c = Circuit("TPP_DAG X0*Y1")
    c_inv = c.inverse()
    combined = (c + c_inv).to_matrix()
    assert unitaries_equal_up_to_global_phase(combined, np.eye(combined.shape[0]))


def test_inverse_mixed_circuit():
    c = Circuit("H 0\nT 0\nR_Z(0.22) 0\nCNOT 0 1")
    c_inv = c.inverse()
    combined = (c + c_inv).to_matrix()
    assert unitaries_equal_up_to_global_phase(combined, np.eye(combined.shape[0]))


def test_inverse_with_repeat_block():
    c = Circuit("H 0\nT 0\nR_Z(0.22) 0\nCNOT 0 1")
    c_repeat = c * 3
    c_inv = c_repeat.inverse()
    # inverse should preserve repeat structure, not flatten
    assert len(c_inv) == len(c_repeat)
    assert isinstance(c_inv[0], stim.CircuitRepeatBlock)
    assert c_inv.flattened() == c_repeat.flattened().inverse()
    combined = (c_repeat + c_inv).to_matrix()
    assert unitaries_equal_up_to_global_phase(combined, np.eye(combined.shape[0]))


def test_diagram_timeline_svg():
    c = Circuit("H 0\nCNOT 0 1\nM 0 1")
    diagram = c.diagram(type="timeline-svg")
    svg_str = str(diagram)
    assert "<svg" in svg_str
    assert "</svg>" in svg_str


def test_diagram_timeslice_svg():
    c = Circuit("H 0\nTICK\nCNOT 0 1\nTICK\nM 0 1")
    diagram = c.diagram(type="timeslice-svg", tick=range(0, 2))
    svg_str = str(diagram)
    assert "<svg" in svg_str


def test_diagram_pyzx():
    c = Circuit("H 0\nCNOT 0 1")
    with patch("pyzx_param.draw") as mock_draw:
        g = c.diagram(type="pyzx")
        mock_draw.assert_called_once()
    assert hasattr(g, "vertices")
    assert hasattr(g, "edges")


def test_diagram_pyzx_empty():
    c = Circuit()
    with patch("pyzx_param.draw") as mock_draw:
        g = c.diagram(type="pyzx")
        mock_draw.assert_not_called()
    assert len(g.vertices()) == 0


def test_diagram_pyzx_meas():
    c = Circuit("H 0\nM 0")
    with patch("pyzx_param.draw") as mock_draw:
        g = c.diagram(type="pyzx-meas")
        mock_draw.assert_called_once()
    assert hasattr(g, "vertices")


def test_diagram_pyzx_dets():
    c = Circuit("H 0\nM 0\nDETECTOR rec[-1]")
    with patch("pyzx_param.draw") as mock_draw:
        g = c.diagram(type="pyzx-dets")
        mock_draw.assert_called_once()
    assert hasattr(g, "vertices")


@pytest.mark.parametrize("type", ["pyzx", "pyzx-meas", "pyzx-dets"])
def test_diagram_pyzx_scale_horizontally(
    type: Literal["pyzx", "pyzx-meas", "pyzx-dets"],
):
    c = Circuit("H 0\nCNOT 0 1")
    with patch("pyzx_param.draw") as mock_draw:
        g = c.diagram(type=type, scale_horizontally=2)
        mock_draw.assert_called_once()
    assert hasattr(g, "vertices")


def test_append():
    c = Circuit()
    c.append("T", [0, 1])
    assert str(c) == "T 0 1"

    c.append("T_DAG", 2)
    assert "T_DAG 2" in str(c)

    c.append("R_Z", 0, arg=0.25)
    assert "R_Z(0.25) 0" in str(c)

    c.append("R_X", 1, arg=[0.1])
    assert "R_X(0.1) 1" in str(c)

    c.append("U3", 0, arg=(0.3, 0.24, 0.49))
    assert "U3(0.3, 0.24, 0.49) 0" in str(c)


def test_append_circuit_instruction():
    c = Circuit()
    c.append(stim.CircuitInstruction("H", [0]))
    assert str(c) == "H 0"


def test_append_circuit_repeat_block():
    c = Circuit()
    block = stim.CircuitRepeatBlock(3, stim.Circuit("H 0"))
    c.append(block)
    assert str(c.flattened()) == "H 0 0 0"
    assert len(c) == 1  # single repeat block


def test_append_circuit():
    c = Circuit()
    sub_c = stim.Circuit("H 0\nCNOT 0 1")
    c.append(sub_c)
    assert "H 0" in str(c)
    assert "CX 0 1" in str(c) or "CNOT 0 1" in str(c)


def test_append_u3_with_generator_arg():
    """U3 must work when arg is a one-shot generator (not just a list)."""
    c = Circuit()
    c.append("U3", 0, (x for x in [0.3, 0.24, 0.49]))
    # The circuit should contain a single U3 instruction.
    assert len(c) == 1
    assert "U3" in str(c)


def test_append_repetition_code():
    stim_c = stim.Circuit.generated("repetition_code:memory", distance=2, rounds=4)
    c = Circuit()
    for instr in stim_c:
        c.append(instr)

    assert str(c.flattened()) == str(stim_c.flattened())
    assert str(c) == str(stim_c)


def _circuit_with_repeat_block() -> Circuit:
    """Helper: build a Circuit that contains a REPEAT block."""
    c = Circuit("H 0")
    block = stim.CircuitRepeatBlock(5, stim.Circuit("CNOT 0 1\nTICK"))
    c.append(block)
    c.append("M", [0, 1])
    return c


def test_mul_preserves_repeat_block():
    """c * n should wrap in a repeat block, not flatten."""
    c = Circuit("H 0\nCNOT 0 1")
    c2 = c * 4
    assert c2._stim_circ == c._stim_circ * 4
    # flattened form should equal the naive expansion
    assert c2.flattened() == c + c + c + c


def test_imul_preserves_repeat_block():
    c = Circuit("H 0\nCNOT 0 1")
    flat_4x = c + c + c + c
    c *= 4
    assert c.flattened() == flat_4x


def test_getitem_repeat_block():
    """Indexing into a circuit may return a CircuitRepeatBlock."""
    c = _circuit_with_repeat_block()
    item = c[1]
    assert isinstance(item, stim.CircuitRepeatBlock)
    assert item.repeat_count == 5


def test_getitem_slice_with_repeat_block():
    c = _circuit_with_repeat_block()
    sliced = c[0:2]
    assert isinstance(sliced, Circuit)
    assert len(sliced) == 2


def test_pop_repeat_block():
    c = Circuit()
    block = stim.CircuitRepeatBlock(3, stim.Circuit("X 0"))
    c.append(block)
    popped = c.pop()
    assert isinstance(popped, stim.CircuitRepeatBlock)
    assert popped.repeat_count == 3
    assert len(c) == 0


def test_copy_preserves_repeat_block():
    c = _circuit_with_repeat_block()
    c2 = c.copy()
    assert c == c2
    assert c is not c2
    assert str(c) == str(c2)


def test_is_clifford_repeat_block_half_pi_parametric():
    c = Circuit()
    c.append("R_Z", [0], 0.5)
    c = c * 4
    assert c.is_clifford


def test_is_clifford_repeat_block_clifford_body():
    c = Circuit("REPEAT 3 {\n    H 0\n    CNOT 0 1\n}")
    assert c.is_clifford


def test_is_clifford_repeat_block_rejects_non_clifford_body():
    c = Circuit("REPEAT 2 {\n    T 0\n}")
    assert not c.is_clifford


def test_is_clifford_repeat_block_rejects_non_clifford_parametric():
    c = Circuit()
    c.append("R_Z", [0], 0.25)
    c = c * 2
    assert not c.is_clifford


def test_stim_circuit_repeat_block_keeps_non_clifford_parametric():
    """Non-Clifford parametric rotations inside REPEAT are not expanded."""
    c = Circuit()
    c.append("R_X", [0], 0.3)
    c = c * 2
    expanded = c.stim_circuit
    block = expanded[0]
    assert isinstance(block, stim.CircuitRepeatBlock)
    body = block.body_copy()
    instr = body[0]
    assert instr.name == "I"
    assert instr.tag == "R_X(theta=0.3*pi)"


def test_stim_circuit_repeat_block_expands_half_pi_parametric():
    """Half-π parametric rotations inside REPEAT are expanded to Clifford gates."""
    c = Circuit()
    c.append("R_X", [0], 0.5)
    c = c * 3
    expanded = c.stim_circuit
    assert len(expanded) == 1
    block = expanded[0]
    assert isinstance(block, stim.CircuitRepeatBlock)
    assert block.repeat_count == 3
    body = block.body_copy()
    assert [instr.name for instr in body] == ["SQRT_X"]
