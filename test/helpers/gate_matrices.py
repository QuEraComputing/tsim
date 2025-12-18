import numpy as np

SINGLE_QUBIT_GATE_MATRICES = {
    "I": np.array([[1, 0], [0, 1]]),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
    "T_DAG": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]]),
    "C_XYZ": np.array([[1 - 1j, -1 - 1j], [1 - 1j, 1 + 1j]]) / 2,
    "C_ZYX": np.array([[1 + 1j, 1 + 1j], [-1 + 1j, 1 - 1j]]) / 2,
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "H_XY": np.array([[0, 1 - 1j], [1 + 1j, 0]]) / np.sqrt(2),
    "H_YZ": np.array([[1, -1j], [1j, -1]]) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]]),
    "SQRT_X": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2,
    "SQRT_X_DAG": np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2,
    "SQRT_Y": np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2,
    "SQRT_Y_DAG": np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2,
    "S_DAG": np.array([[1, 0], [0, -1j]]),
}

TWO_QUBIT_GATE_MATRICES = {
    "CNOT": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
    "CY": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
    "CZ": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
    "ISWAP": np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
    "ISWAP_DAG": np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]]),
    "SQRT_XX": np.array(
        [
            [1 + 1j, 0, 0, 1 - 1j],
            [0, 1 + 1j, 1 - 1j, 0],
            [0, 1 - 1j, 1 + 1j, 0],
            [1 - 1j, 0, 0, 1 + 1j],
        ]
    )
    / 2,
    "SQRT_XX_DAG": np.array(
        [
            [1 - 1j, 0, 0, 1 + 1j],
            [0, 1 - 1j, 1 + 1j, 0],
            [0, 1 + 1j, 1 - 1j, 0],
            [1 + 1j, 0, 0, 1 - 1j],
        ]
    )
    / 2,
    "SQRT_YY": np.array(
        [
            [1 + 1j, 0, 0, -1 + 1j],
            [0, 1 + 1j, 1 - 1j, 0],
            [0, 1 - 1j, 1 + 1j, 0],
            [-1 + 1j, 0, 0, 1 + 1j],
        ]
    )
    / 2,
    "SQRT_YY_DAG": np.array(
        [
            [1 - 1j, 0, 0, -1 - 1j],
            [0, 1 - 1j, 1 + 1j, 0],
            [0, 1 + 1j, 1 - 1j, 0],
            [-1 - 1j, 0, 0, 1 - 1j],
        ]
    )
    / 2,
    "SQRT_ZZ": np.array([[1, 0, 0, 0], [0, 1j, 0, 0], [0, 0, 1j, 0], [0, 0, 0, 1]]),
    "SQRT_ZZ_DAG": np.array(
        [[1, 0, 0, 0], [0, -1j, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1]]
    ),
    "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    "XCX": np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]) / 2,
    "XCY": np.array(
        [[1, -1j, 1, 1j], [1j, 1, -1j, 1], [1, 1j, 1, -1j], [-1j, 1, 1j, 1]]
    )
    / 2,
    "XCZ": np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
    "YCX": np.array(
        [[1, 1, -1j, 1j], [1, 1, 1j, -1j], [1j, -1j, 1, 1], [-1j, 1j, 1, 1]]
    )
    / 2,
    "YCY": np.array(
        [[1, -1j, -1j, 1], [1j, 1, -1, -1j], [1j, -1, 1, -1j], [1, 1j, 1j, 1]]
    )
    / 2,
    "YCZ": np.array([[1, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1, 0], [0, 1j, 0, 0]]),
}

# Rotation gate matrices (parametrized by fraction of pi)
ROT_GATE_MATRICES = {
    "R_X": lambda frac: np.array(
        [
            [np.cos(frac * np.pi / 2), -1j * np.sin(frac * np.pi / 2)],
            [-1j * np.sin(frac * np.pi / 2), np.cos(frac * np.pi / 2)],
        ]
    ),
    "R_Z": lambda frac: np.array(
        [[np.exp(-1j * np.pi / 2 * frac), 0], [0, np.exp(1j * np.pi / 2 * frac)]]
    ),
    "U3": lambda frac_theta, frac_phi, frac_lambda: np.array(
        [
            [
                np.cos(frac_theta * np.pi / 2),
                -np.exp(1j * frac_lambda * np.pi) * np.sin(frac_theta * np.pi / 2),
            ],
            [
                np.exp(1j * frac_phi * np.pi) * np.sin(frac_theta * np.pi / 2),
                np.exp(1j * (frac_phi + frac_lambda) * np.pi)
                * np.cos(frac_theta * np.pi / 2),
            ],
        ]
    ),
}
