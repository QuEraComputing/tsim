# tsim

A GPU-accelerated circuit sampler via ZX-calculus stabilizer rank decomposition.
Tsim feels just like [Stim](https://github.com/quantumlib/Stim), but supports non-Clifford gates.

It is based on [Sutcliffe and Kissinger (2025)](https://arxiv.org/abs/2403.06777).

## Installation

Tsim is not yet released on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/QuEraComputing/tsim.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/QuEraComputing/tsim.git
```

If your machine has a GPU, use:
```bash
pip install "git+https://github.com/QuEraComputing/tsim.git#egg=tsim[cuda13]"
```


## Quick Start
An introductory tutorial is available [here](https://github.com/QuEraComputing/tsim/blob/main/docs/demos/encoding_demo.ipynb).

For many existing scripts, replacing `stim` with `tsim` should just work. Tsim mirrors the Stim API and currently supports all [Stim gates](https://github.com/quantumlib/Stim/wiki/Stim-v1.9-Gate-Reference), except `CORRELATED_ERROR` and `ELSE_CORRELATED_ERROR`.

Additionally, Tsim supports the instructions `T`, `T_DAG`, `R_Z`, `R_X`, `R_Y`, and `U3` (see below for more details).
```python
import tsim

c = tsim.Circuit(
    """
    RX 0
    R 1
    T 0
    PAULI_CHANNEL_1(0.1, 0.1, 0.2) 0 1
    H 0
    CNOT 0 1
    DEPOLARIZE2(0.01) 0 1
    M 0 1
    DETECTOR rec[-1] rec[-2]
    """
)

detector_sampler = c.compile_detector_sampler()
samples = detector_sampler.sample(shots=100)
```

Tsim supports non-deterministic detectors and observables. An important consequence is that
Tsim will simulate actual detector samples, whereas Stim only reports detection flips (i.e. detection samples XORed with
a noiseless reference sample). Concretely,
```python
c = tsim.Circuit(
    """
    X 0
    M 0
    DETECTOR rec[-1]
    """
)
sampler = c.compile_detector_sampler()
samples = sampler.sample(shots=100)
print(samples)
```
will report `True` values, whereas the same circuit would result in `False` values in Stim.

## What is Tsim?

Tsim is a quantum circuit simulator that supports fast sampling from Clifford+T circuits with Pauli noise. Its underlying algorithm is stabilizer rank decomposition, together with ZX-calculus simplification rules.

As such, Tsim can simulate hundreds of qubits, as long as the circuit
does not have too many non-Clifford instructions.

Just like Stim, Tsim compiles circuits into measurement or detector samplers.
These samplers manage a contiguous data structure that allows for efficient sampling on CPU or GPU, following the approach described in [Sutcliffe and Kissinger (2025)](https://arxiv.org/abs/2403.06777).

## Supported Instructions

Tsim supports all [Stim instructions](https://github.com/quantumlib/Stim/wiki/Stim-v1.9-Gate-Reference), except currently `CORRELATED_ERROR` and `ELSE_CORRELATED_ERROR`.

In addition, Tsim supports the following non-Clifford instructions:

### 'T' and 'T_DAG'

The T gate applies a π/4 phase rotation, and T_DAG is its inverse:

$$
T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}, \quad
T^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}
$$

```
T 0 1 2  # Apply T to qubits 0, 1, 2
T_DAG 0  # Apply T_DAG to qubit 0
```

### Rotation Gates: 'R_X', 'R_Y', 'R_Z'

Rotation gates around the X, Y, and Z axes by an angle θ = α·π (where α is specified as the parameter):

$$
\begin{align*}
R_X(\alpha) &= \begin{pmatrix} \cos(\alpha\pi/2) & -i\sin(\alpha\pi/2) \\ -i\sin(\alpha\pi/2) & \cos(\alpha\pi/2) \end{pmatrix}\\
R_Y(\alpha) &= \begin{pmatrix} \cos(\alpha\pi/2) & -\sin(\alpha\pi/2) \\ \sin(\alpha\pi/2) & \cos(\alpha\pi/2) \end{pmatrix}\\
R_Z(\alpha) &= \begin{pmatrix} e^{-i\alpha\pi/2} & 0 \\ 0 & e^{i\alpha\pi/2} \end{pmatrix}
\end{align*}
$$

```
R_X(0.5) 0  # Rotate qubit 0 around X by π/2
R_Y(0.25) 1  # Rotate qubit 1 around Y by π/4
R_Z(1.0) 2  # Rotate qubit 2 around Z by π
```

### 'U3' Gate

The general single-qubit unitary with three parameters (θ, φ, λ), each specified as a multiple of π:

$$
U_3(\theta, \phi, \lambda) = \begin{pmatrix} \cos(\theta\pi/2) & -e^{i\lambda\pi}\sin(\theta\pi/2) \\ e^{i\phi\pi}\sin(\theta\pi/2) & e^{i(\phi+\lambda)\pi}\cos(\theta\pi/2) \end{pmatrix}
$$

```
U3(0.5, 0.25, 0.125) 0  # Apply U3 with θ=π/2, φ=π/4, λ=π/8
```
