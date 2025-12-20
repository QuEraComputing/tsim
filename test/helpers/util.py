import numpy as np

from tsim.sampler import CompiledStateProbs


def get_statevector(prob_sampler: CompiledStateProbs) -> np.ndarray:
    """Get the statevector from a probability sampler."""
    num_qubits = prob_sampler.circuit.num_qubits
    sv = []
    for i in range(2**num_qubits):
        state = np.zeros(num_qubits, dtype=np.bool_)
        for j in range(num_qubits):
            state[j] = (i >> j) & 1
        state = state[::-1]
        sv.append(prob_sampler.probability_of(state, batch_size=1)[0])
    return np.array(sv)


def get_matrix(prob_sampler: CompiledStateProbs) -> np.ndarray:
    """Sample a statevector and reshape it into a square matrix."""
    num_qubits = prob_sampler.circuit.num_qubits
    sv = get_statevector(prob_sampler)
    return (np.array(sv) * 2 ** (num_qubits // 2)).reshape(
        (2 ** (num_qubits // 2), 2 ** (num_qubits // 2))
    )


def plot_comparison(
    stim_state: np.ndarray,
    tsim_state: np.ndarray,
    exact_state: np.ndarray,
    plot_difference: bool = True,
) -> None:
    """Plot a comparison of state probabilities from different simulators.

    Args:
        stim_state: Probabilities from VecSampler.
        tsim_state: Probabilities from tsim.
        exact_state: Probabilities from pyzx tensor contraction.
        plot_difference: If True, plot differences relative to tsim_state.
        xlim: Optional x-axis limits for the plot.
    """
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    bar_width = 0.2
    x_pos = np.arange(len(exact_state))

    if plot_difference:
        stim_values = stim_state - exact_state
        tsim_values = tsim_state - exact_state
        exact_values = exact_state - exact_state
        ylabel = "Difference in Probability to exact"
    else:
        stim_values = stim_state
        tsim_values = tsim_state
        exact_values = exact_state
        ylabel = "Probability"

    ax.bar(
        x_pos - bar_width,
        stim_values,
        bar_width,
        label="statevector simulator",
        color="blue",
        alpha=0.7,
        edgecolor="black",
    )
    ax.bar(
        x_pos,
        tsim_values,
        bar_width,
        label="tsim",
        color="orange",
        alpha=0.7,
        edgecolor="black",
    )
    ax.bar(
        x_pos + bar_width,
        exact_values,
        bar_width,
        label="pyzx tensor contraction",
        color="green",
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_xlabel("State (as integer)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()
