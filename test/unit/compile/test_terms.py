import jax.numpy as jnp
import numpy as np
import pytest

from tsim.compile.terms import HalfPiPhases, NodePhases, PhasePairs, PiProducts


def _ref_parity(bits: np.ndarray, param_vals: np.ndarray) -> np.ndarray:
    """Parity of (bits AND param_vals) over the last axis. Shapes (G,T,P) and (B,P) → (B,G,T)."""
    return ((param_vals @ bits.reshape(-1, bits.shape[-1]).T) % 2).reshape(
        param_vals.shape[0], bits.shape[0], bits.shape[1]
    )


def _ref_node_phases(phases, params, counts, param_vals):
    parity = _ref_parity(params, param_vals)
    # 1 + exp(i·π·(α + parity)),  with α = phases/4
    term = 1 + np.exp(1j * np.pi * phases[None] / 4 + 1j * np.pi * parity)
    mask = np.arange(phases.shape[1])[None, :] < counts[:, None]
    term = np.where(mask[None, :, :], term, 1.0)
    return np.prod(term, axis=-1)


def _ref_halfpi_phases(coeffs, params, param_vals):
    parity = _ref_parity(params, param_vals)
    # Per term: exp(i·π·j'·parity/2),  with coeffs = 2·j' (so coeffs/4 = j'/2).
    per_term = np.exp(1j * np.pi * coeffs[None] * parity / 4)
    return np.prod(per_term, axis=-1)


def _ref_pi_products(psi_const, psi_params, phi_const, phi_params, param_vals):
    psi = (psi_const[None] + _ref_parity(psi_params, param_vals)) % 2
    phi = (phi_const[None] + _ref_parity(phi_params, param_vals)) % 2
    # (-1)^(ψ·φ) = exp(i·π·ψ·φ)
    sign = np.exp(1j * np.pi * psi * phi)
    return np.prod(sign, axis=-1)


def _ref_phase_pairs(alpha, alpha_params, beta, beta_params, counts, param_vals):
    a_par = _ref_parity(alpha_params, param_vals)
    b_par = _ref_parity(beta_params, param_vals)
    # ω^(α + 4·parity) = exp(i·π·α/4) · exp(i·π·parity)
    ea = np.exp(1j * np.pi * alpha[None] / 4 + 1j * np.pi * a_par)
    eb = np.exp(1j * np.pi * beta[None] / 4 + 1j * np.pi * b_par)
    term = 1 + ea + eb - ea * eb
    mask = np.arange(alpha.shape[1])[None, :] < counts[:, None]
    term = np.where(mask[None, :, :], term, 1.0)
    return np.prod(term, axis=-1)


def _eval_complex(module, param_vals: np.ndarray) -> np.ndarray:
    return np.asarray(module.evaluate(jnp.asarray(param_vals)).to_complex())


class TestNodePhases:
    @pytest.mark.parametrize("seed", (0, 42))
    def test_matches_reference(self, seed):
        np.random.seed(seed)
        G, T, P, B = 3, 4, 5, 7
        phases = np.random.randint(0, 8, size=(G, T)).astype(np.uint8)
        params = np.random.randint(0, 2, size=(G, T, P)).astype(np.uint8)
        counts = np.array([T, T - 1, 0], dtype=np.int32)
        param_vals = np.random.randint(0, 2, size=(B, P)).astype(np.uint8)

        module = NodePhases(
            jnp.asarray(phases), jnp.asarray(params), jnp.asarray(counts)
        )
        got = _eval_complex(module, param_vals)
        expected = _ref_node_phases(phases, params, counts, param_vals)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    @pytest.mark.parametrize("seed", (0, 42))
    def test_padding_is_identity(self, seed):
        """Real terms followed by nonzero padded slots should evaluate to the real-only product."""
        np.random.seed(seed)
        G, P, B = 2, 4, 3
        counts = np.array([1, 2], dtype=np.int32)
        phases_real = np.random.randint(0, 8, size=(G, 2)).astype(np.uint8)
        params_real = np.random.randint(0, 2, size=(G, 2, P)).astype(np.uint8)
        # Append a nonzero padded slot that must be masked away by counts.
        phases = np.concatenate(
            [phases_real, np.full((G, 1), 5, dtype=np.uint8)], axis=1
        )
        params = np.concatenate(
            [params_real, np.ones((G, 1, P), dtype=np.uint8)], axis=1
        )
        param_vals = np.random.randint(0, 2, size=(B, P)).astype(np.uint8)

        module = NodePhases(
            jnp.asarray(phases), jnp.asarray(params), jnp.asarray(counts)
        )
        got = _eval_complex(module, param_vals)
        expected = _ref_node_phases(phases, params, counts, param_vals)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_max_terms_zero(self):
        G, P, B = 2, 3, 4
        phases = jnp.zeros((G, 0), dtype=jnp.uint8)
        params = jnp.zeros((G, 0, P), dtype=jnp.uint8)
        counts = jnp.zeros((G,), dtype=jnp.int32)
        param_vals = jnp.zeros((B, P), dtype=jnp.uint8)

        module = NodePhases(phases, params, counts)
        got = np.asarray(module.evaluate(param_vals).to_complex())
        np.testing.assert_allclose(got, np.ones((B, G), dtype=complex))


class TestHalfPiPhases:
    @pytest.mark.parametrize("seed", (0, 42))
    def test_matches_reference(self, seed):
        np.random.seed(seed)
        G, T, P, B = 3, 4, 5, 6
        coeffs = np.random.choice([0, 2, 4, 6], size=(G, T)).astype(np.uint8)
        params = np.random.randint(0, 2, size=(G, T, P)).astype(np.uint8)
        param_vals = np.random.randint(0, 2, size=(B, P)).astype(np.uint8)

        module = HalfPiPhases(jnp.asarray(coeffs), jnp.asarray(params))
        got = _eval_complex(module, param_vals)
        expected = _ref_halfpi_phases(coeffs, params, param_vals)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    @pytest.mark.parametrize("seed", (0, 42))
    def test_zero_coeff_is_no_op(self, seed):
        """Slots with coeff=0 must contribute nothing regardless of params content."""
        np.random.seed(seed)
        G, P, B = 2, 4, 3
        coeffs = np.array([[2, 0, 4, 0], [0, 0, 6, 0]], dtype=np.uint8)
        params = np.random.randint(0, 2, size=(G, 4, P)).astype(np.uint8)
        param_vals = np.random.randint(0, 2, size=(B, P)).astype(np.uint8)

        module = HalfPiPhases(jnp.asarray(coeffs), jnp.asarray(params))
        got = _eval_complex(module, param_vals)
        expected = _ref_halfpi_phases(coeffs, params, param_vals)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_max_terms_zero(self):
        G, P, B = 2, 3, 4
        coeffs = jnp.zeros((G, 0), dtype=jnp.uint8)
        params = jnp.zeros((G, 0, P), dtype=jnp.uint8)
        param_vals = jnp.zeros((B, P), dtype=jnp.uint8)

        module = HalfPiPhases(coeffs, params)
        got = np.asarray(module.evaluate(param_vals).to_complex())
        np.testing.assert_allclose(got, np.ones((B, G), dtype=complex))


class TestPiProducts:
    @pytest.mark.parametrize("seed", (0, 42))
    def test_matches_reference(self, seed):
        np.random.seed(seed)
        G, T, P, B = 3, 4, 5, 6
        psi_const = np.random.randint(0, 2, size=(G, T)).astype(np.uint8)
        psi_params = np.random.randint(0, 2, size=(G, T, P)).astype(np.uint8)
        phi_const = np.random.randint(0, 2, size=(G, T)).astype(np.uint8)
        phi_params = np.random.randint(0, 2, size=(G, T, P)).astype(np.uint8)
        param_vals = np.random.randint(0, 2, size=(B, P)).astype(np.uint8)

        module = PiProducts(
            jnp.asarray(psi_const),
            jnp.asarray(psi_params),
            jnp.asarray(phi_const),
            jnp.asarray(phi_params),
        )
        got = _eval_complex(module, param_vals)
        expected = _ref_pi_products(
            psi_const, psi_params, phi_const, phi_params, param_vals
        )
        np.testing.assert_allclose(got, expected, atol=1e-6)

    @pytest.mark.parametrize("seed", (0, 42))
    def test_zero_padding_is_identity(self, seed):
        """All-zero psi/phi slots contribute (-1)^0 = 1, so they are valid padding."""
        np.random.seed(seed)
        G, T, P, B = 2, 5, 3, 4
        psi_const = np.zeros((G, T), dtype=np.uint8)
        psi_params = np.zeros((G, T, P), dtype=np.uint8)
        phi_const = np.zeros((G, T), dtype=np.uint8)
        phi_params = np.zeros((G, T, P), dtype=np.uint8)
        param_vals = np.random.randint(0, 2, size=(B, P)).astype(np.uint8)

        module = PiProducts(
            jnp.asarray(psi_const),
            jnp.asarray(psi_params),
            jnp.asarray(phi_const),
            jnp.asarray(phi_params),
        )
        got = _eval_complex(module, param_vals)
        np.testing.assert_allclose(got, np.ones((B, G), dtype=complex), atol=1e-6)

    def test_max_terms_zero(self):
        G, P, B = 2, 3, 4
        empty = jnp.zeros((G, 0), dtype=jnp.uint8)
        empty_p = jnp.zeros((G, 0, P), dtype=jnp.uint8)
        param_vals = jnp.zeros((B, P), dtype=jnp.uint8)

        module = PiProducts(empty, empty_p, empty, empty_p)
        got = np.asarray(module.evaluate(param_vals).to_complex())
        np.testing.assert_allclose(got, np.ones((B, G), dtype=complex))


class TestPhasePairs:
    @pytest.mark.parametrize("seed", (0, 42))
    def test_matches_reference(self, seed):
        np.random.seed(seed)
        G, T, P, B = 3, 4, 5, 6
        alpha = np.random.randint(0, 8, size=(G, T)).astype(np.uint8)
        alpha_params = np.random.randint(0, 2, size=(G, T, P)).astype(np.uint8)
        beta = np.random.randint(0, 8, size=(G, T)).astype(np.uint8)
        beta_params = np.random.randint(0, 2, size=(G, T, P)).astype(np.uint8)
        counts = np.array([T, T - 2, 0], dtype=np.int32)
        param_vals = np.random.randint(0, 2, size=(B, P)).astype(np.uint8)

        module = PhasePairs(
            jnp.asarray(alpha),
            jnp.asarray(alpha_params),
            jnp.asarray(beta),
            jnp.asarray(beta_params),
            jnp.asarray(counts),
        )
        got = _eval_complex(module, param_vals)
        expected = _ref_phase_pairs(
            alpha, alpha_params, beta, beta_params, counts, param_vals
        )
        np.testing.assert_allclose(got, expected, atol=1e-6)

    @pytest.mark.parametrize("seed", (0, 42))
    def test_padding_is_identity(self, seed):
        """Nonzero entries in padded slots must be masked to the multiplicative identity."""
        np.random.seed(seed)
        G, P, B = 2, 4, 3
        counts = np.array([1, 2], dtype=np.int32)
        alpha_real = np.random.randint(0, 8, size=(G, 2)).astype(np.uint8)
        beta_real = np.random.randint(0, 8, size=(G, 2)).astype(np.uint8)
        ap_real = np.random.randint(0, 2, size=(G, 2, P)).astype(np.uint8)
        bp_real = np.random.randint(0, 2, size=(G, 2, P)).astype(np.uint8)
        alpha = np.concatenate([alpha_real, np.full((G, 1), 3, dtype=np.uint8)], axis=1)
        beta = np.concatenate([beta_real, np.full((G, 1), 5, dtype=np.uint8)], axis=1)
        ap = np.concatenate([ap_real, np.ones((G, 1, P), dtype=np.uint8)], axis=1)
        bp = np.concatenate([bp_real, np.ones((G, 1, P), dtype=np.uint8)], axis=1)
        param_vals = np.random.randint(0, 2, size=(B, P)).astype(np.uint8)

        module = PhasePairs(
            jnp.asarray(alpha),
            jnp.asarray(ap),
            jnp.asarray(beta),
            jnp.asarray(bp),
            jnp.asarray(counts),
        )
        got = _eval_complex(module, param_vals)
        expected = _ref_phase_pairs(alpha, ap, beta, bp, counts, param_vals)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_max_terms_zero(self):
        G, P, B = 2, 3, 4
        empty = jnp.zeros((G, 0), dtype=jnp.uint8)
        empty_p = jnp.zeros((G, 0, P), dtype=jnp.uint8)
        counts = jnp.zeros((G,), dtype=jnp.int32)
        param_vals = jnp.zeros((B, P), dtype=jnp.uint8)

        module = PhasePairs(empty, empty_p, empty, empty_p, counts)
        got = np.asarray(module.evaluate(param_vals).to_complex())
        np.testing.assert_allclose(got, np.ones((B, G), dtype=complex))
