import numpy as np

from bztetra.twod import occupation_weights
from bztetra.twod import solve_fermi_energy
from tests.twod_cases import free_electron_case


def test_twod_occupation_weights_returns_uniform_weights_for_fully_occupied_flat_band() -> None:
    eigenvalues = np.full((2, 2, 1), -1.0, dtype=np.float64)

    weights = occupation_weights(np.eye(2, dtype=np.float64), eigenvalues)

    assert weights.shape == (2, 2, 1)
    np.testing.assert_allclose(weights[..., 0], np.full((2, 2), 0.25))
    np.testing.assert_allclose(weights[..., 0].sum(), 1.0)


def test_twod_occupation_weights_returns_zero_weights_for_empty_flat_band() -> None:
    eigenvalues = np.full((2, 2, 1), 1.0, dtype=np.float64)

    weights = occupation_weights(np.eye(2, dtype=np.float64), eigenvalues)

    np.testing.assert_allclose(weights[..., 0], 0.0)


def test_twod_occupation_weights_interpolate_constant_band_to_denser_output_grid() -> None:
    eigenvalues = np.full((2, 2, 1), -1.0, dtype=np.float64)

    weights = occupation_weights(
        np.eye(2, dtype=np.float64),
        eigenvalues,
        weight_grid_shape=(4, 4),
    )

    assert weights.shape == (4, 4, 1)
    expected = np.zeros((4, 4), dtype=np.float64)
    expected[::2, ::2] = 0.25
    np.testing.assert_allclose(weights[..., 0], expected)
    np.testing.assert_allclose(weights[..., 0].sum(), 1.0)


def test_twod_solve_fermi_energy_matches_free_electron_reference() -> None:
    reciprocal_vectors, eigenvalues = free_electron_case((64, 64))

    result = solve_fermi_energy(
        reciprocal_vectors,
        eigenvalues,
        electrons_per_spin=0.25,
    )

    np.testing.assert_allclose(result.fermi_energy, 0.5 * np.pi, rtol=0.0, atol=2.5e-2)
    np.testing.assert_allclose(result.weights[..., 0].sum(), 0.25, rtol=0.0, atol=2.5e-3)
    assert result.iterations > 0
