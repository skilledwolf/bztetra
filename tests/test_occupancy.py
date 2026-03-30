import numpy as np

from tetrabz import solve_fermi_energy
from tetrabz import occupation_weights
from tests.legacy_cases import brillouin_zone_volume
from tests.legacy_cases import legacy_free_electron_case


def test_occupation_weights_returns_uniform_weights_for_fully_occupied_flat_band() -> None:
    eigenvalues = np.full((2, 2, 2, 1), -1.0, dtype=np.float64)

    weights = occupation_weights(np.eye(3, dtype=np.float64), eigenvalues, method="linear")

    assert weights.shape == (2, 2, 2, 1)
    np.testing.assert_allclose(weights[..., 0], np.full((2, 2, 2), 1.0 / 8.0))
    np.testing.assert_allclose(weights[..., 0].sum(), 1.0)


def test_occupation_weights_returns_zero_weights_for_empty_flat_band() -> None:
    eigenvalues = np.full((2, 2, 2, 1), 1.0, dtype=np.float64)

    weights = occupation_weights(np.eye(3, dtype=np.float64), eigenvalues, method="linear")

    np.testing.assert_allclose(weights[..., 0], 0.0)


def test_occupation_weights_interpolates_constant_band_to_denser_output_grid() -> None:
    eigenvalues = np.full((2, 2, 2, 1), -1.0, dtype=np.float64)

    weights = occupation_weights(
        np.eye(3, dtype=np.float64),
        eigenvalues,
        weight_grid_shape=(4, 4, 4),
        method="linear",
    )

    assert weights.shape == (4, 4, 4, 1)
    expected = np.zeros((4, 4, 4), dtype=np.float64)
    expected[::2, ::2, ::2] = 1.0 / 8.0
    np.testing.assert_allclose(weights[..., 0], expected)
    np.testing.assert_allclose(weights[..., 0].sum(), 1.0)


def test_solve_fermi_energy_solves_midgap_for_two_flat_bands() -> None:
    eigenvalues = np.empty((2, 2, 2, 2), dtype=np.float64)
    eigenvalues[..., 0] = -1.0
    eigenvalues[..., 1] = 1.0

    result = solve_fermi_energy(
        np.eye(3, dtype=np.float64),
        eigenvalues,
        1.0,
        method="linear",
    )

    assert result.fermi_energy == 0.0
    assert result.iterations > 0
    np.testing.assert_allclose(result.weights[..., 0].sum(), 1.0)
    np.testing.assert_allclose(result.weights[..., 1].sum(), 0.0)


def test_occupation_weights_matches_legacy_8x8_reference_integrals() -> None:
    bvec, eigenvalues, weight_metric = legacy_free_electron_case((8, 8, 8), (8, 8, 8))

    weights = occupation_weights(bvec, eigenvalues, weight_grid_shape=(8, 8, 8), method="optimized", fermi_energy=0.5)
    weighted_integrals = (weights * weight_metric[..., None]).sum(axis=(0, 1, 2)) * brillouin_zone_volume(bvec)

    np.testing.assert_allclose(weighted_integrals, np.array([2.5028, 0.43994]), rtol=3.0e-4, atol=1.0e-5)


def test_solve_fermi_energy_matches_legacy_8x8_reference() -> None:
    bvec, eigenvalues, weight_metric = legacy_free_electron_case((8, 8, 8), (8, 8, 8))
    electrons_per_spin = (4.0 * np.pi / 3.0 + np.sqrt(2.0) * np.pi / 3.0) / brillouin_zone_volume(bvec)

    result = solve_fermi_energy(
        bvec,
        eigenvalues,
        electrons_per_spin,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    weighted_integrals = (result.weights * weight_metric[..., None]).sum(axis=(0, 1, 2)) * brillouin_zone_volume(bvec)

    np.testing.assert_allclose(result.fermi_energy, 0.50086, rtol=2.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(weighted_integrals, np.array([2.5136, 0.44385]), rtol=4.0e-4, atol=1.0e-5)
    assert result.iterations > 0
