import numpy as np

from tetrabz import complex_frequency_polarization_weights
from tests.legacy_cases import brillouin_zone_volume
from tests.legacy_cases import exact_complex_frequency_polarization_constant_gap_channels
from tests.legacy_cases import exact_complex_frequency_polarization_weighted_integrals
from tests.legacy_cases import legacy_16x8_complex_frequency_polarization_weighted_integrals
from tests.legacy_cases import legacy_8x8_complex_frequency_polarization_weighted_integrals
from tests.legacy_cases import legacy_free_electron_response_case
from tests.legacy_cases import complex_frequency_polarization_energy_points


def test_polcmplx_exposes_energy_first_pair_band_last_layout() -> None:
    bvec, eigenvalues_1, eigenvalues_2, _ = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    weights = complex_frequency_polarization_weights(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        complex_frequency_polarization_energy_points(),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )

    assert weights.shape == (3, 8, 8, 8, 2, 2)
    assert weights.dtype == np.complex128


def test_polcmplx_matches_legacy_8x8_response_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    weights = complex_frequency_polarization_weights(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        complex_frequency_polarization_energy_points(),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    weighted = _weighted_energy_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_8x8_complex_frequency_polarization_weighted_integrals(), rtol=5.0e-4, atol=1.0e-5)


def test_polcmplx_matches_legacy_16x8_interpolation_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (8, 8, 8))

    weights = complex_frequency_polarization_weights(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        complex_frequency_polarization_energy_points(),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    weighted = _weighted_energy_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_16x8_complex_frequency_polarization_weighted_integrals(), rtol=3.0e-4, atol=1.0e-5)


def test_polcmplx_tracks_exact_anchor_integrals_on_16_grid() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (16, 16, 16))

    weights = complex_frequency_polarization_weights(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        complex_frequency_polarization_energy_points(),
        weight_grid_shape=(16, 16, 16),
        method="optimized",
    )
    weighted = _weighted_energy_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, exact_complex_frequency_polarization_weighted_integrals(), rtol=6.0e-3, atol=1.0e-5)


def test_polcmplx_tracks_exact_constant_gap_channels_on_matsubara_axis() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (16, 16, 16))
    energies = 1j * np.linspace(0.25, 2.5, 9, dtype=np.float64)

    weights = complex_frequency_polarization_weights(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        energies,
        weight_grid_shape=(16, 16, 16),
        method="optimized",
    )
    weighted = _weighted_energy_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted[:, 1, :], exact_complex_frequency_polarization_constant_gap_channels(energies), rtol=2.0e-3, atol=1.0e-5)


def _weighted_energy_matrix(weights: np.ndarray, metric: np.ndarray, reciprocal_vectors: np.ndarray) -> np.ndarray:
    return (weights * metric[None, ..., None, None]).sum(axis=(1, 2, 3)) * brillouin_zone_volume(reciprocal_vectors)
