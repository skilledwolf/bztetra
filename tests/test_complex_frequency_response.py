import numpy as np

from tetrabz import polcmplx
from tests.legacy_cases import brillouin_zone_volume
from tests.legacy_cases import exact_polcmplx_weighted_integrals
from tests.legacy_cases import legacy_16x8_polcmplx_weighted_integrals
from tests.legacy_cases import legacy_8x8_polcmplx_weighted_integrals
from tests.legacy_cases import legacy_free_electron_response_case
from tests.legacy_cases import polcmplx_energy_points


def test_polcmplx_exposes_energy_first_pair_band_last_layout() -> None:
    bvec, eigenvalues_1, eigenvalues_2, _ = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    weights = polcmplx(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        polcmplx_energy_points(),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )

    assert weights.shape == (3, 8, 8, 8, 2, 2)
    assert weights.dtype == np.complex128


def test_polcmplx_matches_legacy_8x8_response_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    weights = polcmplx(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        polcmplx_energy_points(),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    weighted = _weighted_energy_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_8x8_polcmplx_weighted_integrals(), rtol=3.0e-3, atol=1.0e-5)


def test_polcmplx_matches_legacy_16x8_interpolation_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (8, 8, 8))

    weights = polcmplx(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        polcmplx_energy_points(),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    weighted = _weighted_energy_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_16x8_polcmplx_weighted_integrals(), rtol=2.0e-4, atol=1.0e-5)


def test_polcmplx_tracks_exact_integrals_on_16_grid() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (16, 16, 16))

    weights = polcmplx(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        polcmplx_energy_points(),
        weight_grid_shape=(16, 16, 16),
        method="optimized",
    )
    weighted = _weighted_energy_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, exact_polcmplx_weighted_integrals(), rtol=6.0e-3, atol=1.0e-5)


def _weighted_energy_matrix(weights: np.ndarray, metric: np.ndarray, reciprocal_vectors: np.ndarray) -> np.ndarray:
    weighted = (weights * metric[None, ..., None, None]).sum(axis=(1, 2, 3))
    return weighted * brillouin_zone_volume(reciprocal_vectors)
