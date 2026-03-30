import numpy as np

from tetrabz import dbldelta
from tetrabz import dblstep
from tests.legacy_cases import brillouin_zone_volume
from tests.legacy_cases import exact_dbldelta_weighted_integrals
from tests.legacy_cases import exact_dblstep_weighted_integrals
from tests.legacy_cases import legacy_8x8_dbldelta_weighted_integrals
from tests.legacy_cases import legacy_8x8_dblstep_weighted_integrals
from tests.legacy_cases import legacy_free_electron_response_case


def test_dblstep_matches_legacy_8x8_response_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    weights = dblstep(bvec, eigenvalues_1 - 0.5, eigenvalues_2 - 0.5, weight_grid_shape=(8, 8, 8), method="optimized")
    weighted = _weighted_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_8x8_dblstep_weighted_integrals(), rtol=8.0e-4, atol=1.0e-6)


def test_dbldelta_matches_legacy_8x8_response_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    weights = dbldelta(bvec, eigenvalues_1 - 0.5, eigenvalues_2 - 0.5, weight_grid_shape=(8, 8, 8), method="optimized")
    weighted = _weighted_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_8x8_dbldelta_weighted_integrals(), rtol=2.0e-3, atol=1.0e-5)


def test_dblstep_tracks_exact_integrals_on_16_grid() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (16, 16, 16))

    weights = dblstep(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(16, 16, 16),
        method="optimized",
    )
    weighted = _weighted_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, exact_dblstep_weighted_integrals(), rtol=2.0e-3, atol=1.0e-6)


def test_dbldelta_tracks_exact_integrals_on_16_grid() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (16, 16, 16))

    weights = dbldelta(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(16, 16, 16),
        method="optimized",
    )
    weighted = _weighted_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, exact_dbldelta_weighted_integrals(), rtol=5.0e-3, atol=1.0e-5)


def _weighted_matrix(weights: np.ndarray, metric: np.ndarray, reciprocal_vectors: np.ndarray) -> np.ndarray:
    return (weights * metric[..., None, None]).sum(axis=(0, 1, 2)) * brillouin_zone_volume(reciprocal_vectors)
