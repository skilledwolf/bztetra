from pathlib import Path

import numpy as np

from tetrabz import dbldelta
from tetrabz import dblstep
from tetrabz import dos
from tetrabz import polstat
from tests.legacy_cases import brillouin_zone_volume
from tests.legacy_cases import exact_dbldelta_weighted_integrals
from tests.legacy_cases import exact_dblstep_weighted_integrals
from tests.legacy_cases import exact_polstat_weighted_integrals
from tests.legacy_cases import legacy_8x8_dbldelta_weighted_integrals
from tests.legacy_cases import legacy_8x8_dblstep_weighted_integrals
from tests.legacy_cases import legacy_8x8_polstat_weighted_integrals
from tests.legacy_cases import legacy_16x16_polstat_weighted_integrals
from tests.legacy_cases import legacy_free_electron_response_case
from tests.legacy_cases import lindhard_free_electron_case
from tests.legacy_cases import lindhard_q_points


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


def test_polstat_matches_legacy_8x8_response_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    weights = polstat(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    weighted = _weighted_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_8x8_polstat_weighted_integrals(), rtol=2.5e-3, atol=1.0e-5)


def test_polstat_matches_legacy_16x16_response_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (16, 16, 16))

    weights = polstat(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(16, 16, 16),
        method="optimized",
    )
    weighted = _weighted_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_16x16_polstat_weighted_integrals(), rtol=2.0e-4, atol=1.0e-5)


def test_polstat_tracks_exact_integrals_on_16_grid() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (16, 16, 16))

    weights = polstat(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(16, 16, 16),
        method="optimized",
    )
    weighted = _weighted_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, exact_polstat_weighted_integrals(), rtol=5.0e-3, atol=1.0e-5)


def test_lindhard_8x8_curves_match_legacy_examples() -> None:
    q_values = lindhard_q_points()
    legacy_dir = Path(__file__).resolve().parents[1] / "libtetra_original" / "example"

    linear_curve = _lindhard_curve(q_values, method="linear")
    optimized_curve = _lindhard_curve(q_values, method="optimized")

    legacy_linear = np.loadtxt(legacy_dir / "lindhard1_8.dat", dtype=np.float64)
    legacy_optimized = np.loadtxt(legacy_dir / "lindhard2_8.dat", dtype=np.float64)

    np.testing.assert_allclose(legacy_linear[:, 0], q_values, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(legacy_optimized[:, 0], q_values, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(linear_curve, legacy_linear[:, 1], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(optimized_curve, legacy_optimized[:, 1], rtol=1.0e-12, atol=1.0e-12)


def _weighted_matrix(weights: np.ndarray, metric: np.ndarray, reciprocal_vectors: np.ndarray) -> np.ndarray:
    return (weights * metric[..., None, None]).sum(axis=(0, 1, 2)) * brillouin_zone_volume(reciprocal_vectors)


def _lindhard_curve(q_values: np.ndarray, *, method: str) -> np.ndarray:
    values = np.empty(q_values.size, dtype=np.float64)

    for index, q_value in enumerate(q_values):
        bvec, eig1, eig2 = lindhard_free_electron_case((8, 8, 8), float(q_value))
        if index == 0:
            weights = dos(
                bvec,
                eig1,
                np.array([0.0], dtype=np.float64),
                weight_grid_shape=(8, 8, 8),
                method=method,
            )
            values[index] = weights.sum() * brillouin_zone_volume(bvec) / (4.0 * np.pi)
        else:
            weights = polstat(
                bvec,
                eig1,
                eig2,
                weight_grid_shape=(8, 8, 8),
                method=method,
            )
            values[index] = 2.0 * weights.sum() * brillouin_zone_volume(bvec) / (4.0 * np.pi)

    return values
