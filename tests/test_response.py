from pathlib import Path

import numpy as np

from tetrabz import dbldelta
from tetrabz import dblstep
from tetrabz import dos
from tetrabz import polstat
from tetrabz._grids import interpolate_local_values
from tetrabz._grids import interpolated_tetrahedron_energies
from tetrabz._grids import normalize_eigenvalues
from tetrabz.geometry import build_integration_mesh
from tetrabz.response import _accumulate_small_tetra_polstat_outer
from tetrabz.response import _polstat_secondary_weights
from tetrabz.response import _unflatten_pair_band_last
from tests.legacy_cases import brillouin_zone_volume
from tests.legacy_cases import exact_dbldelta_weighted_integrals
from tests.legacy_cases import exact_dblstep_weighted_integrals
from tests.legacy_cases import exact_lindhard_curve
from tests.legacy_cases import exact_polstat_weighted_integrals
from tests.legacy_cases import legacy_16x8_polstat_weighted_integrals
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


def test_polstat_matches_legacy_16x8_interpolation_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (8, 8, 8))

    weights = polstat(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    weighted = _weighted_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_16x8_polstat_weighted_integrals(), rtol=3.0e-4, atol=1.0e-5)


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


def test_polstat_matches_python_reference_weights_on_small_grid() -> None:
    bvec, eigenvalues_1, eigenvalues_2, _ = legacy_free_electron_response_case((4, 4, 4), (4, 4, 4))

    weights = polstat(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(4, 4, 4),
        method="optimized",
    )
    reference = _python_polstat_weights(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(4, 4, 4),
        method="optimized",
    )

    np.testing.assert_allclose(weights, reference, rtol=1.0e-12, atol=1.0e-12)


def test_lindhard_small_q_and_kohn_point_track_exact_curve() -> None:
    q_values = np.array([1.0e-3, 2.0], dtype=np.float64)
    values = np.empty(q_values.size, dtype=np.float64)

    for index, q_value in enumerate(q_values):
        bvec, eig1, eig2 = lindhard_free_electron_case((16, 16, 16), float(q_value))
        weights = polstat(
            bvec,
            eig1,
            eig2,
            weight_grid_shape=(16, 16, 16),
            method="optimized",
        )
        values[index] = 2.0 * weights.sum() * brillouin_zone_volume(bvec) / (4.0 * np.pi)

    np.testing.assert_allclose(values, exact_lindhard_curve(q_values), rtol=3.0e-3, atol=3.0e-3)


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


def _python_polstat_weights(
    reciprocal_vectors: np.ndarray,
    occupied_eigenvalues: np.ndarray,
    target_eigenvalues: np.ndarray,
    *,
    weight_grid_shape: tuple[int, int, int],
    method: str,
) -> np.ndarray:
    occupied_flat, energy_grid_shape = normalize_eigenvalues(occupied_eigenvalues)
    target_flat, _ = normalize_eigenvalues(target_eigenvalues)
    mesh = build_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    occupied_tetra = interpolated_tetrahedron_energies(mesh, occupied_flat)
    target_tetra = interpolated_tetrahedron_energies(mesh, target_flat)

    occupied_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    local_weights = np.zeros((mesh.local_point_count, target_band_count, occupied_band_count), dtype=np.float64)

    for tetrahedron_index in range(mesh.tetrahedron_count):
        local_points = mesh.local_point_indices[tetrahedron_index]
        for occupied_band_index in range(occupied_band_count):
            outer_weights = np.zeros((target_band_count, 4), dtype=np.float64)
            sorted_order = np.argsort(occupied_tetra[tetrahedron_index, :, occupied_band_index])
            sorted_occupied = occupied_tetra[tetrahedron_index, sorted_order, occupied_band_index]
            sorted_target = target_tetra[tetrahedron_index, sorted_order, :]

            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_polstat_outer(outer_weights, "a1", sorted_order, sorted_occupied, sorted_target)
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for kind in ("b1", "b2", "b3"):
                    _accumulate_small_tetra_polstat_outer(
                        outer_weights,
                        kind,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for kind in ("c1", "c2", "c3"):
                    _accumulate_small_tetra_polstat_outer(
                        outer_weights,
                        kind,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                    )
            elif sorted_occupied[3] <= 0.0:
                outer_weights += _polstat_secondary_weights(
                    occupied_tetra[tetrahedron_index, :, occupied_band_index],
                    target_tetra[tetrahedron_index],
                )

            point_weights = outer_weights @ mesh.tetrahedron_weight_matrix
            np.add.at(local_weights[:, :, occupied_band_index], local_points, point_weights.T)

    local_weights /= float(6 * np.prod(mesh.energy_grid_shape, dtype=np.int64))
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_pair_band_last(output_flat, mesh.weight_grid_shape)


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
