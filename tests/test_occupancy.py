import math

import numpy as np
import pytest

from tetrabz import occupation_weights
from tetrabz import solve_fermi_energy


def test_occupation_weights_match_legacy_shell_reference_on_8_grid() -> None:
    reciprocal_vectors, eigenvalues, _, matrix_weights, vbz = _toy_case((8, 8, 8), (8, 8, 8))

    weights = occupation_weights(
        reciprocal_vectors,
        eigenvalues,
        method="optimized",
        fermi_energy=0.5,
    )

    assert weights.shape == (8, 8, 8, 2)
    values = np.sum(weights * matrix_weights[..., None], axis=(0, 1, 2)) * vbz
    np.testing.assert_allclose(values, np.array([2.5028, 0.43994]), rtol=0.0, atol=5.0e-4)


def test_solve_fermi_energy_matches_legacy_shell_reference_on_8_grid() -> None:
    reciprocal_vectors, eigenvalues, _, matrix_weights, vbz = _toy_case((8, 8, 8), (8, 8, 8))
    nelec = (4.0 * math.pi / 3.0 + math.sqrt(2.0) * math.pi / 3.0) / vbz

    fermi_energy, weights = solve_fermi_energy(
        reciprocal_vectors,
        eigenvalues,
        nelec,
        method="optimized",
    )

    assert weights.shape == (8, 8, 8, 2)
    assert fermi_energy == pytest.approx(0.50086, abs=5.0e-4)
    values = np.sum(weights * matrix_weights[..., None], axis=(0, 1, 2)) * vbz
    np.testing.assert_allclose(values, np.array([2.5136, 0.44385]), rtol=0.0, atol=7.0e-4)


def test_interpolated_occupation_weights_preserve_total_weight_sum() -> None:
    reciprocal_vectors, eigenvalues, _, _, _ = _toy_case((4, 4, 4), (4, 4, 4))

    direct = occupation_weights(
        reciprocal_vectors,
        eigenvalues,
        method="optimized",
        fermi_energy=0.5,
    )
    interpolated = occupation_weights(
        reciprocal_vectors,
        eigenvalues,
        weight_grid_shape=(2, 2, 2),
        method="optimized",
        fermi_energy=0.5,
    )

    assert interpolated.shape == (2, 2, 2, 2)
    assert np.sum(interpolated) == pytest.approx(np.sum(direct))


def _toy_case(
    energy_grid_shape: tuple[int, int, int],
    weight_grid_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    vbz = float(np.linalg.det(reciprocal_vectors))

    eigenvalues = np.empty((*energy_grid_shape, 2), dtype=np.float64)
    eig2 = np.empty((*energy_grid_shape, 2), dtype=np.float64)
    for index in np.ndindex(*energy_grid_shape):
        kvec = _wrapped_kvec(index, energy_grid_shape, reciprocal_vectors)
        base = 0.5 * float(np.dot(kvec, kvec))
        eigenvalues[index + (0,)] = base
        eigenvalues[index + (1,)] = base + 0.25

        shifted = kvec.copy()
        shifted[0] += 1.0
        eig2[index + (0,)] = 0.5 * float(np.dot(shifted, shifted))
        eig2[index + (1,)] = base + 0.5

    matrix_weights = np.empty(weight_grid_shape, dtype=np.float64)
    for index in np.ndindex(*weight_grid_shape):
        kvec = _wrapped_kvec(index, weight_grid_shape, reciprocal_vectors)
        matrix_weights[index] = float(np.dot(kvec, kvec))

    return reciprocal_vectors, eigenvalues, eig2, matrix_weights, vbz


def _wrapped_kvec(
    index: tuple[int, int, int],
    grid_shape: tuple[int, int, int],
    reciprocal_vectors: np.ndarray,
) -> np.ndarray:
    fractional = np.array(index, dtype=np.float64) / np.array(grid_shape, dtype=np.float64)
    fractional = fractional - np.rint(fractional)
    return reciprocal_vectors @ fractional
