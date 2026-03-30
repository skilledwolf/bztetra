import numpy as np

from tetrabz import fermieng
from tetrabz import occ


def test_occ_returns_uniform_weights_for_fully_occupied_flat_band() -> None:
    eigenvalues = np.full((2, 2, 2, 1), -1.0, dtype=np.float64)

    weights = occ(np.eye(3, dtype=np.float64), eigenvalues, method="linear")

    assert weights.shape == (2, 2, 2, 1)
    np.testing.assert_allclose(weights[..., 0], np.full((2, 2, 2), 1.0 / 8.0))
    np.testing.assert_allclose(weights[..., 0].sum(), 1.0)


def test_occ_returns_zero_weights_for_empty_flat_band() -> None:
    eigenvalues = np.full((2, 2, 2, 1), 1.0, dtype=np.float64)

    weights = occ(np.eye(3, dtype=np.float64), eigenvalues, method="linear")

    np.testing.assert_allclose(weights[..., 0], 0.0)


def test_occ_interpolates_constant_band_to_denser_output_grid() -> None:
    eigenvalues = np.full((2, 2, 2, 1), -1.0, dtype=np.float64)

    weights = occ(
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


def test_fermieng_solves_midgap_for_two_flat_bands() -> None:
    eigenvalues = np.empty((2, 2, 2, 2), dtype=np.float64)
    eigenvalues[..., 0] = -1.0
    eigenvalues[..., 1] = 1.0

    fermi_energy, weights, iterations = fermieng(
        np.eye(3, dtype=np.float64),
        eigenvalues,
        1.0,
        method="linear",
    )

    assert fermi_energy == 0.0
    assert iterations > 0
    np.testing.assert_allclose(weights[..., 0].sum(), 1.0)
    np.testing.assert_allclose(weights[..., 1].sum(), 0.0)


def test_occ_matches_legacy_8x8_reference_integrals() -> None:
    bvec, eigenvalues, weight_metric = _legacy_free_electron_case((8, 8, 8), (8, 8, 8))

    weights = occ(bvec, eigenvalues, weight_grid_shape=(8, 8, 8), method="optimized", fermi_energy=0.5)
    weighted_integrals = (weights * weight_metric[..., None]).sum(axis=(0, 1, 2)) * _brillouin_zone_volume(bvec)

    np.testing.assert_allclose(weighted_integrals, np.array([2.5028, 0.43994]), rtol=3.0e-4, atol=1.0e-5)


def test_fermieng_matches_legacy_8x8_reference() -> None:
    bvec, eigenvalues, weight_metric = _legacy_free_electron_case((8, 8, 8), (8, 8, 8))
    electrons_per_spin = (4.0 * np.pi / 3.0 + np.sqrt(2.0) * np.pi / 3.0) / _brillouin_zone_volume(bvec)

    fermi_energy, weights, iterations = fermieng(
        bvec,
        eigenvalues,
        electrons_per_spin,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    weighted_integrals = (weights * weight_metric[..., None]).sum(axis=(0, 1, 2)) * _brillouin_zone_volume(bvec)

    np.testing.assert_allclose(fermi_energy, 0.50086, rtol=2.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(weighted_integrals, np.array([2.5136, 0.44385]), rtol=4.0e-4, atol=1.0e-5)
    assert iterations > 0


def _legacy_free_electron_case(
    energy_grid_shape: tuple[int, int, int],
    weight_grid_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bvec = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    eigenvalues = _make_eigenvalues(bvec, energy_grid_shape)
    weight_metric = _make_weight_metric(bvec, weight_grid_shape)
    return bvec, eigenvalues, weight_metric


def _make_eigenvalues(bvec: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = grid_shape
    eigenvalues = np.empty((nx, ny, nz, 2), dtype=np.float64)

    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = np.array(
                    [x_index / nx, y_index / ny, z_index / nz],
                    dtype=np.float64,
                )
                kvec = kvec - np.rint(kvec)
                kvec = bvec @ kvec
                band_0 = 0.5 * float(np.dot(kvec, kvec))
                eigenvalues[x_index, y_index, z_index, 0] = band_0
                eigenvalues[x_index, y_index, z_index, 1] = band_0 + 0.25

    return eigenvalues


def _make_weight_metric(bvec: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = grid_shape
    metric = np.empty((nx, ny, nz), dtype=np.float64)

    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = np.array(
                    [x_index / nx, y_index / ny, z_index / nz],
                    dtype=np.float64,
                )
                kvec = kvec - np.rint(kvec)
                kvec = bvec @ kvec
                metric[x_index, y_index, z_index] = float(np.dot(kvec, kvec))

    return metric


def _brillouin_zone_volume(bvec: np.ndarray) -> float:
    return float(np.linalg.det(bvec))
