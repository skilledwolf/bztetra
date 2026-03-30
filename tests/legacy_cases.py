from __future__ import annotations

from pathlib import Path

import numpy as np


FloatArray = np.ndarray
LEGACY_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "libtetra_original" / "example"


def legacy_free_electron_case(
    energy_grid_shape: tuple[int, int, int],
    weight_grid_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bvec = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    eigenvalues = make_eigenvalues(bvec, energy_grid_shape)
    weight_metric = make_weight_metric(bvec, weight_grid_shape)
    return bvec, eigenvalues, weight_metric


def make_eigenvalues(bvec: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
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


def make_weight_metric(bvec: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
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


def brillouin_zone_volume(bvec: np.ndarray) -> float:
    return float(np.linalg.det(bvec))


def legacy_dos_energy_points() -> np.ndarray:
    x = 0.2 * np.arange(1, 6, dtype=np.float64)
    return 0.5 * x * x


def legacy_8x8_dos_weighted_integrals() -> FloatArray:
    return np.array(
        [
            [0.079273, 0.0],
            [0.85871, 0.0],
            [2.6242, 0.0],
            [6.5716, 0.70796],
            [12.5, 4.5276],
        ],
        dtype=np.float64,
    )


def legacy_8x8_intdos_weighted_integrals() -> FloatArray:
    return np.array(
        [
            [0.00047294, 0.0],
            [0.026509, 0.0],
            [0.19294, 0.0],
            [0.83124, 0.018675],
            [2.5028, 0.43994],
        ],
        dtype=np.float64,
    )


def exact_free_electron_dos_weighted_integrals(energies: np.ndarray) -> FloatArray:
    radii = np.sqrt(2.0 * np.asarray(energies, dtype=np.float64))
    expected = np.zeros((radii.size, 2), dtype=np.float64)
    expected[:, 0] = 4.0 * np.pi * np.power(radii, 3)

    active = radii > 1.0 / np.sqrt(2.0)
    expected[active, 1] = np.sqrt(2.0) * np.pi * np.power(2.0 * radii[active] ** 2 - 1.0, 1.5)
    return expected


def exact_free_electron_intdos_weighted_integrals(energies: np.ndarray) -> FloatArray:
    radii = np.sqrt(2.0 * np.asarray(energies, dtype=np.float64))
    expected = np.zeros((radii.size, 2), dtype=np.float64)
    expected[:, 0] = 4.0 * np.pi * np.power(radii, 5) / 5.0

    active = radii > 1.0 / np.sqrt(2.0)
    expected[active, 1] = np.pi * np.power(2.0 * radii[active] ** 2 - 1.0, 2.5) / (5.0 * np.sqrt(2.0))
    return expected


def tight_binding_dos_energy_points() -> FloatArray:
    return np.linspace(-3.0, 3.0, 100, dtype=np.float64)


def cubic_tight_binding_band(grid_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = grid_shape
    reciprocal_vectors = np.eye(3, dtype=np.float64)
    eigenvalues = np.empty((nx, ny, nz, 1), dtype=np.float64)

    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = 2.0 * np.pi * (
                    np.array((x_index, y_index, z_index), dtype=np.float64) - 0.5 * np.array(grid_shape, dtype=np.float64)
                ) / np.array(grid_shape, dtype=np.float64)
                eigenvalues[x_index, y_index, z_index, 0] = -np.cos(kvec).sum()

    return reciprocal_vectors, eigenvalues


def load_legacy_example_dataset(filename: str) -> FloatArray:
    return np.loadtxt(LEGACY_EXAMPLE_DIR / filename, dtype=np.float64)
