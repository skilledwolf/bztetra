from __future__ import annotations

import numpy as np


def free_electron_case(grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    reciprocal_vectors = 2.0 * np.pi * np.eye(2, dtype=np.float64)
    nx, ny = grid_shape
    eigenvalues = np.empty((nx, ny, 1), dtype=np.float64)

    for x_index in range(nx):
        for y_index in range(ny):
            kfrac = np.array(
                [x_index / nx, y_index / ny],
                dtype=np.float64,
            ) - 0.5
            kcart = reciprocal_vectors @ kfrac
            eigenvalues[x_index, y_index, 0] = 0.5 * np.dot(kcart, kcart)

    return reciprocal_vectors, eigenvalues


def free_electron_energy_points() -> np.ndarray:
    return np.array([0.25, 0.75, 1.25, 2.0, 3.0, 4.0], dtype=np.float64)


def exact_free_electron_dos_normalized(energies: np.ndarray) -> np.ndarray:
    values = np.zeros_like(energies, dtype=np.float64)
    mask = (energies > 0.0) & (energies < 0.5 * np.pi * np.pi)
    values[mask] = 1.0 / (2.0 * np.pi)
    return values


def exact_free_electron_intdos_normalized(energies: np.ndarray) -> np.ndarray:
    values = np.zeros_like(energies, dtype=np.float64)
    mask = (energies > 0.0) & (energies < 0.5 * np.pi * np.pi)
    values[mask] = energies[mask] / (2.0 * np.pi)
    return values
