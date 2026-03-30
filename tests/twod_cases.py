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


def phase_space_overlap_full_triangle_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupied = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    target = np.array([-2.0, -2.0, -2.0], dtype=np.float64)
    expected = np.full(3, 1.0 / 6.0, dtype=np.float64)
    return occupied, target, expected


def phase_space_overlap_empty_triangle_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupied = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    target = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    expected = np.zeros(3, dtype=np.float64)
    return occupied, target, expected


def phase_space_overlap_equal_triangle_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupied = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    target = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    expected = np.full(3, 1.0 / 12.0, dtype=np.float64)
    return occupied, target, expected


def nesting_single_triangle_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.array([-1.0, 1.0, 0.0], dtype=np.float64)
    target = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    expected = np.full(3, 1.0 / 9.0, dtype=np.float64)
    return source, target, expected


def static_polarization_single_triangle_case() -> tuple[np.ndarray, np.ndarray]:
    transfer_energies = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    expected = 0.5 * np.array(
        [
            np.log(2.0),
            2.0 * np.log(2.0) - 1.0,
            1.0 - np.log(2.0),
        ],
        dtype=np.float64,
    )
    return transfer_energies, expected


def fermi_golden_rule_single_triangle_case() -> tuple[np.ndarray, float, np.ndarray]:
    transfer_energies = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    omega = 1.0
    expected = np.array([1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0], dtype=np.float64)
    return transfer_energies, omega, expected


def fermi_golden_rule_zero_case() -> tuple[np.ndarray, float, np.ndarray]:
    transfer_energies = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    omega = 10.0
    expected = np.zeros(3, dtype=np.float64)
    return transfer_energies, omega, expected


def synthetic_multiband_response_case(
    grid_shape: tuple[int, int] = (8, 8),
    *,
    band_count: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal_vectors = np.diag([3.0, 3.0]).astype(np.float64)
    occupied = np.empty((*grid_shape, band_count), dtype=np.float64)
    target = np.empty((*grid_shape, band_count), dtype=np.float64)

    nx, ny = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            kfrac = np.array(
                [x_index / nx, y_index / ny],
                dtype=np.float64,
            ) - 0.5
            kcart = reciprocal_vectors @ kfrac
            base = 0.5 * np.dot(kcart, kcart) - 0.5
            shifted = kcart.copy()
            shifted[0] = shifted[0] + 0.85
            shifted_base = 0.5 * np.dot(shifted, shifted) - 0.5
            for band_index in range(band_count):
                occupied[x_index, y_index, band_index] = base + 0.07 * band_index
                target[x_index, y_index, band_index] = (
                    shifted_base
                    + 0.11 * band_index
                    + 0.03 * (band_index + 1) * kcart[1]
                )

    return reciprocal_vectors, occupied, target
