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
                kvec = _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
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
                kvec = _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
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


def legacy_free_electron_response_case(
    energy_grid_shape: tuple[int, int, int],
    weight_grid_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bvec, eig1, weight_metric = legacy_free_electron_case(energy_grid_shape, weight_grid_shape)
    eig2 = make_response_eigenvalues(bvec, energy_grid_shape)
    return bvec, eig1, eig2, weight_metric


def make_response_eigenvalues(bvec: np.ndarray, grid_shape: tuple[int, int, int]) -> FloatArray:
    nx, ny, nz = grid_shape
    eigenvalues = np.empty((nx, ny, nz, 2), dtype=np.float64)

    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                kvec = bvec @ kvec
                base = 0.5 * float(np.dot(kvec, kvec))
                shifted = kvec.copy()
                shifted[0] = shifted[0] + 1.0
                eigenvalues[x_index, y_index, z_index, 0] = 0.5 * float(np.dot(shifted, shifted))
                eigenvalues[x_index, y_index, z_index, 1] = base + 0.5

    return eigenvalues


def legacy_8x8_dblstep_weighted_integrals() -> FloatArray:
    return np.array(
        [
            [0.48024, 0.12259],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )


def legacy_8x8_dbldelta_weighted_integrals() -> FloatArray:
    return np.array(
        [
            [6.0896, 3.2066],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )


def exact_dblstep_weighted_integrals() -> FloatArray:
    return np.array(
        [
            [49.0 * np.pi / 320.0, np.pi * (512.0 * np.sqrt(2.0) - 319.0) / 10240.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )


def exact_dbldelta_weighted_integrals() -> FloatArray:
    return np.array(
        [
            [2.0 * np.pi, np.pi],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )


def legacy_8x8_polstat_weighted_integrals() -> FloatArray:
    return np.array(
        [
            [3.7810, 1.0451],
            [5.0059, 1.7602],
        ],
        dtype=np.float64,
    )


def legacy_16x16_polstat_weighted_integrals() -> FloatArray:
    return np.array(
        [
            [3.8438, 1.0445],
            [5.0283, 1.7745],
        ],
        dtype=np.float64,
    )


def exact_polstat_weighted_integrals() -> FloatArray:
    return np.array(
        [
            [
                np.pi * (68.0 + 45.0 * np.log(3.0)) / 96.0,
                np.pi
                * (
                    228.0
                    + 22.0 * np.sqrt(2.0)
                    - 96.0 * np.log(2.0)
                    + 192.0 * np.log(4.0 + np.sqrt(2.0))
                    - 3.0 * np.log(1.0 + 2.0 * np.sqrt(2.0))
                )
                / 1536.0,
            ],
            [
                np.pi * 8.0 / 5.0,
                np.pi * np.sqrt(8.0) / 5.0,
            ],
        ],
        dtype=np.float64,
    )


def lindhard_q_points(count: int = 31, qmax: float = 4.0) -> FloatArray:
    return np.linspace(0.0, qmax, count, dtype=np.float64)


def exact_lindhard_curve(q_values: np.ndarray) -> FloatArray:
    q = np.asarray(q_values, dtype=np.float64)
    values = np.empty_like(q)
    q_zero = np.isclose(q, 0.0)
    q_kohn = np.isclose(q, 2.0)
    regular = ~(q_zero | q_kohn)

    values[q_zero] = 1.0
    values[q_kohn] = 0.5

    x = q[regular]
    values[regular] = 0.5 + 0.5 / x * (1.0 - 0.25 * x * x) * np.log(np.abs((x + 2.0) / (x - 2.0)))
    return values


def lindhard_free_electron_case(
    grid_shape: tuple[int, int, int],
    q_value: float,
    *,
    fermi_energy: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bvec = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    eig1 = np.empty((*grid_shape, 1), dtype=np.float64)
    eig2 = np.empty((*grid_shape, 1), dtype=np.float64)
    qvec = np.array([q_value, 0.0, 0.0], dtype=np.float64)

    nx, ny, nz = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = bvec @ _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                eig1[x_index, y_index, z_index, 0] = 0.5 * float(np.dot(kvec, kvec)) - fermi_energy
                shifted = kvec + qvec
                eig2[x_index, y_index, z_index, 0] = 0.5 * float(np.dot(shifted, shifted)) - fermi_energy

    return bvec, eig1, eig2


def _centered_fractional_kpoint(
    indices: tuple[int, int, int],
    grid_shape: tuple[int, int, int],
) -> FloatArray:
    grid = np.asarray(grid_shape, dtype=np.int64)
    half_grid = grid // 2
    integer_indices = np.asarray(indices, dtype=np.int64)
    centered = np.mod(integer_indices + half_grid, grid) - half_grid
    return centered.astype(np.float64) / grid.astype(np.float64)


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
