from __future__ import annotations

import math

import numpy as np

from bztetra import density_of_states_weights
from bztetra import integrated_density_of_states_weights


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    bvec = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    eigenvalues, metric = _legacy_free_electron_case((8, 8, 8))
    sample_energies = 0.5 * np.square(np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64))

    dos_weights = density_of_states_weights(bvec, eigenvalues, sample_energies, weight_grid_shape=(8, 8, 8), method="optimized")
    intdos_weights = integrated_density_of_states_weights(bvec, eigenvalues, sample_energies, weight_grid_shape=(8, 8, 8), method="optimized")

    dos_moments = (dos_weights * metric[None, ..., None]).sum(axis=(1, 2, 3)) * np.linalg.det(bvec)
    intdos_moments = (intdos_weights * metric[None, ..., None]).sum(axis=(1, 2, 3)) * np.linalg.det(bvec)
    exact_dos = _exact_dos_weighted_integrals(sample_energies)
    exact_intdos = _exact_intdos_weighted_integrals(sample_energies)

    print("DOS review (legacy 8x8 free-electron fixture)")
    _print_review_table(sample_energies, dos_moments, exact_dos)

    print()
    print("Integrated DOS review (legacy 8x8 free-electron fixture)")
    _print_review_table(sample_energies, intdos_moments, exact_intdos)


def _legacy_free_electron_case(grid_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = grid_shape
    bvec = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    eigenvalues = np.empty((nx, ny, nz, 2), dtype=np.float64)
    metric = np.empty((nx, ny, nz), dtype=np.float64)

    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = np.array([x_index / nx, y_index / ny, z_index / nz], dtype=np.float64)
                kvec = kvec - np.rint(kvec)
                kvec = bvec @ kvec
                band_0 = 0.5 * float(np.dot(kvec, kvec))
                eigenvalues[x_index, y_index, z_index, 0] = band_0
                eigenvalues[x_index, y_index, z_index, 1] = band_0 + 0.25
                metric[x_index, y_index, z_index] = float(np.dot(kvec, kvec))

    return eigenvalues, metric


def _exact_dos_weighted_integrals(energies: np.ndarray) -> np.ndarray:
    radii = np.sqrt(2.0 * energies)
    expected = np.zeros((energies.size, 2), dtype=np.float64)
    expected[:, 0] = 4.0 * math.pi * np.power(radii, 3)
    active = radii > 1.0 / math.sqrt(2.0)
    expected[active, 1] = math.sqrt(2.0) * math.pi * np.power(2.0 * radii[active] ** 2 - 1.0, 1.5)
    return expected


def _exact_intdos_weighted_integrals(energies: np.ndarray) -> np.ndarray:
    radii = np.sqrt(2.0 * energies)
    expected = np.zeros((energies.size, 2), dtype=np.float64)
    expected[:, 0] = 4.0 * math.pi * np.power(radii, 5) / 5.0
    active = radii > 1.0 / math.sqrt(2.0)
    expected[active, 1] = math.pi * np.power(2.0 * radii[active] ** 2 - 1.0, 2.5) / (5.0 * math.sqrt(2.0))
    return expected


def _print_review_table(energies: np.ndarray, values: np.ndarray, exact_values: np.ndarray) -> None:
    for energy, current, exact in zip(energies, values, exact_values, strict=True):
        relative_error = np.abs(current - exact) / np.maximum(np.abs(exact), 1.0e-12)
        print(
            f"  E={energy:.6f} -> current={current} exact={exact} relerr={relative_error}"
        )


if __name__ == "__main__":
    main()
