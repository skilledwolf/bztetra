from __future__ import annotations

import math

import numpy as np

from bztetra import solve_fermi_energy
from bztetra import occupation_weights


def build_review_system(grid_size: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    eigenvalues = np.empty((grid_size, grid_size, grid_size, 2), dtype=np.float64)
    radial_metric = np.empty((grid_size, grid_size, grid_size), dtype=np.float64)

    for x_index in range(grid_size):
        for y_index in range(grid_size):
            for z_index in range(grid_size):
                kvec = np.array(
                    [x_index / grid_size, y_index / grid_size, z_index / grid_size],
                    dtype=np.float64,
                )
                kvec = kvec - np.rint(kvec)
                kvec = reciprocal_vectors @ kvec
                base = 0.5 * float(np.dot(kvec, kvec))
                eigenvalues[x_index, y_index, z_index, 0] = base
                eigenvalues[x_index, y_index, z_index, 1] = base + 0.25
                radial_metric[x_index, y_index, z_index] = float(np.dot(kvec, kvec))

    return reciprocal_vectors, eigenvalues, radial_metric


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    reciprocal_vectors, eigenvalues, radial_metric = build_review_system()
    bz_volume = float(np.linalg.det(reciprocal_vectors))
    electrons_per_spin = (4.0 * math.pi / 3.0 + math.sqrt(2.0) * math.pi / 3.0) / bz_volume

    occupation = occupation_weights(
        reciprocal_vectors,
        eigenvalues,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
        fermi_energy=0.5,
    )
    occupation_integrals = np.sum(occupation * radial_metric[..., None], axis=(0, 1, 2)) * bz_volume

    result = solve_fermi_energy(
        reciprocal_vectors,
        eigenvalues,
        electrons_per_spin,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    fermi_integrals = np.sum(result.weights * radial_metric[..., None], axis=(0, 1, 2)) * bz_volume

    print("Occupation review")
    print("  integrated weights:", occupation_integrals)
    print()
    print("Fermi search review")
    print("  fermi_energy:", result.fermi_energy)
    print("  iterations:", result.iterations)
    print("  integrated weights:", fermi_integrals)


if __name__ == "__main__":
    main()
