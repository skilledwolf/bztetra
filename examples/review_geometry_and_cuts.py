from __future__ import annotations

import numpy as np

from tetrabz import build_integration_mesh
from tetrabz import fermieng
from tetrabz import occupation_weights
from tetrabz import small_tetrahedron_cut
from tetrabz import triangle_cut


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    mesh = build_integration_mesh(np.eye(3, dtype=np.float64), (2, 2, 2), method="optimized")
    print("Mesh summary")
    print("  tetrahedron_count:", mesh.tetrahedron_count)
    print("  local_point_count:", mesh.local_point_count)
    print("  first tetrahedron corner offsets:")
    print(mesh.tetrahedra_offsets[0, :4])
    print("  first tetrahedron global point indices:")
    print(mesh.global_point_indices[0, :4])
    print()

    a1_cut = small_tetrahedron_cut("a1", np.array([-2.0, 1.0, 3.0, 5.0]))
    print("Small tetrahedron A1 example")
    print("  volume_factor:", a1_cut.volume_factor)
    print("  coefficients:")
    print(a1_cut.coefficients)
    print()

    c1_triangle = triangle_cut("c1", np.array([-5.0, -3.0, -1.0, 2.0]))
    print("Triangle C1 example")
    print("  volume_factor:", c1_triangle.volume_factor)
    print("  coefficients:")
    print(c1_triangle.coefficients)
    print()

    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    eigenvalues, matrix_weights, vbz = _toy_case((4, 4, 4), reciprocal_vectors)
    occupied = occupation_weights(
        reciprocal_vectors,
        eigenvalues,
        method="optimized",
        fermi_energy=0.5,
    )
    occupation_integrals = np.sum(occupied * matrix_weights[..., None], axis=(0, 1, 2)) * vbz
    print("Occupation review on a 4x4x4 toy grid")
    print("  weighted band integrals:")
    print(occupation_integrals)
    electrons_per_spin = (4.0 * np.pi / 3.0 + np.sqrt(2.0) * np.pi / 3.0) / vbz
    fermi_energy, _, iterations = fermieng(
        reciprocal_vectors,
        eigenvalues,
        electrons_per_spin,
        method="optimized",
    )
    print("  fermieng result:")
    print(f"    fermi_energy={fermi_energy:.6f}, iterations={iterations}")

def _toy_case(grid_shape: tuple[int, int, int], reciprocal_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    eigenvalues = np.empty((*grid_shape, 2), dtype=np.float64)
    matrix_weights = np.empty(grid_shape, dtype=np.float64)
    vbz = float(np.linalg.det(reciprocal_vectors))

    for index in np.ndindex(*grid_shape):
        fractional = np.array(index, dtype=np.float64) / np.array(grid_shape, dtype=np.float64)
        fractional = fractional - np.rint(fractional)
        kvec = reciprocal_vectors @ fractional
        base = 0.5 * float(np.dot(kvec, kvec))
        eigenvalues[index + (0,)] = base
        eigenvalues[index + (1,)] = base + 0.25
        matrix_weights[index] = float(np.dot(kvec, kvec))

    return eigenvalues, matrix_weights, vbz


if __name__ == "__main__":
    main()
