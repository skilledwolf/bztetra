from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit

from ._cut_kernels import accumulate_small_tetra_weight_sums
from ._cut_kernels import sort4_shifted
from ._grids import FloatArray
from ._grids import interpolate_local_values
from ._grids import interpolated_tetrahedron_energies
from ._grids import normalize_eigenvalues
from .geometry import IntegrationMesh
from .geometry import TetraMethod
from .geometry import build_integration_mesh


@dataclass(frozen=True, slots=True)
class FermiSearchResult:
    fermi_energy: float
    weights: FloatArray
    iterations: int


def occupation_weights(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
    fermi_energy: float = 0.0,
) -> FloatArray:
    eig_flat, energy_grid_shape = normalize_eigenvalues(eigenvalues)
    mesh = build_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    tetra_band_energies = interpolated_tetrahedron_energies(mesh, eig_flat)
    local_weights = _occupation_weights_on_local_mesh(mesh, tetra_band_energies, fermi_energy)
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_band_last(output_flat, mesh.weight_grid_shape)


def occ(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
    fermi_energy: float = 0.0,
) -> FloatArray:
    return occupation_weights(
        reciprocal_vectors,
        eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
        fermi_energy=fermi_energy,
    )


def find_fermi_energy(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    electrons_per_spin: float,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
    tolerance: float = 1.0e-10,
    max_iterations: int = 300,
) -> FermiSearchResult:
    eig_flat, energy_grid_shape = normalize_eigenvalues(eigenvalues)
    mesh = build_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    tetra_band_energies = interpolated_tetrahedron_energies(mesh, eig_flat)

    lower = float(eig_flat.min())
    upper = float(eig_flat.max())
    local_weights = np.empty((mesh.local_point_count, eig_flat.shape[1]), dtype=np.float64)
    fermi_energy = 0.0
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        fermi_energy = 0.5 * (upper + lower)
        local_weights = _occupation_weights_on_local_mesh(mesh, tetra_band_energies, fermi_energy)
        electron_total = float(local_weights.sum())
        if abs(electron_total - electrons_per_spin) < tolerance:
            break
        if electron_total < electrons_per_spin:
            lower = fermi_energy
        else:
            upper = fermi_energy
    else:
        raise RuntimeError("fermi level search did not converge")

    output_flat = interpolate_local_values(mesh, local_weights)
    return FermiSearchResult(
        fermi_energy=fermi_energy,
        weights=_unflatten_band_last(output_flat, mesh.weight_grid_shape),
        iterations=iteration,
    )


def solve_fermi_energy(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    electrons_per_spin: float,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
    tolerance: float = 1.0e-10,
    max_iterations: int = 300,
) -> FermiSearchResult:
    return find_fermi_energy(
        reciprocal_vectors,
        eigenvalues,
        electrons_per_spin,
        weight_grid_shape=weight_grid_shape,
        method=method,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def fermieng(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    electrons_per_spin: float,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
    tolerance: float = 1.0e-10,
    max_iterations: int = 300,
) -> tuple[float, FloatArray, int]:
    result = find_fermi_energy(
        reciprocal_vectors,
        eigenvalues,
        electrons_per_spin,
        weight_grid_shape=weight_grid_shape,
        method=method,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return result.fermi_energy, result.weights, result.iterations


def _occupation_weights_on_local_mesh(
    mesh: IntegrationMesh,
    tetra_band_energies: FloatArray,
    fermi_energy: float,
) -> FloatArray:
    normalization = 6 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return _occupation_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        tetra_band_energies,
        fermi_energy,
        mesh.local_point_count,
        normalization,
    )


def _unflatten_band_last(values: FloatArray, grid_shape: tuple[int, int, int]) -> FloatArray:
    band_count = values.shape[1]
    reshaped = values.reshape((grid_shape[2], grid_shape[1], grid_shape[0], band_count))
    return np.transpose(reshaped, (2, 1, 0, 3))


@njit(cache=True)
def _occupation_weights_on_local_mesh_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    tetra_band_energies: FloatArray,
    fermi_energy: float,
    local_point_count: int,
    normalization: int,
) -> FloatArray:
    tetrahedron_count = tetra_band_energies.shape[0]
    band_count = tetra_band_energies.shape[2]
    local_weights = np.zeros((local_point_count, band_count), dtype=np.float64)

    sorted_order = np.empty(4, dtype=np.int64)
    sorted_energies = np.empty(4, dtype=np.float64)
    strict_energies = np.empty(4, dtype=np.float64)
    vertex_weights = np.empty(4, dtype=np.float64)
    point_weights = np.empty(20, dtype=np.float64)
    affine = np.empty((4, 4), dtype=np.float64)
    coefficients = np.empty((4, 4), dtype=np.float64)

    for tetrahedron_index in range(tetrahedron_count):
        for band_index in range(band_count):
            sort4_shifted(
                tetra_band_energies[tetrahedron_index, :, band_index],
                fermi_energy,
                sorted_order,
                sorted_energies,
            )
            vertex_weights[:] = 0.0

            if sorted_energies[0] <= 0.0 < sorted_energies[1]:
                accumulate_small_tetra_weight_sums(
                    vertex_weights,
                    sorted_order,
                    0,
                    sorted_energies,
                    strict_energies,
                    0.25,
                    affine,
                    coefficients,
                )
            elif sorted_energies[1] <= 0.0 < sorted_energies[2]:
                accumulate_small_tetra_weight_sums(
                    vertex_weights,
                    sorted_order,
                    1,
                    sorted_energies,
                    strict_energies,
                    0.25,
                    affine,
                    coefficients,
                )
                accumulate_small_tetra_weight_sums(
                    vertex_weights,
                    sorted_order,
                    2,
                    sorted_energies,
                    strict_energies,
                    0.25,
                    affine,
                    coefficients,
                )
                accumulate_small_tetra_weight_sums(
                    vertex_weights,
                    sorted_order,
                    3,
                    sorted_energies,
                    strict_energies,
                    0.25,
                    affine,
                    coefficients,
                )
            elif sorted_energies[2] <= 0.0 < sorted_energies[3]:
                accumulate_small_tetra_weight_sums(
                    vertex_weights,
                    sorted_order,
                    4,
                    sorted_energies,
                    strict_energies,
                    0.25,
                    affine,
                    coefficients,
                )
                accumulate_small_tetra_weight_sums(
                    vertex_weights,
                    sorted_order,
                    5,
                    sorted_energies,
                    strict_energies,
                    0.25,
                    affine,
                    coefficients,
                )
                accumulate_small_tetra_weight_sums(
                    vertex_weights,
                    sorted_order,
                    6,
                    sorted_energies,
                    strict_energies,
                    0.25,
                    affine,
                    coefficients,
                )
            elif sorted_energies[3] <= 0.0:
                vertex_weights[:] = 0.25

            for point_index in range(20):
                total = 0.0
                for vertex_index in range(4):
                    total += vertex_weights[vertex_index] * tetrahedron_weight_matrix[vertex_index, point_index]
                point_weights[point_index] = total

            for point_index in range(20):
                local_weights[local_point_indices[tetrahedron_index, point_index], band_index] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights
