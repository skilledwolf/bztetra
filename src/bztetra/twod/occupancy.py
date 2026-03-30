from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

from ..occupancy import FermiEnergySolution
from ._grids import FloatArray
from ._grids import interpolate_local_values
from ._grids import interpolated_triangle_energies
from ._grids import normalize_eigenvalues
from ._triangle_kernels import fill_occupation_vertex_weights
from ._triangle_kernels import sort3
from .geometry import cached_integration_mesh
from .geometry import TriangleIntegrationMesh
from .geometry import TriangleMethod


def occupation_weights(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
    fermi_energy: float = 0.0,
) -> FloatArray:
    """Compute 2D occupation weights for `(nx, ny, nbands)` eigenvalues."""

    eig_flat, energy_grid_shape = normalize_eigenvalues(eigenvalues)
    mesh = cached_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    triangle_band_energies = interpolated_triangle_energies(mesh, eig_flat)
    local_weights = _occupation_weights_on_local_mesh(mesh, triangle_band_energies, fermi_energy)
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_band_last(output_flat, mesh.weight_grid_shape)


def solve_fermi_energy(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    electrons_per_spin: float,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
    tolerance: float = 1.0e-10,
    max_iterations: int = 300,
) -> FermiEnergySolution:
    """Solve for the 2D Fermi energy and occupation weights."""

    eig_flat, energy_grid_shape = normalize_eigenvalues(eigenvalues)
    mesh = cached_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    triangle_band_energies = interpolated_triangle_energies(mesh, eig_flat)

    lower = float(eig_flat.min())
    upper = float(eig_flat.max())
    fermi_energy = 0.0
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        fermi_energy = 0.5 * (upper + lower)
        electron_total = _occupation_total_on_local_mesh(
            triangle_band_energies,
            fermi_energy,
            mesh.energy_grid_shape,
        )
        if abs(electron_total - electrons_per_spin) < tolerance:
            break
        if electron_total < electrons_per_spin:
            lower = fermi_energy
        else:
            upper = fermi_energy
    else:
        raise RuntimeError("fermi level search did not converge")

    local_weights = _occupation_weights_on_local_mesh(mesh, triangle_band_energies, fermi_energy)
    output_flat = interpolate_local_values(mesh, local_weights)
    return FermiEnergySolution(
        fermi_energy=fermi_energy,
        weights=_unflatten_band_last(output_flat, mesh.weight_grid_shape),
        iterations=iteration,
    )


def _occupation_weights_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    triangle_band_energies: FloatArray,
    fermi_energy: float,
) -> FloatArray:
    normalization = 2 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return _occupation_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        triangle_band_energies,
        fermi_energy,
        mesh.local_point_count,
        normalization,
    )


def _occupation_total_on_local_mesh(
    triangle_band_energies: FloatArray,
    fermi_energy: float,
    energy_grid_shape: tuple[int, int],
) -> float:
    normalization = 2 * int(np.prod(energy_grid_shape, dtype=np.int64))
    return _occupation_total_on_local_mesh_numba(
        triangle_band_energies,
        fermi_energy,
        normalization,
    )


def _unflatten_band_last(values: FloatArray, grid_shape: tuple[int, int]) -> FloatArray:
    band_count = values.shape[1]
    reshaped = values.reshape((grid_shape[1], grid_shape[0], band_count))
    return np.transpose(reshaped, (1, 0, 2))


@njit(cache=True)
def _occupation_weights_on_local_mesh_numba(
    local_point_indices,
    triangle_band_energies,
    fermi_energy,
    local_point_count,
    normalization,
) -> FloatArray:
    triangle_count = triangle_band_energies.shape[0]
    band_count = triangle_band_energies.shape[2]
    local_weights = np.zeros((local_point_count, band_count), dtype=np.float64)

    sorted_order = np.empty(3, dtype=np.int64)
    sorted_energies = np.empty(3, dtype=np.float64)
    strict_energies = np.empty(3, dtype=np.float64)
    vertex_weights = np.empty(3, dtype=np.float64)

    for triangle_index in range(triangle_count):
        for band_index in range(band_count):
            sort3(
                triangle_band_energies[triangle_index, :, band_index],
                sorted_order,
                sorted_energies,
            )
            fill_occupation_vertex_weights(
                vertex_weights,
                sorted_order,
                sorted_energies,
                fermi_energy,
                strict_energies,
            )
            for vertex_index in range(3):
                local_weights[
                    local_point_indices[triangle_index, vertex_index],
                    band_index,
                ] += vertex_weights[vertex_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True)
def _occupation_total_on_local_mesh_numba(
    triangle_band_energies,
    fermi_energy,
    normalization,
) -> float:
    triangle_count = triangle_band_energies.shape[0]
    band_count = triangle_band_energies.shape[2]
    electron_total = 0.0

    sorted_order = np.empty(3, dtype=np.int64)
    sorted_energies = np.empty(3, dtype=np.float64)
    strict_energies = np.empty(3, dtype=np.float64)
    vertex_weights = np.empty(3, dtype=np.float64)

    for triangle_index in range(triangle_count):
        for band_index in range(band_count):
            sort3(
                triangle_band_energies[triangle_index, :, band_index],
                sorted_order,
                sorted_energies,
            )
            fill_occupation_vertex_weights(
                vertex_weights,
                sorted_order,
                sorted_energies,
                fermi_energy,
                strict_energies,
            )
            electron_total += vertex_weights[0] + vertex_weights[1] + vertex_weights[2]

    return electron_total / float(normalization)
