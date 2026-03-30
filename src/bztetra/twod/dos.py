from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

from ._grids import FloatArray
from ._grids import interpolate_local_values
from ._grids import interpolated_triangle_energies
from ._grids import normalize_eigenvalues
from ._grids import normalize_energy_samples
from ._triangle_kernels import fill_dos_vertex_weights
from ._triangle_kernels import fill_occupation_vertex_weights
from ._triangle_kernels import sort3
from .geometry import cached_integration_mesh
from .geometry import TriangleIntegrationMesh
from .geometry import TriangleMethod


def density_of_states_weights(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> FloatArray:
    """Compute 2D DOS weights for `(nx, ny, nbands)` eigenvalues."""

    eig_flat, energy_grid_shape = normalize_eigenvalues(eigenvalues)
    sample_energies = normalize_energy_samples(energies)
    mesh = cached_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    triangle_band_energies = interpolated_triangle_energies(mesh, eig_flat)
    local_weights = _dos_weights_on_local_mesh(mesh, triangle_band_energies, sample_energies)
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_energy_band_last(output_flat, mesh.weight_grid_shape)


def integrated_density_of_states_weights(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> FloatArray:
    """Compute 2D integrated-DOS weights for `(nx, ny, nbands)` eigenvalues."""

    eig_flat, energy_grid_shape = normalize_eigenvalues(eigenvalues)
    sample_energies = normalize_energy_samples(energies)
    mesh = cached_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    triangle_band_energies = interpolated_triangle_energies(mesh, eig_flat)
    local_weights = _intdos_weights_on_local_mesh(mesh, triangle_band_energies, sample_energies)
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_energy_band_last(output_flat, mesh.weight_grid_shape)


def _dos_weights_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    triangle_band_energies: FloatArray,
    sample_energies: FloatArray,
) -> FloatArray:
    normalization = 2 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return _dos_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        triangle_band_energies,
        sample_energies,
        mesh.local_point_count,
        normalization,
    )


def _intdos_weights_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    triangle_band_energies: FloatArray,
    sample_energies: FloatArray,
) -> FloatArray:
    normalization = 2 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return _intdos_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        triangle_band_energies,
        sample_energies,
        mesh.local_point_count,
        normalization,
    )


def _unflatten_energy_band_last(values: FloatArray, grid_shape: tuple[int, int]) -> FloatArray:
    energy_count = values.shape[1]
    band_count = values.shape[2]
    reshaped = values.reshape((grid_shape[1], grid_shape[0], energy_count, band_count))
    return np.transpose(reshaped, (2, 1, 0, 3))


@njit(cache=True)
def _dos_weights_on_local_mesh_numba(
    local_point_indices,
    triangle_band_energies,
    sample_energies,
    local_point_count,
    normalization,
) -> FloatArray:
    triangle_count = triangle_band_energies.shape[0]
    band_count = triangle_band_energies.shape[2]
    energy_count = sample_energies.shape[0]
    local_weights = np.zeros((local_point_count, energy_count, band_count), dtype=np.float64)

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
            for energy_index in range(energy_count):
                fill_dos_vertex_weights(
                    vertex_weights,
                    sorted_order,
                    sorted_energies,
                    sample_energies[energy_index],
                    strict_energies,
                )
                for vertex_index in range(3):
                    local_weights[
                        local_point_indices[triangle_index, vertex_index],
                        energy_index,
                        band_index,
                    ] += vertex_weights[vertex_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True)
def _intdos_weights_on_local_mesh_numba(
    local_point_indices,
    triangle_band_energies,
    sample_energies,
    local_point_count,
    normalization,
) -> FloatArray:
    triangle_count = triangle_band_energies.shape[0]
    band_count = triangle_band_energies.shape[2]
    energy_count = sample_energies.shape[0]
    local_weights = np.zeros((local_point_count, energy_count, band_count), dtype=np.float64)

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
            for energy_index in range(energy_count):
                fill_occupation_vertex_weights(
                    vertex_weights,
                    sorted_order,
                    sorted_energies,
                    sample_energies[energy_index],
                    strict_energies,
                )
                for vertex_index in range(3):
                    local_weights[
                        local_point_indices[triangle_index, vertex_index],
                        energy_index,
                        band_index,
                    ] += vertex_weights[vertex_index]

    local_weights /= float(normalization)
    return local_weights
