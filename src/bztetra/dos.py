from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

from ._cut_kernels import accumulate_small_tetra_weight_sums
from ._cut_kernels import accumulate_triangle_weight_sums
from ._cut_kernels import sort4
from ._grids import FloatArray
from ._grids import interpolate_local_values
from ._grids import interpolated_tetrahedron_energies
from ._grids import normalize_eigenvalues
from ._grids import normalize_energy_samples
from .geometry import IntegrationMesh
from .geometry import TetraMethod
from .geometry import cached_integration_mesh


def density_of_states_weights(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    """Compute DOS weights for `(nx, ny, nz, nbands)` eigenvalues.

    `energies` must be one-dimensional. The result has shape
    `(nenergy, wx, wy, wz, nbands)`, where `(wx, wy, wz)` is `weight_grid_shape`
    or the input grid when `weight_grid_shape` is omitted. Replaces
    `libtetrabz_dos`. Set `method="linear"` only when reproducing the legacy
    linear tetrahedron scheme.
    """

    eig_flat, energy_grid_shape = normalize_eigenvalues(eigenvalues)
    sample_energies = normalize_energy_samples(energies)
    mesh = cached_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    tetra_band_energies = interpolated_tetrahedron_energies(mesh, eig_flat)
    local_weights = _dos_weights_on_local_mesh(mesh, tetra_band_energies, sample_energies)
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_energy_band_last(output_flat, mesh.weight_grid_shape)


def integrated_density_of_states_weights(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    """Compute integrated-DOS weights for `(nx, ny, nz, nbands)` eigenvalues.

    `energies` must be one-dimensional. The result has shape
    `(nenergy, wx, wy, wz, nbands)`, where `(wx, wy, wz)` is `weight_grid_shape`
    or the input grid when `weight_grid_shape` is omitted. Replaces
    `libtetrabz_intdos`. Set `method="linear"` only when reproducing the legacy
    linear tetrahedron scheme.
    """

    eig_flat, energy_grid_shape = normalize_eigenvalues(eigenvalues)
    sample_energies = normalize_energy_samples(energies)
    mesh = cached_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    tetra_band_energies = interpolated_tetrahedron_energies(mesh, eig_flat)
    local_weights = _intdos_weights_on_local_mesh(mesh, tetra_band_energies, sample_energies)
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_energy_band_last(output_flat, mesh.weight_grid_shape)


def _dos_weights_on_local_mesh(
    mesh: IntegrationMesh,
    tetra_band_energies: FloatArray,
    sample_energies: FloatArray,
) -> FloatArray:
    normalization = 6 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    sample_energies_sorted = bool(np.all(sample_energies[1:] >= sample_energies[:-1]))
    return _dos_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        tetra_band_energies,
        sample_energies,
        sample_energies_sorted,
        mesh.local_point_count,
        normalization,
    )


def _intdos_weights_on_local_mesh(
    mesh: IntegrationMesh,
    tetra_band_energies: FloatArray,
    sample_energies: FloatArray,
) -> FloatArray:
    normalization = 6 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    sample_energies_sorted = bool(np.all(sample_energies[1:] >= sample_energies[:-1]))
    return _intdos_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        tetra_band_energies,
        sample_energies,
        sample_energies_sorted,
        mesh.local_point_count,
        normalization,
    )


def _unflatten_energy_band_last(values: FloatArray, grid_shape: tuple[int, int, int]) -> FloatArray:
    energy_count = values.shape[1]
    band_count = values.shape[2]
    reshaped = values.reshape((grid_shape[2], grid_shape[1], grid_shape[0], energy_count, band_count))
    return np.transpose(reshaped, (3, 2, 1, 0, 4))


@njit(cache=True)
def _active_open_energy_window(
    sample_energies: FloatArray,
    sample_energies_sorted: bool,
    lower_exclusive: float,
    upper_exclusive: float,
) -> tuple[int, int]:
    energy_count = sample_energies.shape[0]
    if not sample_energies_sorted:
        return 0, energy_count

    left = 0
    right = energy_count
    while left < right:
        middle = (left + right) // 2
        if sample_energies[middle] <= lower_exclusive:
            left = middle + 1
        else:
            right = middle
    start = left

    left = start
    right = energy_count
    while left < right:
        middle = (left + right) // 2
        if sample_energies[middle] < upper_exclusive:
            left = middle + 1
        else:
            right = middle
    return start, left


@njit(cache=True)
def _occupation_energy_windows(
    sample_energies: FloatArray,
    sample_energies_sorted: bool,
    lower_exclusive: float,
    upper_inclusive: float,
) -> tuple[int, int]:
    energy_count = sample_energies.shape[0]
    if not sample_energies_sorted:
        return 0, energy_count

    left = 0
    right = energy_count
    while left < right:
        middle = (left + right) // 2
        if sample_energies[middle] <= lower_exclusive:
            left = middle + 1
        else:
            right = middle
    start = left

    left = start
    right = energy_count
    while left < right:
        middle = (left + right) // 2
        if sample_energies[middle] < upper_inclusive:
            left = middle + 1
        else:
            right = middle
    return start, left


@njit(cache=True)
def _dos_weights_on_local_mesh_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    tetra_band_energies: FloatArray,
    sample_energies: FloatArray,
    sample_energies_sorted: bool,
    local_point_count: int,
    normalization: int,
) -> FloatArray:
    tetrahedron_count = tetra_band_energies.shape[0]
    band_count = tetra_band_energies.shape[2]
    energy_count = sample_energies.shape[0]
    local_weights = np.zeros((local_point_count, energy_count, band_count), dtype=np.float64)

    sorted_order = np.empty(4, dtype=np.int64)
    sorted_energies = np.empty(4, dtype=np.float64)
    shifted_energies = np.empty(4, dtype=np.float64)
    strict_energies = np.empty(4, dtype=np.float64)
    vertex_weights = np.empty(4, dtype=np.float64)
    point_weights = np.empty(20, dtype=np.float64)
    affine = np.empty((4, 4), dtype=np.float64)
    coefficients = np.empty((3, 4), dtype=np.float64)

    for tetrahedron_index in range(tetrahedron_count):
        for band_index in range(band_count):
            sort4(
                tetra_band_energies[tetrahedron_index, :, band_index],
                sorted_order,
                sorted_energies,
            )
            start, end = _active_open_energy_window(
                sample_energies,
                sample_energies_sorted,
                sorted_energies[0],
                sorted_energies[3],
            )
            for energy_index in range(start, end):
                energy = sample_energies[energy_index]
                vertex_weights[:] = 0.0
                active = False
                for vertex_index in range(4):
                    shifted_energies[vertex_index] = sorted_energies[vertex_index] - energy

                if sorted_energies[0] <= energy <= sorted_energies[1]:
                    accumulate_triangle_weight_sums(
                        vertex_weights,
                        sorted_order,
                        0,
                        shifted_energies,
                        strict_energies,
                        1.0 / 3.0,
                        affine,
                        coefficients,
                    )
                    active = True
                elif sorted_energies[1] <= energy <= sorted_energies[2]:
                    accumulate_triangle_weight_sums(
                        vertex_weights,
                        sorted_order,
                        1,
                        shifted_energies,
                        strict_energies,
                        1.0 / 3.0,
                        affine,
                        coefficients,
                    )
                    accumulate_triangle_weight_sums(
                        vertex_weights,
                        sorted_order,
                        2,
                        shifted_energies,
                        strict_energies,
                        1.0 / 3.0,
                        affine,
                        coefficients,
                    )
                    active = True
                elif sorted_energies[2] <= energy <= sorted_energies[3]:
                    accumulate_triangle_weight_sums(
                        vertex_weights,
                        sorted_order,
                        3,
                        shifted_energies,
                        strict_energies,
                        1.0 / 3.0,
                        affine,
                        coefficients,
                    )
                    active = True

                if not active:
                    continue

                for point_index in range(20):
                    total = 0.0
                    for vertex_index in range(4):
                        total += vertex_weights[vertex_index] * tetrahedron_weight_matrix[vertex_index, point_index]
                    point_weights[point_index] = total

                for point_index in range(20):
                    local_weights[local_point_indices[tetrahedron_index, point_index], energy_index, band_index] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True)
def _intdos_weights_on_local_mesh_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    tetra_band_energies: FloatArray,
    sample_energies: FloatArray,
    sample_energies_sorted: bool,
    local_point_count: int,
    normalization: int,
) -> FloatArray:
    tetrahedron_count = tetra_band_energies.shape[0]
    band_count = tetra_band_energies.shape[2]
    energy_count = sample_energies.shape[0]
    local_weights = np.zeros((local_point_count, energy_count, band_count), dtype=np.float64)

    sorted_order = np.empty(4, dtype=np.int64)
    sorted_energies = np.empty(4, dtype=np.float64)
    shifted_energies = np.empty(4, dtype=np.float64)
    strict_energies = np.empty(4, dtype=np.float64)
    vertex_weights = np.empty(4, dtype=np.float64)
    point_weights = np.empty(20, dtype=np.float64)
    full_point_weights = np.empty(20, dtype=np.float64)
    affine = np.empty((4, 4), dtype=np.float64)
    coefficients = np.empty((4, 4), dtype=np.float64)

    for point_index in range(20):
        total = 0.0
        for vertex_index in range(4):
            total += 0.25 * tetrahedron_weight_matrix[vertex_index, point_index]
        full_point_weights[point_index] = total

    for tetrahedron_index in range(tetrahedron_count):
        for band_index in range(band_count):
            sort4(
                tetra_band_energies[tetrahedron_index, :, band_index],
                sorted_order,
                sorted_energies,
            )
            varying_start, full_start = _occupation_energy_windows(
                sample_energies,
                sample_energies_sorted,
                sorted_energies[0],
                sorted_energies[3],
            )

            if sample_energies_sorted:
                for energy_index in range(varying_start, full_start):
                    energy = sample_energies[energy_index]
                    vertex_weights[:] = 0.0
                    for vertex_index in range(4):
                        shifted_energies[vertex_index] = sorted_energies[vertex_index] - energy

                    if energy <= sorted_energies[1]:
                        accumulate_small_tetra_weight_sums(
                            vertex_weights,
                            sorted_order,
                            0,
                            shifted_energies,
                            strict_energies,
                            0.25,
                            affine,
                            coefficients,
                        )
                    elif energy <= sorted_energies[2]:
                        accumulate_small_tetra_weight_sums(
                            vertex_weights,
                            sorted_order,
                            1,
                            shifted_energies,
                            strict_energies,
                            0.25,
                            affine,
                            coefficients,
                        )
                        accumulate_small_tetra_weight_sums(
                            vertex_weights,
                            sorted_order,
                            2,
                            shifted_energies,
                            strict_energies,
                            0.25,
                            affine,
                            coefficients,
                        )
                        accumulate_small_tetra_weight_sums(
                            vertex_weights,
                            sorted_order,
                            3,
                            shifted_energies,
                            strict_energies,
                            0.25,
                            affine,
                            coefficients,
                        )
                    else:
                        accumulate_small_tetra_weight_sums(
                            vertex_weights,
                            sorted_order,
                            4,
                            shifted_energies,
                            strict_energies,
                            0.25,
                            affine,
                            coefficients,
                        )
                        accumulate_small_tetra_weight_sums(
                            vertex_weights,
                            sorted_order,
                            5,
                            shifted_energies,
                            strict_energies,
                            0.25,
                            affine,
                            coefficients,
                        )
                        accumulate_small_tetra_weight_sums(
                            vertex_weights,
                            sorted_order,
                            6,
                            shifted_energies,
                            strict_energies,
                            0.25,
                            affine,
                            coefficients,
                        )

                    for point_index in range(20):
                        total = 0.0
                        for vertex_index in range(4):
                            total += vertex_weights[vertex_index] * tetrahedron_weight_matrix[vertex_index, point_index]
                        point_weights[point_index] = total

                    for point_index in range(20):
                        local_weights[local_point_indices[tetrahedron_index, point_index], energy_index, band_index] += point_weights[point_index]

                for energy_index in range(full_start, energy_count):
                    for point_index in range(20):
                        local_weights[
                            local_point_indices[tetrahedron_index, point_index],
                            energy_index,
                            band_index,
                        ] += full_point_weights[point_index]
                continue

            for energy_index in range(energy_count):
                energy = sample_energies[energy_index]
                vertex_weights[:] = 0.0
                active = False
                for vertex_index in range(4):
                    shifted_energies[vertex_index] = sorted_energies[vertex_index] - energy

                if sorted_energies[0] <= energy <= sorted_energies[1]:
                    accumulate_small_tetra_weight_sums(
                        vertex_weights,
                        sorted_order,
                        0,
                        shifted_energies,
                        strict_energies,
                        0.25,
                        affine,
                        coefficients,
                    )
                    active = True
                elif sorted_energies[1] <= energy <= sorted_energies[2]:
                    accumulate_small_tetra_weight_sums(
                        vertex_weights,
                        sorted_order,
                        1,
                        shifted_energies,
                        strict_energies,
                        0.25,
                        affine,
                        coefficients,
                    )
                    accumulate_small_tetra_weight_sums(
                        vertex_weights,
                        sorted_order,
                        2,
                        shifted_energies,
                        strict_energies,
                        0.25,
                        affine,
                        coefficients,
                    )
                    accumulate_small_tetra_weight_sums(
                        vertex_weights,
                        sorted_order,
                        3,
                        shifted_energies,
                        strict_energies,
                        0.25,
                        affine,
                        coefficients,
                    )
                    active = True
                elif sorted_energies[2] <= energy <= sorted_energies[3]:
                    accumulate_small_tetra_weight_sums(
                        vertex_weights,
                        sorted_order,
                        4,
                        shifted_energies,
                        strict_energies,
                        0.25,
                        affine,
                        coefficients,
                    )
                    accumulate_small_tetra_weight_sums(
                        vertex_weights,
                        sorted_order,
                        5,
                        shifted_energies,
                        strict_energies,
                        0.25,
                        affine,
                        coefficients,
                    )
                    accumulate_small_tetra_weight_sums(
                        vertex_weights,
                        sorted_order,
                        6,
                        shifted_energies,
                        strict_energies,
                        0.25,
                        affine,
                        coefficients,
                    )
                    active = True
                elif sorted_energies[3] <= energy:
                    vertex_weights[:] = 0.25
                    active = True

                if not active:
                    continue

                for point_index in range(20):
                    total = 0.0
                    for vertex_index in range(4):
                        total += vertex_weights[vertex_index] * tetrahedron_weight_matrix[vertex_index, point_index]
                    point_weights[point_index] = total

                for point_index in range(20):
                    local_weights[local_point_indices[tetrahedron_index, point_index], energy_index, band_index] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights
