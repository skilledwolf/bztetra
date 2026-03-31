from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit
from numba import prange

from ._cut_kernels import accumulate_triangle_weight_sums
from ._cut_kernels import small_tetra_volume_and_coefficients
from ._cut_kernels import sort4
from ._grids import FloatArray
from .geometry import IntegrationMesh
from ._response_common import ComplexArray


def _fermi_golden_rule_weights_on_local_mesh(
    mesh: IntegrationMesh,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: FloatArray,
) -> FloatArray:
    normalization = 6 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    pair_count = occupied_tetra.shape[2] * target_tetra.shape[2]
    sample_energies_sorted = bool(np.all(sample_energies[1:] >= sample_energies[:-1]))
    if pair_count >= 16 and target_tetra.shape[2] >= 4:
        return _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            mesh.tetrahedron_weight_matrix,
            occupied_tetra,
            target_tetra,
            sample_energies,
            sample_energies_sorted,
            mesh.local_point_count,
            normalization,
        )
    return _fermi_golden_rule_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        occupied_tetra,
        target_tetra,
        sample_energies,
        sample_energies_sorted,
        mesh.local_point_count,
        normalization,
    )


def _complex_polarization_weights_on_local_mesh(
    mesh: IntegrationMesh,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: ComplexArray,
) -> ComplexArray:
    normalization = 6 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    sample_energy_real = np.ascontiguousarray(sample_energies.real)
    sample_energy_imag = np.ascontiguousarray(sample_energies.imag)
    return _complex_polarization_weights_on_local_mesh_pair_parallel_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        occupied_tetra,
        target_tetra,
        sample_energy_real,
        sample_energy_imag,
        mesh.local_point_count,
        normalization,
    )


@njit(cache=True, parallel=True)
def _fermi_golden_rule_weights_on_local_mesh_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: FloatArray,
    sample_energies_sorted: bool,
    local_point_count: int,
    normalization: int,
) -> FloatArray:
    tetrahedron_count = occupied_tetra.shape[0]
    source_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    energy_count = sample_energies.shape[0]
    local_weights = np.zeros(
        (local_point_count, energy_count, target_band_count, source_band_count), dtype=np.float64
    )

    for source_band_index in prange(source_band_count):
        sorted_order = np.empty(4, dtype=np.int64)
        sorted_occupied = np.empty(4, dtype=np.float64)
        sorted_target = np.empty((4, target_band_count), dtype=np.float64)
        outer_weights = np.empty((energy_count, target_band_count, 4), dtype=np.float64)
        point_weights = np.empty(20, dtype=np.float64)

        outer_strict = np.empty(4, dtype=np.float64)
        outer_affine = np.empty((4, 4), dtype=np.float64)
        outer_coefficients = np.empty((4, 4), dtype=np.float64)
        transformed_occupied = np.empty(4, dtype=np.float64)
        transformed_target = np.empty((4, target_band_count), dtype=np.float64)

        secondary_weights = np.empty((energy_count, target_band_count, 4), dtype=np.float64)
        step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_order = np.empty(4, dtype=np.int64)
        secondary_sorted_step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_target = np.empty(4, dtype=np.float64)
        secondary_sorted_occupied = np.empty(4, dtype=np.float64)
        secondary_sorted_weights = np.empty((energy_count, 4), dtype=np.float64)
        inner_strict = np.empty(4, dtype=np.float64)
        inner_affine = np.empty((4, 4), dtype=np.float64)
        inner_coefficients = np.empty((4, 4), dtype=np.float64)
        energy_differences = np.empty(4, dtype=np.float64)
        delta_weights = np.empty((energy_count, 4), dtype=np.float64)
        delta_sorted_order = np.empty(4, dtype=np.int64)
        delta_sorted_energies = np.empty(4, dtype=np.float64)
        delta_shifted_energies = np.empty(4, dtype=np.float64)
        triangle_strict = np.empty(4, dtype=np.float64)
        triangle_affine = np.empty((4, 4), dtype=np.float64)
        triangle_coefficients = np.empty((3, 4), dtype=np.float64)
        secondary_active_starts = np.empty(target_band_count, dtype=np.int64)
        secondary_active_ends = np.empty(target_band_count, dtype=np.int64)
        outer_active_starts = np.empty(target_band_count, dtype=np.int64)
        outer_active_ends = np.empty(target_band_count, dtype=np.int64)

        for tetrahedron_index in range(tetrahedron_count):
            sort4(
                occupied_tetra[tetrahedron_index, :, source_band_index],
                sorted_order,
                sorted_occupied,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                for target_band_index in range(target_band_count):
                    sorted_target[vertex_index, target_band_index] = target_tetra[
                        tetrahedron_index,
                        source_vertex,
                        target_band_index,
                    ]
            outer_weights[:, :, :] = 0.0
            for target_band_index in range(target_band_count):
                outer_active_starts[target_band_index] = energy_count
                outer_active_ends[target_band_index] = 0
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_fermigr_outer_numba(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    sample_energies_sorted,
                    target_band_count,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_occupied,
                    transformed_target,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                    secondary_active_starts,
                    secondary_active_ends,
                    outer_active_starts,
                    outer_active_ends,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for case_id in (1, 2, 3):
                    _accumulate_small_tetra_fermigr_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        sample_energies_sorted,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        delta_weights,
                        delta_sorted_order,
                        delta_sorted_energies,
                        delta_shifted_energies,
                        triangle_strict,
                        triangle_affine,
                        triangle_coefficients,
                        secondary_active_starts,
                        secondary_active_ends,
                        outer_active_starts,
                        outer_active_ends,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for case_id in (4, 5, 6):
                    _accumulate_small_tetra_fermigr_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        sample_energies_sorted,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        delta_weights,
                        delta_sorted_order,
                        delta_sorted_energies,
                        delta_shifted_energies,
                        triangle_strict,
                        triangle_affine,
                        triangle_coefficients,
                        secondary_active_starts,
                        secondary_active_ends,
                        outer_active_starts,
                        outer_active_ends,
                    )
            elif sorted_occupied[3] <= 0.0:
                _fermigr_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    target_tetra[tetrahedron_index],
                    sample_energies,
                    sample_energies_sorted,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                    secondary_active_starts,
                    secondary_active_ends,
                )
                for target_band_index in range(target_band_count):
                    start = secondary_active_starts[target_band_index]
                    end = secondary_active_ends[target_band_index]
                    if start < outer_active_starts[target_band_index]:
                        outer_active_starts[target_band_index] = start
                    if end > outer_active_ends[target_band_index]:
                        outer_active_ends[target_band_index] = end
                    for energy_index in range(start, end):
                        for vertex_index in range(4):
                            outer_weights[energy_index, target_band_index, vertex_index] += (
                                secondary_weights[
                                    energy_index,
                                    target_band_index,
                                    vertex_index,
                                ]
                            )

            for target_band_index in range(target_band_count):
                start = outer_active_starts[target_band_index]
                end = outer_active_ends[target_band_index]
                for energy_index in range(start, end):
                    for point_index in range(20):
                        total = 0.0
                        for vertex_index in range(4):
                            total += (
                                outer_weights[energy_index, target_band_index, vertex_index]
                                * tetrahedron_weight_matrix[
                                    vertex_index,
                                    point_index,
                                ]
                            )
                        point_weights[point_index] = total
                    for point_index in range(20):
                        local_weights[
                            local_point_indices[tetrahedron_index, point_index],
                            energy_index,
                            target_band_index,
                            source_band_index,
                        ] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True, parallel=True)
def _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: FloatArray,
    sample_energies_sorted: bool,
    local_point_count: int,
    normalization: int,
) -> FloatArray:
    tetrahedron_count = occupied_tetra.shape[0]
    source_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    energy_count = sample_energies.shape[0]
    pair_count = source_band_count * target_band_count
    local_weights = np.zeros(
        (local_point_count, energy_count, target_band_count, source_band_count), dtype=np.float64
    )

    for pair_index in prange(pair_count):
        source_band_index = pair_index // target_band_count
        target_band_index = pair_index % target_band_count

        sorted_order = np.empty(4, dtype=np.int64)
        sorted_occupied = np.empty(4, dtype=np.float64)
        sorted_target = np.empty((4, 1), dtype=np.float64)
        full_target = np.empty((4, 1), dtype=np.float64)
        outer_weights = np.empty((energy_count, 1, 4), dtype=np.float64)
        point_weights = np.empty(20, dtype=np.float64)

        outer_strict = np.empty(4, dtype=np.float64)
        outer_affine = np.empty((4, 4), dtype=np.float64)
        outer_coefficients = np.empty((4, 4), dtype=np.float64)
        transformed_occupied = np.empty(4, dtype=np.float64)
        transformed_target = np.empty((4, 1), dtype=np.float64)

        secondary_weights = np.empty((energy_count, 1, 4), dtype=np.float64)
        step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_order = np.empty(4, dtype=np.int64)
        secondary_sorted_step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_target = np.empty(4, dtype=np.float64)
        secondary_sorted_occupied = np.empty(4, dtype=np.float64)
        secondary_sorted_weights = np.empty((energy_count, 4), dtype=np.float64)
        inner_strict = np.empty(4, dtype=np.float64)
        inner_affine = np.empty((4, 4), dtype=np.float64)
        inner_coefficients = np.empty((4, 4), dtype=np.float64)
        energy_differences = np.empty(4, dtype=np.float64)
        delta_weights = np.empty((energy_count, 4), dtype=np.float64)
        delta_sorted_order = np.empty(4, dtype=np.int64)
        delta_sorted_energies = np.empty(4, dtype=np.float64)
        delta_shifted_energies = np.empty(4, dtype=np.float64)
        triangle_strict = np.empty(4, dtype=np.float64)
        triangle_affine = np.empty((4, 4), dtype=np.float64)
        triangle_coefficients = np.empty((3, 4), dtype=np.float64)
        secondary_active_starts = np.empty(1, dtype=np.int64)
        secondary_active_ends = np.empty(1, dtype=np.int64)
        outer_active_starts = np.empty(1, dtype=np.int64)
        outer_active_ends = np.empty(1, dtype=np.int64)

        for tetrahedron_index in range(tetrahedron_count):
            sort4(
                occupied_tetra[tetrahedron_index, :, source_band_index],
                sorted_order,
                sorted_occupied,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                sorted_target[vertex_index, 0] = target_tetra[
                    tetrahedron_index,
                    source_vertex,
                    target_band_index,
                ]
                full_target[vertex_index, 0] = target_tetra[
                    tetrahedron_index,
                    vertex_index,
                    target_band_index,
                ]

            outer_weights[:, :, :] = 0.0
            outer_active_starts[0] = energy_count
            outer_active_ends[0] = 0
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_fermigr_outer_numba(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    sample_energies_sorted,
                    1,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_occupied,
                    transformed_target,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                    secondary_active_starts,
                    secondary_active_ends,
                    outer_active_starts,
                    outer_active_ends,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for case_id in (1, 2, 3):
                    _accumulate_small_tetra_fermigr_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        sample_energies_sorted,
                        1,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        delta_weights,
                        delta_sorted_order,
                        delta_sorted_energies,
                        delta_shifted_energies,
                        triangle_strict,
                        triangle_affine,
                        triangle_coefficients,
                        secondary_active_starts,
                        secondary_active_ends,
                        outer_active_starts,
                        outer_active_ends,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for case_id in (4, 5, 6):
                    _accumulate_small_tetra_fermigr_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        sample_energies_sorted,
                        1,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        delta_weights,
                        delta_sorted_order,
                        delta_sorted_energies,
                        delta_shifted_energies,
                        triangle_strict,
                        triangle_affine,
                        triangle_coefficients,
                        secondary_active_starts,
                        secondary_active_ends,
                        outer_active_starts,
                        outer_active_ends,
                    )
            elif sorted_occupied[3] <= 0.0:
                _fermigr_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    full_target,
                    sample_energies,
                    sample_energies_sorted,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                    secondary_active_starts,
                    secondary_active_ends,
                )
                if secondary_active_starts[0] < outer_active_starts[0]:
                    outer_active_starts[0] = secondary_active_starts[0]
                if secondary_active_ends[0] > outer_active_ends[0]:
                    outer_active_ends[0] = secondary_active_ends[0]
                for energy_index in range(secondary_active_starts[0], secondary_active_ends[0]):
                    for vertex_index in range(4):
                        outer_weights[energy_index, 0, vertex_index] += secondary_weights[
                            energy_index,
                            0,
                            vertex_index,
                        ]

            for energy_index in range(outer_active_starts[0], outer_active_ends[0]):
                for point_index in range(20):
                    total = 0.0
                    for vertex_index in range(4):
                        total += (
                            outer_weights[energy_index, 0, vertex_index]
                            * tetrahedron_weight_matrix[
                                vertex_index,
                                point_index,
                            ]
                        )
                    point_weights[point_index] = total
                for point_index in range(20):
                    local_weights[
                        local_point_indices[tetrahedron_index, point_index],
                        energy_index,
                        target_band_index,
                        source_band_index,
                    ] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True)
def _accumulate_small_tetra_fermigr_outer_numba(
    outer_weights: FloatArray,
    case_id: int,
    sorted_order: npt.NDArray[np.int64],
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energies: FloatArray,
    sample_energies_sorted: bool,
    target_band_count: int,
    outer_strict: FloatArray,
    outer_affine: FloatArray,
    outer_coefficients: FloatArray,
    transformed_occupied: FloatArray,
    transformed_target: FloatArray,
    secondary_weights: FloatArray,
    step_energies: FloatArray,
    secondary_sorted_order: npt.NDArray[np.int64],
    secondary_sorted_step_energies: FloatArray,
    secondary_sorted_target: FloatArray,
    secondary_sorted_occupied: FloatArray,
    secondary_sorted_weights: FloatArray,
    inner_strict: FloatArray,
    inner_affine: FloatArray,
    inner_coefficients: FloatArray,
    energy_differences: FloatArray,
    delta_weights: FloatArray,
    delta_sorted_order: npt.NDArray[np.int64],
    delta_sorted_energies: FloatArray,
    delta_shifted_energies: FloatArray,
    triangle_strict: FloatArray,
    triangle_affine: FloatArray,
    triangle_coefficients: FloatArray,
    secondary_active_starts: npt.NDArray[np.int64],
    secondary_active_ends: npt.NDArray[np.int64],
    outer_active_starts: npt.NDArray[np.int64],
    outer_active_ends: npt.NDArray[np.int64],
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_occupied,
        outer_strict,
        outer_affine,
        outer_coefficients,
    )
    if volume_factor <= 1.0e-10:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += outer_coefficients[row_index, column_index] * sorted_occupied[column_index]
        transformed_occupied[row_index] = total

    for row_index in range(4):
        for target_band_index in range(target_band_count):
            total = 0.0
            for column_index in range(4):
                total += (
                    outer_coefficients[row_index, column_index]
                    * sorted_target[column_index, target_band_index]
                )
            transformed_target[row_index, target_band_index] = total

    _fermigr_secondary_weights_numba(
        transformed_occupied,
        transformed_target,
        sample_energies,
        sample_energies_sorted,
        secondary_weights,
        step_energies,
        secondary_sorted_order,
        secondary_sorted_step_energies,
        secondary_sorted_target,
        secondary_sorted_occupied,
        secondary_sorted_weights,
        inner_strict,
        inner_affine,
        inner_coefficients,
        energy_differences,
        delta_weights,
        delta_sorted_order,
        delta_sorted_energies,
        delta_shifted_energies,
        triangle_strict,
        triangle_affine,
        triangle_coefficients,
        secondary_active_starts,
        secondary_active_ends,
    )

    for target_band_index in range(target_band_count):
        start = secondary_active_starts[target_band_index]
        end = secondary_active_ends[target_band_index]
        if start < outer_active_starts[target_band_index]:
            outer_active_starts[target_band_index] = start
        if end > outer_active_ends[target_band_index]:
            outer_active_ends[target_band_index] = end
        for energy_index in range(start, end):
            for column_index in range(4):
                total = 0.0
                for row_index in range(4):
                    total += (
                        secondary_weights[energy_index, target_band_index, row_index]
                        * outer_coefficients[
                            row_index,
                            column_index,
                        ]
                    )
                outer_weights[energy_index, target_band_index, sorted_order[column_index]] += (
                    volume_factor * total
                )


@njit(cache=True)
def _fermigr_secondary_weights_numba(
    occupied_vertices: FloatArray,
    target_vertices: FloatArray,
    sample_energies: FloatArray,
    sample_energies_sorted: bool,
    weights: FloatArray,
    step_energies: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_step_energies: FloatArray,
    sorted_target: FloatArray,
    sorted_occupied: FloatArray,
    sorted_weights: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    delta_weights: FloatArray,
    delta_sorted_order: npt.NDArray[np.int64],
    delta_sorted_energies: FloatArray,
    delta_shifted_energies: FloatArray,
    triangle_strict: FloatArray,
    triangle_affine: FloatArray,
    triangle_coefficients: FloatArray,
    active_starts: npt.NDArray[np.int64],
    active_ends: npt.NDArray[np.int64],
) -> None:
    target_band_count = target_vertices.shape[1]
    energy_count = sample_energies.shape[0]
    weights[:, :, :] = 0.0

    for target_band_index in range(target_band_count):
        for vertex_index in range(4):
            step_energies[vertex_index] = -target_vertices[vertex_index, target_band_index]
        sort4(step_energies, sorted_order, sorted_step_energies)
        for vertex_index in range(4):
            source_vertex = sorted_order[vertex_index]
            sorted_target[vertex_index] = target_vertices[source_vertex, target_band_index]
            sorted_occupied[vertex_index] = occupied_vertices[source_vertex]
        sorted_weights[:, :] = 0.0
        active_start = energy_count
        active_end = 0

        if (sorted_step_energies[0] <= 0.0 < sorted_step_energies[1]) or (
            sorted_step_energies[0] < 0.0 <= sorted_step_energies[1]
        ):
            active_start, active_end = _accumulate_small_tetra_fermigr_inner_numba(
                sorted_weights,
                0,
                sorted_step_energies,
                sorted_occupied,
                sorted_target,
                sample_energies,
                sample_energies_sorted,
                strict_energies,
                affine,
                coefficients,
                energy_differences,
                delta_weights,
                delta_sorted_order,
                delta_sorted_energies,
                delta_shifted_energies,
                triangle_strict,
                triangle_affine,
                triangle_coefficients,
            )
        elif (sorted_step_energies[1] <= 0.0 < sorted_step_energies[2]) or (
            sorted_step_energies[1] < 0.0 <= sorted_step_energies[2]
        ):
            for case_id in (1, 2, 3):
                start, end = _accumulate_small_tetra_fermigr_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    sample_energies_sorted,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                )
                if start < active_start:
                    active_start = start
                if end > active_end:
                    active_end = end
        elif (sorted_step_energies[2] <= 0.0 < sorted_step_energies[3]) or (
            sorted_step_energies[2] < 0.0 <= sorted_step_energies[3]
        ):
            for case_id in (4, 5, 6):
                start, end = _accumulate_small_tetra_fermigr_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    sample_energies_sorted,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                )
                if start < active_start:
                    active_start = start
                if end > active_end:
                    active_end = end
        elif sorted_step_energies[3] <= 0.0:
            for vertex_index in range(4):
                energy_differences[vertex_index] = (
                    sorted_target[vertex_index] - sorted_occupied[vertex_index]
                )
            active_start, active_end = _fermigr_delta_weights_numba(
                sample_energies,
                sample_energies_sorted,
                energy_differences,
                delta_weights,
                delta_sorted_order,
                delta_sorted_energies,
                delta_shifted_energies,
                triangle_strict,
                triangle_affine,
                triangle_coefficients,
            )
            for energy_index in range(active_start, active_end):
                for vertex_index in range(4):
                    sorted_weights[energy_index, vertex_index] += delta_weights[
                        energy_index, vertex_index
                    ]

        active_starts[target_band_index] = active_start
        active_ends[target_band_index] = active_end
        for energy_index in range(active_start, active_end):
            for vertex_index in range(4):
                weights[energy_index, target_band_index, sorted_order[vertex_index]] = (
                    sorted_weights[
                        energy_index,
                        vertex_index,
                    ]
                )


@njit(cache=True)
def _accumulate_small_tetra_fermigr_inner_numba(
    weights: FloatArray,
    case_id: int,
    sorted_step_energies: FloatArray,
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energies: FloatArray,
    sample_energies_sorted: bool,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    delta_weights: FloatArray,
    delta_sorted_order: npt.NDArray[np.int64],
    delta_sorted_energies: FloatArray,
    delta_shifted_energies: FloatArray,
    triangle_strict: FloatArray,
    triangle_affine: FloatArray,
    triangle_coefficients: FloatArray,
) -> tuple[int, int]:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_step_energies,
        strict_energies,
        affine,
        coefficients,
    )
    if volume_factor <= 1.0e-8:
        return weights.shape[0], 0

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += coefficients[row_index, column_index] * (
                sorted_target[column_index] - sorted_occupied[column_index]
            )
        energy_differences[row_index] = total

    start, end = _fermigr_delta_weights_numba(
        sample_energies,
        sample_energies_sorted,
        energy_differences,
        delta_weights,
        delta_sorted_order,
        delta_sorted_energies,
        delta_shifted_energies,
        triangle_strict,
        triangle_affine,
        triangle_coefficients,
    )

    for energy_index in range(start, end):
        for column_index in range(4):
            total = 0.0
            for row_index in range(4):
                total += (
                    delta_weights[energy_index, row_index] * coefficients[row_index, column_index]
                )
            weights[energy_index, column_index] += volume_factor * total
    return start, end


@njit(cache=True)
def _fermigr_delta_weights_numba(
    sample_energies: FloatArray,
    sample_energies_sorted: bool,
    energy_differences: FloatArray,
    weights: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_energies: FloatArray,
    shifted_energies: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
) -> tuple[int, int]:
    sort4(energy_differences, sorted_order, sorted_energies)
    start, end = _fermigr_active_energy_window_numba(
        sample_energies,
        sample_energies_sorted,
        sorted_energies[0],
        sorted_energies[3],
    )

    for energy_index in range(start, end):
        energy = sample_energies[energy_index]
        for vertex_index in range(4):
            weights[energy_index, vertex_index] = 0.0
            shifted_energies[vertex_index] = sorted_energies[vertex_index] - energy

        if sorted_energies[0] < energy <= sorted_energies[1]:
            accumulate_triangle_weight_sums(
                weights[energy_index],
                sorted_order,
                0,
                shifted_energies,
                strict_energies,
                1.0 / 3.0,
                affine,
                coefficients,
            )
        elif sorted_energies[1] < energy <= sorted_energies[2]:
            accumulate_triangle_weight_sums(
                weights[energy_index],
                sorted_order,
                1,
                shifted_energies,
                strict_energies,
                1.0 / 3.0,
                affine,
                coefficients,
            )
            accumulate_triangle_weight_sums(
                weights[energy_index],
                sorted_order,
                2,
                shifted_energies,
                strict_energies,
                1.0 / 3.0,
                affine,
                coefficients,
            )
        elif sorted_energies[2] < energy < sorted_energies[3]:
            accumulate_triangle_weight_sums(
                weights[energy_index],
                sorted_order,
                3,
                shifted_energies,
                strict_energies,
                1.0 / 3.0,
                affine,
                coefficients,
            )
    return start, end


@njit(cache=True)
def _fermigr_active_energy_window_numba(
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


@njit(cache=True, parallel=True)
def _complex_polarization_weights_on_local_mesh_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: ComplexArray,
    local_point_count: int,
    normalization: int,
) -> ComplexArray:
    tetrahedron_count = occupied_tetra.shape[0]
    source_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    energy_count = sample_energies.shape[0]
    local_weights = np.zeros(
        (local_point_count, energy_count, target_band_count, source_band_count), dtype=np.complex128
    )

    for source_band_index in prange(source_band_count):
        sorted_order = np.empty(4, dtype=np.int64)
        sorted_occupied = np.empty(4, dtype=np.float64)
        sorted_target = np.empty((4, target_band_count), dtype=np.float64)
        outer_weights = np.empty((energy_count, target_band_count, 4), dtype=np.complex128)
        point_weights = np.empty(20, dtype=np.complex128)

        outer_strict = np.empty(4, dtype=np.float64)
        outer_affine = np.empty((4, 4), dtype=np.float64)
        outer_coefficients = np.empty((4, 4), dtype=np.float64)
        transformed_occupied = np.empty(4, dtype=np.float64)
        transformed_target = np.empty((4, target_band_count), dtype=np.float64)

        secondary_weights = np.empty((energy_count, target_band_count, 4), dtype=np.complex128)
        step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_order = np.empty(4, dtype=np.int64)
        secondary_sorted_step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_target = np.empty(4, dtype=np.float64)
        secondary_sorted_occupied = np.empty(4, dtype=np.float64)
        inner_strict = np.empty(4, dtype=np.float64)
        inner_affine = np.empty((4, 4), dtype=np.float64)
        inner_coefficients = np.empty((4, 4), dtype=np.float64)
        energy_differences = np.empty(4, dtype=np.float64)

        for tetrahedron_index in range(tetrahedron_count):
            sort4(
                occupied_tetra[tetrahedron_index, :, source_band_index],
                sorted_order,
                sorted_occupied,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                for target_band_index in range(target_band_count):
                    sorted_target[vertex_index, target_band_index] = target_tetra[
                        tetrahedron_index,
                        source_vertex,
                        target_band_index,
                    ]

            outer_weights[:, :, :] = 0.0j
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_polcmplx_outer_numba(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    target_band_count,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_occupied,
                    transformed_target,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for case_id in (1, 2, 3):
                    _accumulate_small_tetra_polcmplx_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for case_id in (4, 5, 6):
                    _accumulate_small_tetra_polcmplx_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                    )
            elif sorted_occupied[3] <= 0.0:
                _polcmplx_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    target_tetra[tetrahedron_index],
                    sample_energies,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                )
                for energy_index in range(energy_count):
                    for target_band_index in range(target_band_count):
                        for vertex_index in range(4):
                            outer_weights[energy_index, target_band_index, vertex_index] += (
                                secondary_weights[
                                    energy_index,
                                    target_band_index,
                                    vertex_index,
                                ]
                            )

            for energy_index in range(energy_count):
                for target_band_index in range(target_band_count):
                    for point_index in range(20):
                        total = 0.0j
                        for vertex_index in range(4):
                            total += (
                                outer_weights[energy_index, target_band_index, vertex_index]
                                * tetrahedron_weight_matrix[
                                    vertex_index,
                                    point_index,
                                ]
                            )
                        point_weights[point_index] = total
                    for point_index in range(20):
                        local_weights[
                            local_point_indices[tetrahedron_index, point_index],
                            energy_index,
                            target_band_index,
                            source_band_index,
                        ] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True, parallel=True)
def _complex_polarization_weights_on_local_mesh_pair_parallel_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energy_real: FloatArray,
    sample_energy_imag: FloatArray,
    local_point_count: int,
    normalization: int,
) -> ComplexArray:
    tetrahedron_count = occupied_tetra.shape[0]
    source_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    energy_count = sample_energy_real.shape[0]
    pair_count = source_band_count * target_band_count
    local_weights = np.zeros(
        (local_point_count, energy_count, target_band_count, source_band_count), dtype=np.complex128
    )

    for pair_index in prange(pair_count):
        source_band_index = pair_index // target_band_count
        target_band_index = pair_index % target_band_count
        pair_weights = np.zeros((local_point_count, energy_count), dtype=np.complex128)

        sorted_order = np.empty(4, dtype=np.int64)
        sorted_occupied = np.empty(4, dtype=np.float64)
        sorted_target = np.empty(4, dtype=np.float64)
        full_target = np.empty(4, dtype=np.float64)

        outer_strict = np.empty(4, dtype=np.float64)
        outer_affine = np.empty((4, 4), dtype=np.float64)
        outer_coefficients = np.empty((4, 4), dtype=np.float64)
        transformed_occupied = np.empty(4, dtype=np.float64)
        transformed_target = np.empty(4, dtype=np.float64)
        outer_point_coefficients = np.empty((4, 20), dtype=np.float64)

        step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_order = np.empty(4, dtype=np.int64)
        secondary_sorted_step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_target = np.empty(4, dtype=np.float64)
        secondary_sorted_occupied = np.empty(4, dtype=np.float64)
        inner_strict = np.empty(4, dtype=np.float64)
        inner_affine = np.empty((4, 4), dtype=np.float64)
        inner_coefficients = np.empty((4, 4), dtype=np.float64)
        energy_differences = np.empty(4, dtype=np.float64)
        point_coefficients = np.empty((4, 20), dtype=np.float64)
        sample_weight_real = np.empty(4, dtype=np.float64)
        sample_weight_imag = np.empty(4, dtype=np.float64)

        for tetrahedron_index in range(tetrahedron_count):
            local_points = local_point_indices[tetrahedron_index]
            sort4(
                occupied_tetra[tetrahedron_index, :, source_band_index],
                sorted_order,
                sorted_occupied,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                sorted_target[vertex_index] = target_tetra[
                    tetrahedron_index,
                    source_vertex,
                    target_band_index,
                ]
                full_target[vertex_index] = target_tetra[
                    tetrahedron_index,
                    vertex_index,
                    target_band_index,
                ]

            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_polcmplx_outer_pair_direct_numba(
                    pair_weights,
                    local_points,
                    tetrahedron_weight_matrix,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    sample_energy_real,
                    sample_energy_imag,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_occupied,
                    transformed_target,
                    outer_point_coefficients,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    point_coefficients,
                    sample_weight_real,
                    sample_weight_imag,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for case_id in (1, 2, 3):
                    _accumulate_small_tetra_polcmplx_outer_pair_direct_numba(
                        pair_weights,
                        local_points,
                        tetrahedron_weight_matrix,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energy_real,
                        sample_energy_imag,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        outer_point_coefficients,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        point_coefficients,
                        sample_weight_real,
                        sample_weight_imag,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for case_id in (4, 5, 6):
                    _accumulate_small_tetra_polcmplx_outer_pair_direct_numba(
                        pair_weights,
                        local_points,
                        tetrahedron_weight_matrix,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energy_real,
                        sample_energy_imag,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        outer_point_coefficients,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        point_coefficients,
                        sample_weight_real,
                        sample_weight_imag,
                    )
            elif sorted_occupied[3] <= 0.0:
                _polcmplx_secondary_pair_direct_numba(
                    pair_weights,
                    local_points,
                    tetrahedron_weight_matrix,
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    full_target,
                    sample_energy_real,
                    sample_energy_imag,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    point_coefficients,
                    sample_weight_real,
                    sample_weight_imag,
                )

        for local_point_index in range(local_point_count):
            for energy_index in range(energy_count):
                local_weights[
                    local_point_index,
                    energy_index,
                    target_band_index,
                    source_band_index,
                ] = pair_weights[local_point_index, energy_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True)
def _accumulate_small_tetra_polcmplx_outer_pair_direct_numba(
    pair_weights: ComplexArray,
    local_points: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    case_id: int,
    sorted_order: npt.NDArray[np.int64],
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energy_real: FloatArray,
    sample_energy_imag: FloatArray,
    outer_strict: FloatArray,
    outer_affine: FloatArray,
    outer_coefficients: FloatArray,
    transformed_occupied: FloatArray,
    transformed_target: FloatArray,
    outer_point_coefficients: FloatArray,
    step_energies: FloatArray,
    secondary_sorted_order: npt.NDArray[np.int64],
    secondary_sorted_step_energies: FloatArray,
    secondary_sorted_target: FloatArray,
    secondary_sorted_occupied: FloatArray,
    inner_strict: FloatArray,
    inner_affine: FloatArray,
    inner_coefficients: FloatArray,
    energy_differences: FloatArray,
    point_coefficients: FloatArray,
    sample_weight_real: FloatArray,
    sample_weight_imag: FloatArray,
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_occupied,
        outer_strict,
        outer_affine,
        outer_coefficients,
    )
    if volume_factor <= 1.0e-8:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += outer_coefficients[row_index, column_index] * sorted_occupied[column_index]
        transformed_occupied[row_index] = total

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += outer_coefficients[row_index, column_index] * sorted_target[column_index]
        transformed_target[row_index] = total

    for row_index in range(4):
        for point_index in range(20):
            total = 0.0
            for column_index in range(4):
                total += (
                    outer_coefficients[row_index, column_index]
                    * tetrahedron_weight_matrix[
                        sorted_order[column_index],
                        point_index,
                    ]
                )
            outer_point_coefficients[row_index, point_index] = volume_factor * total

    _polcmplx_secondary_pair_direct_numba(
        pair_weights,
        local_points,
        outer_point_coefficients,
        transformed_occupied,
        transformed_target,
        sample_energy_real,
        sample_energy_imag,
        step_energies,
        secondary_sorted_order,
        secondary_sorted_step_energies,
        secondary_sorted_target,
        secondary_sorted_occupied,
        inner_strict,
        inner_affine,
        inner_coefficients,
        energy_differences,
        point_coefficients,
        sample_weight_real,
        sample_weight_imag,
    )


@njit(cache=True)
def _polcmplx_secondary_pair_direct_numba(
    pair_weights: ComplexArray,
    local_points: npt.NDArray[np.int64],
    point_coefficients_base: FloatArray,
    occupied_vertices: FloatArray,
    target_vertices: FloatArray,
    sample_energy_real: FloatArray,
    sample_energy_imag: FloatArray,
    step_energies: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_step_energies: FloatArray,
    sorted_target: FloatArray,
    sorted_occupied: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    point_coefficients: FloatArray,
    sample_weight_real: FloatArray,
    sample_weight_imag: FloatArray,
) -> None:
    for vertex_index in range(4):
        step_energies[vertex_index] = -target_vertices[vertex_index]
    sort4(step_energies, sorted_order, sorted_step_energies)
    for vertex_index in range(4):
        source_vertex = sorted_order[vertex_index]
        sorted_target[vertex_index] = target_vertices[source_vertex]
        sorted_occupied[vertex_index] = occupied_vertices[source_vertex]

    if (sorted_step_energies[0] <= 0.0 < sorted_step_energies[1]) or (
        sorted_step_energies[0] < 0.0 <= sorted_step_energies[1]
    ):
        _accumulate_small_tetra_polcmplx_inner_pair_direct_numba(
            pair_weights,
            local_points,
            point_coefficients_base,
            0,
            sorted_order,
            sorted_step_energies,
            sorted_occupied,
            sorted_target,
            sample_energy_real,
            sample_energy_imag,
            strict_energies,
            affine,
            coefficients,
            energy_differences,
            point_coefficients,
            sample_weight_real,
            sample_weight_imag,
        )
    elif (sorted_step_energies[1] <= 0.0 < sorted_step_energies[2]) or (
        sorted_step_energies[1] < 0.0 <= sorted_step_energies[2]
    ):
        for case_id in (1, 2, 3):
            _accumulate_small_tetra_polcmplx_inner_pair_direct_numba(
                pair_weights,
                local_points,
                point_coefficients_base,
                case_id,
                sorted_order,
                sorted_step_energies,
                sorted_occupied,
                sorted_target,
                sample_energy_real,
                sample_energy_imag,
                strict_energies,
                affine,
                coefficients,
                energy_differences,
                point_coefficients,
                sample_weight_real,
                sample_weight_imag,
            )
    elif (sorted_step_energies[2] <= 0.0 < sorted_step_energies[3]) or (
        sorted_step_energies[2] < 0.0 <= sorted_step_energies[3]
    ):
        for case_id in (4, 5, 6):
            _accumulate_small_tetra_polcmplx_inner_pair_direct_numba(
                pair_weights,
                local_points,
                point_coefficients_base,
                case_id,
                sorted_order,
                sorted_step_energies,
                sorted_occupied,
                sorted_target,
                sample_energy_real,
                sample_energy_imag,
                strict_energies,
                affine,
                coefficients,
                energy_differences,
                point_coefficients,
                sample_weight_real,
                sample_weight_imag,
            )
    elif sorted_step_energies[3] <= 0.0:
        for vertex_index in range(4):
            energy_differences[vertex_index] = sorted_target[vertex_index] - sorted_occupied[vertex_index]
        for row_index in range(4):
            for point_index in range(20):
                point_coefficients[row_index, point_index] = point_coefficients_base[
                    sorted_order[row_index],
                    point_index,
                ]
        _accumulate_polcmplx_point_weights_numba(
            pair_weights,
            local_points,
            sample_energy_real,
            sample_energy_imag,
            energy_differences,
            point_coefficients,
            sample_weight_real,
            sample_weight_imag,
        )


@njit(cache=True)
def _accumulate_small_tetra_polcmplx_inner_pair_direct_numba(
    pair_weights: ComplexArray,
    local_points: npt.NDArray[np.int64],
    point_coefficients_base: FloatArray,
    case_id: int,
    sorted_order: npt.NDArray[np.int64],
    sorted_step_energies: FloatArray,
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energy_real: FloatArray,
    sample_energy_imag: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    point_coefficients: FloatArray,
    sample_weight_real: FloatArray,
    sample_weight_imag: FloatArray,
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_step_energies,
        strict_energies,
        affine,
        coefficients,
    )
    if volume_factor <= 1.0e-8:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += coefficients[row_index, column_index] * (
                sorted_target[column_index] - sorted_occupied[column_index]
            )
        energy_differences[row_index] = total

    for row_index in range(4):
        for point_index in range(20):
            total = 0.0
            for column_index in range(4):
                total += (
                    coefficients[row_index, column_index]
                    * point_coefficients_base[
                        sorted_order[column_index],
                        point_index,
                    ]
                )
            point_coefficients[row_index, point_index] = volume_factor * total

    _accumulate_polcmplx_point_weights_numba(
        pair_weights,
        local_points,
        sample_energy_real,
        sample_energy_imag,
        energy_differences,
        point_coefficients,
        sample_weight_real,
        sample_weight_imag,
    )


@njit(cache=True)
def _accumulate_polcmplx_point_weights_numba(
    pair_weights: ComplexArray,
    local_points: npt.NDArray[np.int64],
    sample_energy_real: FloatArray,
    sample_energy_imag: FloatArray,
    energy_differences: FloatArray,
    point_coefficients: FloatArray,
    sample_weight_real: FloatArray,
    sample_weight_imag: FloatArray,
) -> None:
    energy_count = sample_energy_real.shape[0]
    for energy_index in range(energy_count):
        real_part = sample_energy_real[energy_index]
        imag_part = sample_energy_imag[energy_index]
        for vertex_index in range(4):
            denominator_real = energy_differences[vertex_index] + real_part
            denominator_imag = imag_part
            scale = 0.25 / (
                denominator_real * denominator_real + denominator_imag * denominator_imag
            )
            sample_weight_real[vertex_index] = denominator_real * scale
            sample_weight_imag[vertex_index] = -denominator_imag * scale
        for point_index in range(20):
            total_real = 0.0
            total_imag = 0.0
            for vertex_index in range(4):
                coefficient = point_coefficients[vertex_index, point_index]
                total_real += coefficient * sample_weight_real[vertex_index]
                total_imag += coefficient * sample_weight_imag[vertex_index]
            pair_weights[local_points[point_index], energy_index] += total_real + 1j * total_imag


@njit(cache=True)
def _accumulate_small_tetra_polcmplx_outer_numba(
    outer_weights: ComplexArray,
    case_id: int,
    sorted_order: npt.NDArray[np.int64],
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energies: ComplexArray,
    target_band_count: int,
    outer_strict: FloatArray,
    outer_affine: FloatArray,
    outer_coefficients: FloatArray,
    transformed_occupied: FloatArray,
    transformed_target: FloatArray,
    secondary_weights: ComplexArray,
    step_energies: FloatArray,
    secondary_sorted_order: npt.NDArray[np.int64],
    secondary_sorted_step_energies: FloatArray,
    secondary_sorted_target: FloatArray,
    secondary_sorted_occupied: FloatArray,
    inner_strict: FloatArray,
    inner_affine: FloatArray,
    inner_coefficients: FloatArray,
    energy_differences: FloatArray,
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_occupied,
        outer_strict,
        outer_affine,
        outer_coefficients,
    )
    if volume_factor <= 1.0e-8:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += outer_coefficients[row_index, column_index] * sorted_occupied[column_index]
        transformed_occupied[row_index] = total

    for row_index in range(4):
        for target_band_index in range(target_band_count):
            total = 0.0
            for column_index in range(4):
                total += (
                    outer_coefficients[row_index, column_index]
                    * sorted_target[column_index, target_band_index]
                )
            transformed_target[row_index, target_band_index] = total

    _polcmplx_secondary_weights_numba(
        transformed_occupied,
        transformed_target,
        sample_energies,
        secondary_weights,
        step_energies,
        secondary_sorted_order,
        secondary_sorted_step_energies,
        secondary_sorted_target,
        secondary_sorted_occupied,
        inner_strict,
        inner_affine,
        inner_coefficients,
        energy_differences,
    )

    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        for target_band_index in range(target_band_count):
            for column_index in range(4):
                total = 0.0j
                for row_index in range(4):
                    total += (
                        secondary_weights[energy_index, target_band_index, row_index]
                        * outer_coefficients[
                            row_index,
                            column_index,
                        ]
                    )
                outer_weights[energy_index, target_band_index, sorted_order[column_index]] += (
                    volume_factor * total
                )


@njit(cache=True)
def _polcmplx_secondary_weights_numba(
    occupied_vertices: FloatArray,
    target_vertices: FloatArray,
    sample_energies: ComplexArray,
    weights: ComplexArray,
    step_energies: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_step_energies: FloatArray,
    sorted_target: FloatArray,
    sorted_occupied: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
) -> None:
    target_band_count = target_vertices.shape[1]
    energy_count = sample_energies.shape[0]
    weights[:, :, :] = 0.0j

    for target_band_index in range(target_band_count):
        for vertex_index in range(4):
            step_energies[vertex_index] = -target_vertices[vertex_index, target_band_index]
        sort4(step_energies, sorted_order, sorted_step_energies)
        for vertex_index in range(4):
            source_vertex = sorted_order[vertex_index]
            sorted_target[vertex_index] = target_vertices[source_vertex, target_band_index]
            sorted_occupied[vertex_index] = occupied_vertices[source_vertex]

        if (sorted_step_energies[0] <= 0.0 < sorted_step_energies[1]) or (
            sorted_step_energies[0] < 0.0 <= sorted_step_energies[1]
        ):
            _accumulate_small_tetra_polcmplx_inner_numba(
                weights,
                target_band_index,
                sorted_order,
                0,
                sorted_step_energies,
                sorted_occupied,
                sorted_target,
                sample_energies,
                strict_energies,
                affine,
                coefficients,
                energy_differences,
            )
        elif (sorted_step_energies[1] <= 0.0 < sorted_step_energies[2]) or (
            sorted_step_energies[1] < 0.0 <= sorted_step_energies[2]
        ):
            for case_id in (1, 2, 3):
                _accumulate_small_tetra_polcmplx_inner_numba(
                    weights,
                    target_band_index,
                    sorted_order,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                )
        elif (sorted_step_energies[2] <= 0.0 < sorted_step_energies[3]) or (
            sorted_step_energies[2] < 0.0 <= sorted_step_energies[3]
        ):
            for case_id in (4, 5, 6):
                _accumulate_small_tetra_polcmplx_inner_numba(
                    weights,
                    target_band_index,
                    sorted_order,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                )
        elif sorted_step_energies[3] <= 0.0:
            for vertex_index in range(4):
                energy_differences[vertex_index] = (
                    sorted_target[vertex_index] - sorted_occupied[vertex_index]
                )
            for energy_index in range(energy_count):
                for vertex_index in range(4):
                    weights[energy_index, target_band_index, sorted_order[vertex_index]] += 0.25 / (
                        energy_differences[vertex_index] + sample_energies[energy_index]
                    )


@njit(cache=True)
def _accumulate_small_tetra_polcmplx_inner_numba(
    weights: ComplexArray,
    target_band_index: int,
    sorted_order: npt.NDArray[np.int64],
    case_id: int,
    sorted_step_energies: FloatArray,
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energies: ComplexArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_step_energies,
        strict_energies,
        affine,
        coefficients,
    )
    if volume_factor <= 1.0e-8:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += coefficients[row_index, column_index] * (
                sorted_target[column_index] - sorted_occupied[column_index]
            )
        energy_differences[row_index] = total

    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        sample_energy = sample_energies[energy_index]
        for column_index in range(4):
            total = 0.0j
            for row_index in range(4):
                total += 0.25 / (energy_differences[row_index] + sample_energy) * coefficients[
                    row_index,
                    column_index,
                ]
            weights[energy_index, target_band_index, sorted_order[column_index]] += (
                volume_factor * total
            )
