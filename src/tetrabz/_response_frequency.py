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
    if pair_count >= 16 and target_tetra.shape[2] >= 4:
        return _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            mesh.tetrahedron_weight_matrix,
            occupied_tetra,
            target_tetra,
            sample_energies,
            mesh.local_point_count,
            normalization,
        )
    return _fermi_golden_rule_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        occupied_tetra,
        target_tetra,
        sample_energies,
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
    pair_count = occupied_tetra.shape[2] * target_tetra.shape[2]
    if pair_count >= 16 and target_tetra.shape[2] >= 4:
        return _complex_polarization_weights_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            mesh.tetrahedron_weight_matrix,
            occupied_tetra,
            target_tetra,
            sample_energies,
            mesh.local_point_count,
            normalization,
        )
    return _complex_polarization_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        occupied_tetra,
        target_tetra,
        sample_energies,
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
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_fermigr_outer_numba(
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
                    )
            elif sorted_occupied[3] <= 0.0:
                _fermigr_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    target_tetra[tetrahedron_index],
                    sample_energies,
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
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_fermigr_outer_numba(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
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
                    )
            elif sorted_occupied[3] <= 0.0:
                _fermigr_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    full_target,
                    sample_energies,
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
                )
                for energy_index in range(energy_count):
                    for vertex_index in range(4):
                        outer_weights[energy_index, 0, vertex_index] += secondary_weights[
                            energy_index,
                            0,
                            vertex_index,
                        ]

            for energy_index in range(energy_count):
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
    )

    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        for target_band_index in range(target_band_count):
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

        if (sorted_step_energies[0] <= 0.0 < sorted_step_energies[1]) or (
            sorted_step_energies[0] < 0.0 <= sorted_step_energies[1]
        ):
            _accumulate_small_tetra_fermigr_inner_numba(
                sorted_weights,
                0,
                sorted_step_energies,
                sorted_occupied,
                sorted_target,
                sample_energies,
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
                _accumulate_small_tetra_fermigr_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
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
        elif (sorted_step_energies[2] <= 0.0 < sorted_step_energies[3]) or (
            sorted_step_energies[2] < 0.0 <= sorted_step_energies[3]
        ):
            for case_id in (4, 5, 6):
                _accumulate_small_tetra_fermigr_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
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
        elif sorted_step_energies[3] <= 0.0:
            for vertex_index in range(4):
                energy_differences[vertex_index] = (
                    sorted_target[vertex_index] - sorted_occupied[vertex_index]
                )
            _fermigr_delta_weights_numba(
                sample_energies,
                energy_differences,
                delta_weights,
                delta_sorted_order,
                delta_sorted_energies,
                delta_shifted_energies,
                triangle_strict,
                triangle_affine,
                triangle_coefficients,
            )
            for energy_index in range(energy_count):
                for vertex_index in range(4):
                    sorted_weights[energy_index, vertex_index] += delta_weights[
                        energy_index, vertex_index
                    ]

        for energy_index in range(energy_count):
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

    _fermigr_delta_weights_numba(
        sample_energies,
        energy_differences,
        delta_weights,
        delta_sorted_order,
        delta_sorted_energies,
        delta_shifted_energies,
        triangle_strict,
        triangle_affine,
        triangle_coefficients,
    )

    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        for column_index in range(4):
            total = 0.0
            for row_index in range(4):
                total += (
                    delta_weights[energy_index, row_index] * coefficients[row_index, column_index]
                )
            weights[energy_index, column_index] += volume_factor * total


@njit(cache=True)
def _fermigr_delta_weights_numba(
    sample_energies: FloatArray,
    energy_differences: FloatArray,
    weights: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_energies: FloatArray,
    shifted_energies: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
) -> None:
    sort4(energy_differences, sorted_order, sorted_energies)
    energy_count = sample_energies.shape[0]

    for energy_index in range(energy_count):
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
        secondary_sorted_weights = np.empty((energy_count, 4), dtype=np.complex128)
        inner_strict = np.empty(4, dtype=np.float64)
        inner_affine = np.empty((4, 4), dtype=np.float64)
        inner_coefficients = np.empty((4, 4), dtype=np.float64)
        energy_differences = np.empty(4, dtype=np.float64)
        sample_weights = np.empty((energy_count, 4), dtype=np.complex128)

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
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    sample_weights,
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
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        sample_weights,
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
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        sample_weights,
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
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    sample_weights,
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
    sample_energies: ComplexArray,
    local_point_count: int,
    normalization: int,
) -> ComplexArray:
    tetrahedron_count = occupied_tetra.shape[0]
    source_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    energy_count = sample_energies.shape[0]
    pair_count = source_band_count * target_band_count
    local_weights = np.zeros(
        (local_point_count, energy_count, target_band_count, source_band_count), dtype=np.complex128
    )

    for pair_index in prange(pair_count):
        source_band_index = pair_index // target_band_count
        target_band_index = pair_index % target_band_count

        sorted_order = np.empty(4, dtype=np.int64)
        sorted_occupied = np.empty(4, dtype=np.float64)
        sorted_target = np.empty((4, 1), dtype=np.float64)
        full_target = np.empty((4, 1), dtype=np.float64)
        outer_weights = np.empty((energy_count, 1, 4), dtype=np.complex128)
        point_weights = np.empty(20, dtype=np.complex128)

        outer_strict = np.empty(4, dtype=np.float64)
        outer_affine = np.empty((4, 4), dtype=np.float64)
        outer_coefficients = np.empty((4, 4), dtype=np.float64)
        transformed_occupied = np.empty(4, dtype=np.float64)
        transformed_target = np.empty((4, 1), dtype=np.float64)

        secondary_weights = np.empty((energy_count, 1, 4), dtype=np.complex128)
        step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_order = np.empty(4, dtype=np.int64)
        secondary_sorted_step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_target = np.empty(4, dtype=np.float64)
        secondary_sorted_occupied = np.empty(4, dtype=np.float64)
        secondary_sorted_weights = np.empty((energy_count, 4), dtype=np.complex128)
        inner_strict = np.empty(4, dtype=np.float64)
        inner_affine = np.empty((4, 4), dtype=np.float64)
        inner_coefficients = np.empty((4, 4), dtype=np.float64)
        energy_differences = np.empty(4, dtype=np.float64)
        sample_weights = np.empty((energy_count, 4), dtype=np.complex128)

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

            outer_weights[:, :, :] = 0.0j
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_polcmplx_outer_numba(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
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
                    sample_weights,
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
                        sample_weights,
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
                        sample_weights,
                    )
            elif sorted_occupied[3] <= 0.0:
                _polcmplx_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    full_target,
                    sample_energies,
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
                    sample_weights,
                )
                for energy_index in range(energy_count):
                    for vertex_index in range(4):
                        outer_weights[energy_index, 0, vertex_index] += secondary_weights[
                            energy_index,
                            0,
                            vertex_index,
                        ]

            for energy_index in range(energy_count):
                for point_index in range(20):
                    total = 0.0j
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
    secondary_sorted_weights: ComplexArray,
    inner_strict: FloatArray,
    inner_affine: FloatArray,
    inner_coefficients: FloatArray,
    energy_differences: FloatArray,
    sample_weights: ComplexArray,
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
        secondary_sorted_weights,
        inner_strict,
        inner_affine,
        inner_coefficients,
        energy_differences,
        sample_weights,
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
    sorted_weights: ComplexArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    sample_weights: ComplexArray,
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
        sorted_weights[:, :] = 0.0j

        if (sorted_step_energies[0] <= 0.0 < sorted_step_energies[1]) or (
            sorted_step_energies[0] < 0.0 <= sorted_step_energies[1]
        ):
            _accumulate_small_tetra_polcmplx_inner_numba(
                sorted_weights,
                0,
                sorted_step_energies,
                sorted_occupied,
                sorted_target,
                sample_energies,
                strict_energies,
                affine,
                coefficients,
                energy_differences,
                sample_weights,
            )
        elif (sorted_step_energies[1] <= 0.0 < sorted_step_energies[2]) or (
            sorted_step_energies[1] < 0.0 <= sorted_step_energies[2]
        ):
            for case_id in (1, 2, 3):
                _accumulate_small_tetra_polcmplx_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                    sample_weights,
                )
        elif (sorted_step_energies[2] <= 0.0 < sorted_step_energies[3]) or (
            sorted_step_energies[2] < 0.0 <= sorted_step_energies[3]
        ):
            for case_id in (4, 5, 6):
                _accumulate_small_tetra_polcmplx_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                    sample_weights,
                )
        elif sorted_step_energies[3] <= 0.0:
            for vertex_index in range(4):
                energy_differences[vertex_index] = (
                    sorted_target[vertex_index] - sorted_occupied[vertex_index]
                )
            _polcmplx_sample_weights_numba(sample_energies, energy_differences, sample_weights)
            for energy_index in range(energy_count):
                for vertex_index in range(4):
                    sorted_weights[energy_index, vertex_index] += sample_weights[
                        energy_index, vertex_index
                    ]

        for energy_index in range(energy_count):
            for vertex_index in range(4):
                weights[energy_index, target_band_index, sorted_order[vertex_index]] = (
                    sorted_weights[
                        energy_index,
                        vertex_index,
                    ]
                )


@njit(cache=True)
def _accumulate_small_tetra_polcmplx_inner_numba(
    weights: ComplexArray,
    case_id: int,
    sorted_step_energies: FloatArray,
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energies: ComplexArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    sample_weights: ComplexArray,
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

    _polcmplx_sample_weights_numba(sample_energies, energy_differences, sample_weights)

    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        for column_index in range(4):
            total = 0.0j
            for row_index in range(4):
                total += (
                    sample_weights[energy_index, row_index] * coefficients[row_index, column_index]
                )
            weights[energy_index, column_index] += volume_factor * total


@njit(cache=True)
def _polcmplx_sample_weights_numba(
    sample_energies: ComplexArray,
    energy_differences: FloatArray,
    weights: ComplexArray,
) -> None:
    # The live legacy polcmplx3 path is just the direct vertex average 0.25 / (de + z).
    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        for vertex_index in range(4):
            weights[energy_index, vertex_index] = 0.25 / (
                energy_differences[vertex_index] + sample_energies[energy_index]
            )
