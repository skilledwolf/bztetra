from __future__ import annotations

import numpy as np
from numba import njit

from ._triangle_kernels import sort3


MAX_POLYGON_VERTICES = 6
GEOMETRY_EPS = 1.0e-12
ENERGY_ATOL = 1.0e-12
MACHINE_EPS = np.finfo(np.float64).eps


def _phase_space_overlap_triangle_weights(
    occupied_vertices: np.ndarray,
    target_vertices: np.ndarray,
    triangle_area: float,
) -> np.ndarray:
    occupied = np.asarray(occupied_vertices, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    weights = np.zeros(3, dtype=np.float64)
    polygon_a = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    polygon_b = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    _phase_space_overlap_parent_weights_numba(
        weights,
        occupied,
        target,
        triangle_area,
        polygon_a,
        polygon_b,
    )
    return weights


def _nesting_function_triangle_weights(
    occupied_vertices: np.ndarray,
    target_vertices: np.ndarray,
    triangle_area: float,
) -> np.ndarray:
    occupied = np.asarray(occupied_vertices, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    weights = np.zeros(3, dtype=np.float64)
    _nesting_parent_weights_numba(weights, occupied, target, triangle_area)
    return weights


def _static_polarization_triangle_weights(
    occupied_vertices: np.ndarray,
    target_vertices: np.ndarray,
    triangle_area: float,
) -> np.ndarray:
    occupied = np.asarray(occupied_vertices, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    weights = np.zeros(3, dtype=np.float64)
    polygon_a = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    polygon_b = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    sorted_order = np.empty(3, dtype=np.int64)
    sorted_delta = np.empty(3, dtype=np.float64)
    sorted_weights = np.empty(3, dtype=np.float64)
    _static_polarization_parent_weights_numba(
        weights,
        occupied,
        target,
        triangle_area,
        polygon_a,
        polygon_b,
        sorted_order,
        sorted_delta,
        sorted_weights,
    )
    return weights


def _fermi_golden_rule_triangle_weights(
    occupied_vertices: np.ndarray,
    target_vertices: np.ndarray,
    energies: np.ndarray,
    triangle_area: float,
) -> np.ndarray:
    occupied = np.asarray(occupied_vertices, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    samples = np.asarray(energies, dtype=np.float64)
    weights = np.zeros((samples.size, 3), dtype=np.float64)
    polygon_a = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    polygon_b = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    sorted_order = np.empty(3, dtype=np.int64)
    sorted_delta = np.empty(3, dtype=np.float64)
    sorted_weights = np.empty(3, dtype=np.float64)
    _fermi_golden_rule_parent_weights_numba(
        weights,
        occupied,
        target,
        samples,
        triangle_area,
        polygon_a,
        polygon_b,
        sorted_order,
        sorted_delta,
        sorted_weights,
    )
    return weights


def _complex_polarization_triangle_weights(
    occupied_vertices: np.ndarray,
    target_vertices: np.ndarray,
    energies: np.ndarray,
    triangle_area: float,
) -> np.ndarray:
    occupied = np.asarray(occupied_vertices, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    samples = np.asarray(energies, dtype=np.complex128)
    weights = np.zeros((samples.size, 3), dtype=np.complex128)
    polygon_a = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    polygon_b = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    sorted_order = np.empty(3, dtype=np.int64)
    sorted_delta = np.empty(3, dtype=np.float64)
    sorted_weights = np.empty(3, dtype=np.complex128)
    _complex_polarization_parent_weights_numba(
        weights,
        occupied,
        target,
        samples,
        triangle_area,
        polygon_a,
        polygon_b,
        sorted_order,
        sorted_delta,
        sorted_weights,
    )
    return weights


@njit(cache=True)
def _phase_space_overlap_weights_on_local_mesh_numba(
    local_point_indices,
    occupied_triangles,
    target_triangles,
    local_point_count,
    triangle_area,
):
    source_band_count = occupied_triangles.shape[2]
    target_band_count = target_triangles.shape[2]
    local_weights = np.zeros((local_point_count, target_band_count, source_band_count), dtype=np.float64)
    parent_weights = np.empty(3, dtype=np.float64)
    polygon_a = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    polygon_b = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)

    for triangle_index in range(occupied_triangles.shape[0]):
        for source_band_index in range(source_band_count):
            occupied_vertices = occupied_triangles[triangle_index, :, source_band_index]
            for target_band_index in range(target_band_count):
                target_vertices = target_triangles[triangle_index, :, target_band_index]
                _phase_space_overlap_parent_weights_numba(
                    parent_weights,
                    occupied_vertices,
                    target_vertices,
                    triangle_area,
                    polygon_a,
                    polygon_b,
                )
                for vertex_index in range(3):
                    local_weights[
                        local_point_indices[triangle_index, vertex_index],
                        target_band_index,
                        source_band_index,
                    ] += parent_weights[vertex_index]

    return local_weights


@njit(cache=True)
def _nesting_function_weights_on_local_mesh_numba(
    local_point_indices,
    occupied_triangles,
    target_triangles,
    local_point_count,
    triangle_area,
):
    source_band_count = occupied_triangles.shape[2]
    target_band_count = target_triangles.shape[2]
    local_weights = np.zeros((local_point_count, target_band_count, source_band_count), dtype=np.float64)
    parent_weights = np.empty(3, dtype=np.float64)

    for triangle_index in range(occupied_triangles.shape[0]):
        for source_band_index in range(source_band_count):
            occupied_vertices = occupied_triangles[triangle_index, :, source_band_index]
            for target_band_index in range(target_band_count):
                target_vertices = target_triangles[triangle_index, :, target_band_index]
                _nesting_parent_weights_numba(
                    parent_weights,
                    occupied_vertices,
                    target_vertices,
                    triangle_area,
                )
                for vertex_index in range(3):
                    local_weights[
                        local_point_indices[triangle_index, vertex_index],
                        target_band_index,
                        source_band_index,
                    ] += parent_weights[vertex_index]

    return local_weights


@njit(cache=True)
def _static_polarization_weights_on_local_mesh_numba(
    local_point_indices,
    occupied_triangles,
    target_triangles,
    local_point_count,
    triangle_area,
):
    source_band_count = occupied_triangles.shape[2]
    target_band_count = target_triangles.shape[2]
    local_weights = np.zeros((local_point_count, target_band_count, source_band_count), dtype=np.float64)
    parent_weights = np.empty(3, dtype=np.float64)
    polygon_a = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    polygon_b = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    sorted_order = np.empty(3, dtype=np.int64)
    sorted_delta = np.empty(3, dtype=np.float64)
    sorted_weights = np.empty(3, dtype=np.float64)

    for triangle_index in range(occupied_triangles.shape[0]):
        for source_band_index in range(source_band_count):
            occupied_vertices = occupied_triangles[triangle_index, :, source_band_index]
            for target_band_index in range(target_band_count):
                target_vertices = target_triangles[triangle_index, :, target_band_index]
                _static_polarization_parent_weights_numba(
                    parent_weights,
                    occupied_vertices,
                    target_vertices,
                    triangle_area,
                    polygon_a,
                    polygon_b,
                    sorted_order,
                    sorted_delta,
                    sorted_weights,
                )
                for vertex_index in range(3):
                    local_weights[
                        local_point_indices[triangle_index, vertex_index],
                        target_band_index,
                        source_band_index,
                    ] += parent_weights[vertex_index]

    return local_weights


@njit(cache=True)
def _fermi_golden_rule_weights_on_local_mesh_numba(
    local_point_indices,
    occupied_triangles,
    target_triangles,
    sample_energies,
    local_point_count,
    triangle_area,
):
    source_band_count = occupied_triangles.shape[2]
    target_band_count = target_triangles.shape[2]
    energy_count = sample_energies.shape[0]
    local_weights = np.zeros(
        (local_point_count, energy_count, target_band_count, source_band_count),
        dtype=np.float64,
    )
    parent_weights = np.empty((energy_count, 3), dtype=np.float64)
    polygon_a = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    polygon_b = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    sorted_order = np.empty(3, dtype=np.int64)
    sorted_delta = np.empty(3, dtype=np.float64)
    sorted_weights = np.empty(3, dtype=np.float64)

    for triangle_index in range(occupied_triangles.shape[0]):
        for source_band_index in range(source_band_count):
            occupied_vertices = occupied_triangles[triangle_index, :, source_band_index]
            for target_band_index in range(target_band_count):
                target_vertices = target_triangles[triangle_index, :, target_band_index]
                _fermi_golden_rule_parent_weights_numba(
                    parent_weights,
                    occupied_vertices,
                    target_vertices,
                    sample_energies,
                    triangle_area,
                    polygon_a,
                    polygon_b,
                    sorted_order,
                    sorted_delta,
                    sorted_weights,
                )
                for energy_index in range(energy_count):
                    for vertex_index in range(3):
                        local_weights[
                            local_point_indices[triangle_index, vertex_index],
                            energy_index,
                            target_band_index,
                            source_band_index,
                        ] += parent_weights[energy_index, vertex_index]

    return local_weights


@njit(cache=True)
def _complex_polarization_weights_on_local_mesh_numba(
    local_point_indices,
    occupied_triangles,
    target_triangles,
    sample_energies,
    local_point_count,
    triangle_area,
):
    source_band_count = occupied_triangles.shape[2]
    target_band_count = target_triangles.shape[2]
    energy_count = sample_energies.shape[0]
    local_weights = np.zeros(
        (local_point_count, energy_count, target_band_count, source_band_count),
        dtype=np.complex128,
    )
    parent_weights = np.empty((energy_count, 3), dtype=np.complex128)
    polygon_a = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    polygon_b = np.empty((MAX_POLYGON_VERTICES, 3), dtype=np.float64)
    sorted_order = np.empty(3, dtype=np.int64)
    sorted_delta = np.empty(3, dtype=np.float64)
    sorted_weights = np.empty(3, dtype=np.complex128)

    for triangle_index in range(occupied_triangles.shape[0]):
        for source_band_index in range(source_band_count):
            occupied_vertices = occupied_triangles[triangle_index, :, source_band_index]
            for target_band_index in range(target_band_count):
                target_vertices = target_triangles[triangle_index, :, target_band_index]
                _complex_polarization_parent_weights_numba(
                    parent_weights,
                    occupied_vertices,
                    target_vertices,
                    sample_energies,
                    triangle_area,
                    polygon_a,
                    polygon_b,
                    sorted_order,
                    sorted_delta,
                    sorted_weights,
                )
                for energy_index in range(energy_count):
                    for vertex_index in range(3):
                        local_weights[
                            local_point_indices[triangle_index, vertex_index],
                            energy_index,
                            target_band_index,
                            source_band_index,
                        ] += parent_weights[energy_index, vertex_index]

    return local_weights


@njit(cache=True)
def _phase_space_overlap_parent_weights_numba(
    parent_weights,
    occupied_vertices,
    target_vertices,
    triangle_area,
    polygon_a,
    polygon_b,
) -> None:
    parent_weights[:] = 0.0
    difference_tolerance = _field_tolerance(occupied_vertices, target_vertices, 0.0)
    if _max_abs_difference(occupied_vertices, target_vertices) <= difference_tolerance:
        polygon_count = _build_occupied_polygon(
            occupied_vertices,
            target_vertices,
            polygon_a,
            polygon_b,
        )
        if polygon_count < 3:
            return
        _accumulate_polygon_overlap(parent_weights, polygon_b, polygon_count, triangle_area)
        parent_weights[0] *= 0.5
        parent_weights[1] *= 0.5
        parent_weights[2] *= 0.5
        return

    polygon_count = _build_double_step_polygon(
        occupied_vertices,
        target_vertices,
        polygon_a,
        polygon_b,
    )
    if polygon_count < 3:
        return

    _accumulate_polygon_overlap(parent_weights, polygon_a, polygon_count, triangle_area)


@njit(cache=True)
def _nesting_parent_weights_numba(
    parent_weights,
    occupied_vertices,
    target_vertices,
    triangle_area,
) -> None:
    parent_weights[:] = 0.0

    a11 = occupied_vertices[1] - occupied_vertices[0]
    a12 = occupied_vertices[2] - occupied_vertices[0]
    a21 = target_vertices[1] - target_vertices[0]
    a22 = target_vertices[2] - target_vertices[0]
    determinant = a11 * a22 - a12 * a21

    determinant_tolerance = _determinant_tolerance(occupied_vertices, target_vertices)
    if abs(determinant) <= determinant_tolerance:
        if _field_crosses_zero(occupied_vertices) and _field_crosses_zero(target_vertices):
            raise RuntimeError("encountered singular condition in nesting_function_weights")
        return

    rhs1 = -occupied_vertices[0]
    rhs2 = -target_vertices[0]
    xi = (rhs1 * a22 - rhs2 * a12) / determinant
    eta = (-rhs1 * a21 + rhs2 * a11) / determinant
    lambda0 = 1.0 - xi - eta
    lambda1 = xi
    lambda2 = eta
    barycentric_tolerance = _field_tolerance(occupied_vertices, target_vertices, 0.0)

    if (
        lambda0 < -barycentric_tolerance
        or lambda1 < -barycentric_tolerance
        or lambda2 < -barycentric_tolerance
    ):
        return

    if lambda0 < 0.0:
        lambda0 = 0.0
    if lambda1 < 0.0:
        lambda1 = 0.0
    if lambda2 < 0.0:
        lambda2 = 0.0

    normalization = lambda0 + lambda1 + lambda2
    if normalization == 0.0:
        return

    prefactor = 2.0 * triangle_area / abs(determinant)
    parent_weights[0] = prefactor * lambda0 / normalization
    parent_weights[1] = prefactor * lambda1 / normalization
    parent_weights[2] = prefactor * lambda2 / normalization


@njit(cache=True)
def _static_polarization_parent_weights_numba(
    parent_weights,
    occupied_vertices,
    target_vertices,
    triangle_area,
    polygon_a,
    polygon_b,
    sorted_order,
    sorted_delta,
    sorted_weights,
) -> None:
    parent_weights[:] = 0.0
    polygon_count = _build_occupied_empty_polygon(
        occupied_vertices,
        target_vertices,
        polygon_a,
        polygon_b,
    )
    if polygon_count < 3:
        return

    _accumulate_polygon_static_polarization(
        parent_weights,
        polygon_a,
        polygon_count,
        triangle_area,
        target_vertices,
        occupied_vertices,
        sorted_order,
        sorted_delta,
        sorted_weights,
    )


@njit(cache=True)
def _fermi_golden_rule_parent_weights_numba(
    parent_weights,
    occupied_vertices,
    target_vertices,
    sample_energies,
    triangle_area,
    polygon_a,
    polygon_b,
    sorted_order,
    sorted_delta,
    sorted_weights,
) -> None:
    parent_weights[:, :] = 0.0
    polygon_count = _build_occupied_empty_polygon(
        occupied_vertices,
        target_vertices,
        polygon_a,
        polygon_b,
    )
    if polygon_count < 3:
        return

    _accumulate_polygon_fermi_golden_rule(
        parent_weights,
        polygon_a,
        polygon_count,
        triangle_area,
        target_vertices,
        occupied_vertices,
        sample_energies,
        sorted_order,
        sorted_delta,
        sorted_weights,
    )


@njit(cache=True)
def _complex_polarization_parent_weights_numba(
    parent_weights,
    occupied_vertices,
    target_vertices,
    sample_energies,
    triangle_area,
    polygon_a,
    polygon_b,
    sorted_order,
    sorted_delta,
    sorted_weights,
) -> None:
    parent_weights[:, :] = 0.0
    polygon_count = _build_occupied_empty_polygon(
        occupied_vertices,
        target_vertices,
        polygon_a,
        polygon_b,
    )
    if polygon_count < 3:
        return

    _accumulate_polygon_complex_polarization(
        parent_weights,
        polygon_a,
        polygon_count,
        triangle_area,
        target_vertices,
        occupied_vertices,
        sample_energies,
        sorted_order,
        sorted_delta,
        sorted_weights,
    )


@njit(cache=True)
def _build_occupied_empty_polygon(occupied_vertices, target_vertices, polygon_a, polygon_b) -> int:
    _initialize_parent_triangle_polygon(polygon_a)
    polygon_count = 3
    occupied_tolerance = _field_tolerance(occupied_vertices, target_vertices, 0.0)
    target_tolerance = occupied_tolerance

    polygon_count = _clip_polygon_by_halfplane(
        polygon_a,
        polygon_count,
        occupied_vertices,
        -1.0,
        occupied_tolerance,
        polygon_b,
    )
    if polygon_count < 3:
        return 0

    polygon_count = _clip_polygon_by_halfplane(
        polygon_b,
        polygon_count,
        target_vertices,
        1.0,
        target_tolerance,
        polygon_a,
    )
    return polygon_count


@njit(cache=True)
def _build_double_step_polygon(occupied_vertices, target_vertices, polygon_a, polygon_b) -> int:
    _initialize_parent_triangle_polygon(polygon_a)
    polygon_count = 3
    tolerance = _field_tolerance(occupied_vertices, target_vertices, 0.0)

    polygon_count = _clip_polygon_by_halfplane(
        polygon_a,
        polygon_count,
        occupied_vertices,
        -1.0,
        tolerance,
        polygon_b,
    )
    if polygon_count < 3:
        return 0

    polygon_count = _clip_polygon_by_halfplane_difference(
        polygon_b,
        polygon_count,
        occupied_vertices,
        target_vertices,
        tolerance,
        polygon_a,
    )
    return polygon_count


@njit(cache=True)
def _build_occupied_polygon(occupied_vertices, target_vertices, polygon_a, polygon_b) -> int:
    _initialize_parent_triangle_polygon(polygon_a)
    polygon_count = 3
    tolerance = _field_tolerance(occupied_vertices, target_vertices, 0.0)
    polygon_count = _clip_polygon_by_halfplane(
        polygon_a,
        polygon_count,
        occupied_vertices,
        -1.0,
        tolerance,
        polygon_b,
    )
    return polygon_count


@njit(cache=True)
def _accumulate_polygon_overlap(parent_weights, polygon, polygon_count, triangle_area) -> None:
    area_tolerance = triangle_area * GEOMETRY_EPS
    one_third = 1.0 / 3.0

    for polygon_index in range(1, polygon_count - 1):
        subtriangle_area = _subtriangle_area(
            polygon[0],
            polygon[polygon_index],
            polygon[polygon_index + 1],
            triangle_area,
        )
        if subtriangle_area <= area_tolerance:
            continue

        local_weight = subtriangle_area * one_third
        _accumulate_local_triangle_weights(
            parent_weights,
            polygon[0],
            polygon[polygon_index],
            polygon[polygon_index + 1],
            local_weight,
            local_weight,
            local_weight,
        )


@njit(cache=True)
def _accumulate_polygon_static_polarization(
    parent_weights,
    polygon,
    polygon_count,
    triangle_area,
    target_vertices,
    occupied_vertices,
    sorted_order,
    sorted_delta,
    sorted_weights,
) -> None:
    delta_vertices = np.empty(3, dtype=np.float64)
    area_tolerance = triangle_area * GEOMETRY_EPS

    for polygon_index in range(1, polygon_count - 1):
        beta0 = polygon[0]
        beta1 = polygon[polygon_index]
        beta2 = polygon[polygon_index + 1]
        subtriangle_area = _subtriangle_area(beta0, beta1, beta2, triangle_area)
        if subtriangle_area <= area_tolerance:
            continue

        delta_vertices[0] = _delta_value(beta0, occupied_vertices, target_vertices)
        delta_vertices[1] = _delta_value(beta1, occupied_vertices, target_vertices)
        delta_vertices[2] = _delta_value(beta2, occupied_vertices, target_vertices)
        sort3(delta_vertices, sorted_order, sorted_delta)
        _snap_sorted_delta(sorted_delta)
        _static_basis_sorted_weights_numba(
            sorted_weights,
            sorted_delta[0],
            sorted_delta[1],
            sorted_delta[2],
            subtriangle_area,
        )
        _accumulate_sorted_local_triangle_weights(
            parent_weights,
            beta0,
            beta1,
            beta2,
            sorted_order,
            sorted_weights,
        )

    return


@njit(cache=True)
def _accumulate_polygon_fermi_golden_rule(
    parent_weights,
    polygon,
    polygon_count,
    triangle_area,
    target_vertices,
    occupied_vertices,
    sample_energies,
    sorted_order,
    sorted_delta,
    sorted_weights,
) -> None:
    delta_vertices = np.empty(3, dtype=np.float64)
    area_tolerance = triangle_area * GEOMETRY_EPS

    for polygon_index in range(1, polygon_count - 1):
        beta0 = polygon[0]
        beta1 = polygon[polygon_index]
        beta2 = polygon[polygon_index + 1]
        subtriangle_area = _subtriangle_area(beta0, beta1, beta2, triangle_area)
        if subtriangle_area <= area_tolerance:
            continue

        delta_vertices[0] = _delta_value(beta0, occupied_vertices, target_vertices)
        delta_vertices[1] = _delta_value(beta1, occupied_vertices, target_vertices)
        delta_vertices[2] = _delta_value(beta2, occupied_vertices, target_vertices)
        sort3(delta_vertices, sorted_order, sorted_delta)
        _snap_sorted_delta(sorted_delta)

        for energy_index in range(sample_energies.shape[0]):
            _dos_basis_sorted_weights_numba(
                sorted_weights,
                sorted_delta[0],
                sorted_delta[1],
                sorted_delta[2],
                sample_energies[energy_index],
                subtriangle_area,
            )
            _accumulate_sorted_local_triangle_weights(
                parent_weights[energy_index],
                beta0,
                beta1,
                beta2,
                sorted_order,
                sorted_weights,
            )


@njit(cache=True)
def _accumulate_polygon_complex_polarization(
    parent_weights,
    polygon,
    polygon_count,
    triangle_area,
    target_vertices,
    occupied_vertices,
    sample_energies,
    sorted_order,
    sorted_delta,
    sorted_weights,
) -> None:
    delta_vertices = np.empty(3, dtype=np.float64)
    area_tolerance = triangle_area * GEOMETRY_EPS

    for polygon_index in range(1, polygon_count - 1):
        beta0 = polygon[0]
        beta1 = polygon[polygon_index]
        beta2 = polygon[polygon_index + 1]
        subtriangle_area = _subtriangle_area(beta0, beta1, beta2, triangle_area)
        if subtriangle_area <= area_tolerance:
            continue

        delta_vertices[0] = _delta_value(beta0, occupied_vertices, target_vertices)
        delta_vertices[1] = _delta_value(beta1, occupied_vertices, target_vertices)
        delta_vertices[2] = _delta_value(beta2, occupied_vertices, target_vertices)
        sort3(delta_vertices, sorted_order, sorted_delta)
        _snap_sorted_delta(sorted_delta)

        for energy_index in range(sample_energies.shape[0]):
            _complex_basis_sorted_weights_numba(
                sorted_weights,
                sorted_delta[0],
                sorted_delta[1],
                sorted_delta[2],
                sample_energies[energy_index],
                subtriangle_area,
            )
            _accumulate_sorted_local_triangle_weights(
                parent_weights[energy_index],
                beta0,
                beta1,
                beta2,
                sorted_order,
                sorted_weights,
            )


@njit(cache=True)
def _initialize_parent_triangle_polygon(polygon) -> None:
    polygon[0, 0] = 1.0
    polygon[0, 1] = 0.0
    polygon[0, 2] = 0.0
    polygon[1, 0] = 0.0
    polygon[1, 1] = 1.0
    polygon[1, 2] = 0.0
    polygon[2, 0] = 0.0
    polygon[2, 1] = 0.0
    polygon[2, 2] = 1.0


@njit(cache=True)
def _clip_polygon_by_halfplane(
    polygon_in,
    polygon_count,
    field_values,
    sign,
    tolerance,
    polygon_out,
) -> int:
    output_count = 0
    if polygon_count == 0:
        return 0

    for polygon_index in range(polygon_count):
        next_index = polygon_index + 1
        if next_index == polygon_count:
            next_index = 0

        beta_a0 = polygon_in[polygon_index, 0]
        beta_a1 = polygon_in[polygon_index, 1]
        beta_a2 = polygon_in[polygon_index, 2]
        beta_b0 = polygon_in[next_index, 0]
        beta_b1 = polygon_in[next_index, 1]
        beta_b2 = polygon_in[next_index, 2]

        value_a = sign * (
            beta_a0 * field_values[0]
            + beta_a1 * field_values[1]
            + beta_a2 * field_values[2]
        )
        value_b = sign * (
            beta_b0 * field_values[0]
            + beta_b1 * field_values[1]
            + beta_b2 * field_values[2]
        )

        if abs(value_a) <= tolerance:
            value_a = 0.0
        if abs(value_b) <= tolerance:
            value_b = 0.0

        inside_a = value_a >= 0.0
        inside_b = value_b >= 0.0

        if inside_a and inside_b:
            output_count = _append_polygon_vertex(
                polygon_out,
                output_count,
                beta_b0,
                beta_b1,
                beta_b2,
            )
        elif inside_a and (not inside_b):
            fraction = value_a / (value_a - value_b)
            output_count = _append_polygon_vertex(
                polygon_out,
                output_count,
                (1.0 - fraction) * beta_a0 + fraction * beta_b0,
                (1.0 - fraction) * beta_a1 + fraction * beta_b1,
                (1.0 - fraction) * beta_a2 + fraction * beta_b2,
            )
        elif (not inside_a) and inside_b:
            fraction = value_a / (value_a - value_b)
            output_count = _append_polygon_vertex(
                polygon_out,
                output_count,
                (1.0 - fraction) * beta_a0 + fraction * beta_b0,
                (1.0 - fraction) * beta_a1 + fraction * beta_b1,
                (1.0 - fraction) * beta_a2 + fraction * beta_b2,
            )
            output_count = _append_polygon_vertex(
                polygon_out,
                output_count,
                beta_b0,
                beta_b1,
                beta_b2,
            )

    if output_count > 1 and _same_beta_row(polygon_out[0], polygon_out[output_count - 1]):
        output_count -= 1

    return output_count


@njit(cache=True)
def _append_polygon_vertex(polygon, count, beta0, beta1, beta2) -> int:
    if count > 0:
        if (
            abs(polygon[count - 1, 0] - beta0) <= GEOMETRY_EPS
            and abs(polygon[count - 1, 1] - beta1) <= GEOMETRY_EPS
            and abs(polygon[count - 1, 2] - beta2) <= GEOMETRY_EPS
        ):
            return count

    total = beta0 + beta1 + beta2
    if total != 0.0:
        beta0 = beta0 / total
        beta1 = beta1 / total
        beta2 = beta2 / total

    polygon[count, 0] = beta0
    polygon[count, 1] = beta1
    polygon[count, 2] = beta2
    return count + 1


@njit(cache=True)
def _clip_polygon_by_halfplane_difference(
    polygon_in,
    polygon_count,
    first_values,
    second_values,
    tolerance,
    polygon_out,
) -> int:
    output_count = 0
    if polygon_count == 0:
        return 0

    for polygon_index in range(polygon_count):
        next_index = polygon_index + 1
        if next_index == polygon_count:
            next_index = 0

        beta_a0 = polygon_in[polygon_index, 0]
        beta_a1 = polygon_in[polygon_index, 1]
        beta_a2 = polygon_in[polygon_index, 2]
        beta_b0 = polygon_in[next_index, 0]
        beta_b1 = polygon_in[next_index, 1]
        beta_b2 = polygon_in[next_index, 2]

        value_a = (
            beta_a0 * (first_values[0] - second_values[0])
            + beta_a1 * (first_values[1] - second_values[1])
            + beta_a2 * (first_values[2] - second_values[2])
        )
        value_b = (
            beta_b0 * (first_values[0] - second_values[0])
            + beta_b1 * (first_values[1] - second_values[1])
            + beta_b2 * (first_values[2] - second_values[2])
        )

        if abs(value_a) <= tolerance:
            value_a = 0.0
        if abs(value_b) <= tolerance:
            value_b = 0.0

        inside_a = value_a >= 0.0
        inside_b = value_b >= 0.0

        if inside_a and inside_b:
            output_count = _append_polygon_vertex(
                polygon_out,
                output_count,
                beta_b0,
                beta_b1,
                beta_b2,
            )
        elif inside_a and (not inside_b):
            fraction = value_a / (value_a - value_b)
            output_count = _append_polygon_vertex(
                polygon_out,
                output_count,
                (1.0 - fraction) * beta_a0 + fraction * beta_b0,
                (1.0 - fraction) * beta_a1 + fraction * beta_b1,
                (1.0 - fraction) * beta_a2 + fraction * beta_b2,
            )
        elif (not inside_a) and inside_b:
            fraction = value_a / (value_a - value_b)
            output_count = _append_polygon_vertex(
                polygon_out,
                output_count,
                (1.0 - fraction) * beta_a0 + fraction * beta_b0,
                (1.0 - fraction) * beta_a1 + fraction * beta_b1,
                (1.0 - fraction) * beta_a2 + fraction * beta_b2,
            )
            output_count = _append_polygon_vertex(
                polygon_out,
                output_count,
                beta_b0,
                beta_b1,
                beta_b2,
            )

    if output_count > 1 and _same_beta_row(polygon_out[0], polygon_out[output_count - 1]):
        output_count -= 1

    return output_count


@njit(cache=True)
def _same_beta_row(first, second) -> bool:
    return (
        abs(first[0] - second[0]) <= GEOMETRY_EPS
        and abs(first[1] - second[1]) <= GEOMETRY_EPS
        and abs(first[2] - second[2]) <= GEOMETRY_EPS
    )


@njit(cache=True)
def _subtriangle_area(beta0, beta1, beta2, triangle_area) -> float:
    determinant = (beta1[1] - beta0[1]) * (beta2[2] - beta0[2]) - (
        beta1[2] - beta0[2]
    ) * (beta2[1] - beta0[1])
    return abs(determinant) * triangle_area


@njit(cache=True)
def _delta_value(beta, occupied_vertices, target_vertices) -> float:
    return (
        beta[0] * (target_vertices[0] - occupied_vertices[0])
        + beta[1] * (target_vertices[1] - occupied_vertices[1])
        + beta[2] * (target_vertices[2] - occupied_vertices[2])
    )


@njit(cache=True)
def _accumulate_local_triangle_weights(
    parent_weights,
    beta0,
    beta1,
    beta2,
    weight0,
    weight1,
    weight2,
) -> None:
    parent_weights[0] += beta0[0] * weight0 + beta1[0] * weight1 + beta2[0] * weight2
    parent_weights[1] += beta0[1] * weight0 + beta1[1] * weight1 + beta2[1] * weight2
    parent_weights[2] += beta0[2] * weight0 + beta1[2] * weight1 + beta2[2] * weight2


@njit(cache=True)
def _accumulate_sorted_local_triangle_weights(
    parent_weights,
    beta0,
    beta1,
    beta2,
    sorted_order,
    sorted_weights,
) -> None:
    for local_index in range(3):
        weight = sorted_weights[local_index]
        vertex_index = sorted_order[local_index]
        if vertex_index == 0:
            parent_weights[0] += beta0[0] * weight
            parent_weights[1] += beta0[1] * weight
            parent_weights[2] += beta0[2] * weight
        elif vertex_index == 1:
            parent_weights[0] += beta1[0] * weight
            parent_weights[1] += beta1[1] * weight
            parent_weights[2] += beta1[2] * weight
        else:
            parent_weights[0] += beta2[0] * weight
            parent_weights[1] += beta2[1] * weight
            parent_weights[2] += beta2[2] * weight


@njit(cache=True)
def _dos_basis_sorted_weights_numba(sorted_weights, e1, e2, e3, energy, triangle_area) -> None:
    sorted_weights[:] = 0.0
    energy_tolerance = _field_tolerance3(e1, e2, e3, energy)

    if e3 - e1 <= energy_tolerance:
        return

    if abs(energy - e2) <= energy_tolerance:
        energy = e2

    if energy <= e1 + energy_tolerance or energy >= e3 - energy_tolerance:
        return

    if energy < e2:
        x = energy - e1
        d21 = e2 - e1
        d31 = e3 - e1
        sorted_weights[0] = triangle_area * x / (d21 * d31) * (2.0 - x / d21 - x / d31)
        sorted_weights[1] = triangle_area * x * x / (d21 * d21 * d31)
        sorted_weights[2] = triangle_area * x * x / (d21 * d31 * d31)
        return

    y = e3 - energy
    d31 = e3 - e1
    d32 = e3 - e2
    sorted_weights[0] = triangle_area * y * y / (d31 * d31 * d32)
    sorted_weights[1] = triangle_area * y * y / (d31 * d32 * d32)
    sorted_weights[2] = triangle_area * y / (d31 * d32) * (2.0 - y / d31 - y / d32)


@njit(cache=True)
def _static_basis_sorted_weights_numba(sorted_weights, delta1, delta2, delta3, triangle_area) -> None:
    sorted_weights[:] = 0.0
    delta_tolerance = _field_tolerance3(delta1, delta2, delta3, 0.0)
    span = delta3 - delta1
    mean_delta = (delta1 + delta2 + delta3) / 3.0

    if span <= delta_tolerance:
        if mean_delta > delta_tolerance:
            fallback = (triangle_area / 3.0) / mean_delta
            sorted_weights[0] = fallback
            sorted_weights[1] = fallback
            sorted_weights[2] = fallback
            return
        raise RuntimeError("encountered singular condition in static_polarization_weights")

    if abs(delta1) <= delta_tolerance:
        delta1 = 0.0
    if abs(delta2) <= delta_tolerance:
        delta2 = 0.0
    if abs(delta3) <= delta_tolerance:
        delta3 = 0.0

    low_width = delta2 - delta1
    high_width = delta3 - delta2
    full_width = delta3 - delta1

    if low_width > delta_tolerance:
        r1 = _lower_interval_r1(low_width, delta1)
        r2 = _lower_interval_r2(low_width, delta1)
        sorted_weights[0] += triangle_area * (
            2.0 * r1 / (low_width * full_width)
            - (1.0 / (low_width * low_width * full_width) + 1.0 / (low_width * full_width * full_width)) * r2
        )
        sorted_weights[1] += triangle_area * r2 / (low_width * low_width * full_width)
        sorted_weights[2] += triangle_area * r2 / (low_width * full_width * full_width)

    if high_width > delta_tolerance:
        s1 = _upper_interval_s1(high_width, delta3)
        s2 = _upper_interval_s2(high_width, delta3)
        sorted_weights[0] += triangle_area * s2 / (full_width * full_width * high_width)
        sorted_weights[1] += triangle_area * s2 / (full_width * high_width * high_width)
        sorted_weights[2] += triangle_area * (
            2.0 * s1 / (full_width * high_width)
            - (1.0 / (full_width * full_width * high_width) + 1.0 / (full_width * high_width * high_width)) * s2
        )


@njit(cache=True)
def _complex_basis_sorted_weights_numba(
    sorted_weights,
    delta1,
    delta2,
    delta3,
    energy,
    triangle_area,
) -> None:
    sorted_weights[:] = 0.0j
    delta_tolerance = _field_tolerance3(delta1, delta2, delta3, 0.0)
    span = delta3 - delta1
    mean_delta = (delta1 + delta2 + delta3) / 3.0

    if span <= delta_tolerance:
        denominator = mean_delta + energy
        if abs(denominator) > delta_tolerance:
            fallback = (triangle_area / 3.0) / denominator
            sorted_weights[0] = fallback
            sorted_weights[1] = fallback
            sorted_weights[2] = fallback
            return
        raise RuntimeError("encountered singular condition in complex_frequency_polarization_weights")

    if abs(energy.imag) <= delta_tolerance and abs(energy.real) <= delta_tolerance:
        real_sorted = np.empty(3, dtype=np.float64)
        _static_basis_sorted_weights_numba(real_sorted, delta1, delta2, delta3, triangle_area)
        sorted_weights[0] = real_sorted[0]
        sorted_weights[1] = real_sorted[1]
        sorted_weights[2] = real_sorted[2]
        return

    low_width = delta2 - delta1
    high_width = delta3 - delta2
    full_width = delta3 - delta1
    z1 = delta1 + energy
    z3 = delta3 + energy

    if low_width > delta_tolerance:
        r1 = _complex_lower_interval_r1(low_width, z1)
        r2 = _complex_lower_interval_r2(low_width, z1)
        sorted_weights[0] += triangle_area * (
            2.0 * r1 / (low_width * full_width)
            - (1.0 / (low_width * low_width * full_width) + 1.0 / (low_width * full_width * full_width)) * r2
        )
        sorted_weights[1] += triangle_area * r2 / (low_width * low_width * full_width)
        sorted_weights[2] += triangle_area * r2 / (low_width * full_width * full_width)

    if high_width > delta_tolerance:
        s1 = _complex_upper_interval_s1(high_width, z3)
        s2 = _complex_upper_interval_s2(high_width, z3)
        sorted_weights[0] += triangle_area * s2 / (full_width * full_width * high_width)
        sorted_weights[1] += triangle_area * s2 / (full_width * high_width * high_width)
        sorted_weights[2] += triangle_area * (
            2.0 * s1 / (full_width * high_width)
            - (1.0 / (full_width * full_width * high_width) + 1.0 / (full_width * high_width * high_width)) * s2
        )


@njit(cache=True)
def _lower_interval_r1(width, offset) -> float:
    if offset == 0.0:
        return width
    return width - offset * np.log((offset + width) / offset)


@njit(cache=True)
def _lower_interval_r2(width, offset) -> float:
    if offset == 0.0:
        return 0.5 * width * width
    return 0.5 * width * width - offset * width + offset * offset * np.log((offset + width) / offset)


@njit(cache=True)
def _upper_interval_s1(width, offset) -> float:
    return -width + offset * np.log(offset / (offset - width))


@njit(cache=True)
def _upper_interval_s2(width, offset) -> float:
    return -0.5 * width * width - offset * width + offset * offset * np.log(offset / (offset - width))


@njit(cache=True)
def _complex_lower_interval_r1(width, offset):
    return width - offset * np.log((offset + width) / offset)


@njit(cache=True)
def _complex_lower_interval_r2(width, offset):
    return 0.5 * width * width - offset * width + offset * offset * np.log((offset + width) / offset)


@njit(cache=True)
def _complex_upper_interval_s1(width, offset):
    return -width + offset * np.log(offset / (offset - width))


@njit(cache=True)
def _complex_upper_interval_s2(width, offset):
    return -0.5 * width * width - offset * width + offset * offset * np.log(offset / (offset - width))


@njit(cache=True)
def _field_crosses_zero(values) -> bool:
    minimum = values[0]
    maximum = values[0]
    for index in range(1, values.shape[0]):
        value = values[index]
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value
    return minimum <= 0.0 <= maximum


@njit(cache=True)
def _snap_sorted_delta(sorted_delta) -> None:
    tolerance = _field_tolerance3(sorted_delta[0], sorted_delta[1], sorted_delta[2], 0.0)
    for index in range(3):
        if abs(sorted_delta[index]) <= tolerance:
            sorted_delta[index] = 0.0


@njit(cache=True)
def _field_tolerance(values_a, values_b, extra) -> float:
    scale = 1.0
    for index in range(values_a.shape[0]):
        value_a = abs(values_a[index])
        value_b = abs(values_b[index])
        if value_a > scale:
            scale = value_a
        if value_b > scale:
            scale = value_b
    extra_abs = abs(extra)
    if extra_abs > scale:
        scale = extra_abs
    return ENERGY_ATOL + 64.0 * MACHINE_EPS * scale


@njit(cache=True)
def _field_tolerance3(value0, value1, value2, extra) -> float:
    scale = 1.0
    absolute0 = abs(value0)
    absolute1 = abs(value1)
    absolute2 = abs(value2)
    absolute_extra = abs(extra)
    if absolute0 > scale:
        scale = absolute0
    if absolute1 > scale:
        scale = absolute1
    if absolute2 > scale:
        scale = absolute2
    if absolute_extra > scale:
        scale = absolute_extra
    return ENERGY_ATOL + 64.0 * MACHINE_EPS * scale


@njit(cache=True)
def _determinant_tolerance(values_a, values_b) -> float:
    scale = 1.0
    for index in range(values_a.shape[0]):
        value_a = abs(values_a[index])
        value_b = abs(values_b[index])
        if value_a > scale:
            scale = value_a
        if value_b > scale:
            scale = value_b
    return ENERGY_ATOL + 64.0 * MACHINE_EPS * scale * scale


@njit(cache=True)
def _max_abs_difference(values_a, values_b) -> float:
    maximum = 0.0
    for index in range(values_a.shape[0]):
        difference = abs(values_a[index] - values_b[index])
        if difference > maximum:
            maximum = difference
    return maximum
