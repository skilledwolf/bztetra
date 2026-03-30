from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._grids import FloatArray
from ._grids import interpolate_local_values
from ._grids import interpolated_tetrahedron_energies
from ._grids import normalize_eigenvalues
from .formulas import small_tetrahedron_cut
from .formulas import triangle_cut
from .geometry import IntegrationMesh
from .geometry import TetraMethod
from .geometry import build_integration_mesh


def double_step_weights(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    occupied_flat, target_flat, energy_grid_shape = _normalize_eigenvalue_pair(occupied_eigenvalues, target_eigenvalues)
    mesh = build_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    occupied_tetra = interpolated_tetrahedron_energies(mesh, occupied_flat)
    target_tetra = interpolated_tetrahedron_energies(mesh, target_flat)
    local_weights = _double_step_weights_on_local_mesh(mesh, occupied_tetra, target_tetra)
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_pair_band_last(output_flat, mesh.weight_grid_shape)


def dblstep(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    return double_step_weights(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )


def double_delta_weights(
    reciprocal_vectors: npt.ArrayLike,
    source_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    source_flat, target_flat, energy_grid_shape = _normalize_eigenvalue_pair(source_eigenvalues, target_eigenvalues)
    mesh = build_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    source_tetra = interpolated_tetrahedron_energies(mesh, source_flat)
    target_tetra = interpolated_tetrahedron_energies(mesh, target_flat)
    local_weights = _double_delta_weights_on_local_mesh(mesh, source_tetra, target_tetra)
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_pair_band_last(output_flat, mesh.weight_grid_shape)


def dbldelta(
    reciprocal_vectors: npt.ArrayLike,
    source_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    return double_delta_weights(
        reciprocal_vectors,
        source_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )


def _double_step_weights_on_local_mesh(
    mesh: IntegrationMesh,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
) -> FloatArray:
    occupied_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    local_weights = np.zeros((mesh.local_point_count, target_band_count, occupied_band_count), dtype=np.float64)

    for tetrahedron_index in range(mesh.tetrahedron_count):
        local_points = mesh.local_point_indices[tetrahedron_index]
        for occupied_band_index in range(occupied_band_count):
            outer_weights = np.zeros((target_band_count, 4), dtype=np.float64)
            sorted_order = np.argsort(occupied_tetra[tetrahedron_index, :, occupied_band_index])
            sorted_occupied = occupied_tetra[tetrahedron_index, sorted_order, occupied_band_index]
            sorted_target = target_tetra[tetrahedron_index, sorted_order, :]

            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_response(
                    outer_weights,
                    "a1",
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    _double_step_secondary_weights,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for kind in ("b1", "b2", "b3"):
                    _accumulate_small_tetra_response(
                        outer_weights,
                        kind,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        _double_step_secondary_weights,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for kind in ("c1", "c2", "c3"):
                    _accumulate_small_tetra_response(
                        outer_weights,
                        kind,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        _double_step_secondary_weights,
                    )
            elif sorted_occupied[3] <= 0.0:
                outer_weights += _double_step_secondary_weights(
                    occupied_tetra[tetrahedron_index, :, occupied_band_index],
                    target_tetra[tetrahedron_index],
                )

            point_weights = outer_weights @ mesh.tetrahedron_weight_matrix
            np.add.at(local_weights[:, :, occupied_band_index], local_points, point_weights.T)

    local_weights /= float(6 * np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return local_weights


def _double_delta_weights_on_local_mesh(
    mesh: IntegrationMesh,
    source_tetra: FloatArray,
    target_tetra: FloatArray,
) -> FloatArray:
    source_band_count = source_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    local_weights = np.zeros((mesh.local_point_count, target_band_count, source_band_count), dtype=np.float64)

    for tetrahedron_index in range(mesh.tetrahedron_count):
        local_points = mesh.local_point_indices[tetrahedron_index]
        for source_band_index in range(source_band_count):
            outer_weights = np.zeros((target_band_count, 4), dtype=np.float64)
            sorted_order = np.argsort(source_tetra[tetrahedron_index, :, source_band_index])
            sorted_source = source_tetra[tetrahedron_index, sorted_order, source_band_index]
            sorted_target = target_tetra[tetrahedron_index, sorted_order, :]

            if sorted_source[0] < 0.0 <= sorted_source[1]:
                _accumulate_triangle_response(
                    outer_weights,
                    "a1",
                    sorted_order,
                    sorted_source,
                    sorted_target,
                )
            elif sorted_source[1] < 0.0 <= sorted_source[2]:
                for kind in ("b1", "b2"):
                    _accumulate_triangle_response(
                        outer_weights,
                        kind,
                        sorted_order,
                        sorted_source,
                        sorted_target,
                    )
            elif sorted_source[2] < 0.0 < sorted_source[3]:
                _accumulate_triangle_response(
                    outer_weights,
                    "c1",
                    sorted_order,
                    sorted_source,
                    sorted_target,
                )

            point_weights = outer_weights @ mesh.tetrahedron_weight_matrix
            np.add.at(local_weights[:, :, source_band_index], local_points, point_weights.T)

    local_weights /= float(6 * np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return local_weights


def _accumulate_small_tetra_response(
    weights: FloatArray,
    kind: str,
    sorted_order: npt.NDArray[np.int64],
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    inner_kernel,
) -> None:
    cut = small_tetrahedron_cut(kind, sorted_occupied)
    transformed_occupied = cut.coefficients @ sorted_occupied
    transformed_target = cut.coefficients @ sorted_target
    weights[:, sorted_order] += cut.volume_factor * (inner_kernel(transformed_occupied, transformed_target) @ cut.coefficients)


def _accumulate_triangle_response(
    weights: FloatArray,
    kind: str,
    sorted_order: npt.NDArray[np.int64],
    sorted_source: FloatArray,
    sorted_target: FloatArray,
) -> None:
    cut = triangle_cut(kind, sorted_source)
    transformed_target = cut.coefficients @ sorted_target
    weights[:, sorted_order] += cut.volume_factor * (_double_delta_secondary_weights(transformed_target) @ cut.coefficients)


def _double_step_secondary_weights(occupied_vertices: FloatArray, target_vertices: FloatArray) -> FloatArray:
    target_band_count = target_vertices.shape[1]
    weights = np.zeros((target_band_count, 4), dtype=np.float64)

    for target_band_index in range(target_band_count):
        energy_difference = -occupied_vertices + target_vertices[:, target_band_index]
        sorted_order = np.argsort(energy_difference)
        sorted_difference = energy_difference[sorted_order]

        if abs(sorted_difference[0]) < 1.0e-8 and abs(sorted_difference[3]) < 1.0e-8:
            weights[target_band_index] = 0.125
            continue

        sorted_weights = np.zeros(4, dtype=np.float64)
        if (sorted_difference[0] <= 0.0 < sorted_difference[1]) or (sorted_difference[0] < 0.0 <= sorted_difference[1]):
            _accumulate_small_tetra_step(sorted_weights, "a1", sorted_difference)
        elif (sorted_difference[1] <= 0.0 < sorted_difference[2]) or (sorted_difference[1] < 0.0 <= sorted_difference[2]):
            for kind in ("b1", "b2", "b3"):
                _accumulate_small_tetra_step(sorted_weights, kind, sorted_difference)
        elif (sorted_difference[2] <= 0.0 < sorted_difference[3]) or (sorted_difference[2] < 0.0 <= sorted_difference[3]):
            for kind in ("c1", "c2", "c3"):
                _accumulate_small_tetra_step(sorted_weights, kind, sorted_difference)
        elif sorted_difference[3] <= 0.0:
            sorted_weights[:] = 0.25

        weights[target_band_index, sorted_order] = sorted_weights

    return weights


def _double_delta_secondary_weights(triangle_vertices: FloatArray) -> FloatArray:
    target_band_count = triangle_vertices.shape[1]
    weights = np.zeros((target_band_count, 3), dtype=np.float64)

    for target_band_index in range(target_band_count):
        energies = triangle_vertices[:, target_band_index]
        if float(np.max(np.abs(energies))) < 1.0e-10:
            raise RuntimeError("encountered nesting condition in dbldelta")

        sorted_order = np.argsort(energies)
        sorted_energies = _normalize_sorted_triangle_energies(energies[sorted_order])
        affine = _triangle_affine_coefficients(sorted_energies)
        sorted_weights = np.zeros(3, dtype=np.float64)

        if (sorted_energies[0] < 0.0 <= sorted_energies[1]) or (sorted_energies[0] <= 0.0 < sorted_energies[1]):
            volume_factor = affine[1, 0] / (sorted_energies[2] - sorted_energies[0])
            sorted_weights[0] = volume_factor * (affine[0, 1] + affine[0, 2])
            sorted_weights[1] = volume_factor * affine[1, 0]
            sorted_weights[2] = volume_factor * affine[2, 0]
        elif (sorted_energies[1] <= 0.0 < sorted_energies[2]) or (sorted_energies[1] < 0.0 <= sorted_energies[2]):
            volume_factor = affine[1, 2] / (sorted_energies[2] - sorted_energies[0])
            sorted_weights[0] = volume_factor * affine[0, 2]
            sorted_weights[1] = volume_factor * affine[1, 2]
            sorted_weights[2] = volume_factor * (affine[2, 0] + affine[2, 1])

        weights[target_band_index, sorted_order] = sorted_weights

    return weights


def _accumulate_small_tetra_step(weights: FloatArray, kind: str, sorted_energies: FloatArray) -> None:
    cut = small_tetrahedron_cut(kind, sorted_energies)
    weights += cut.volume_factor * cut.coefficients.sum(axis=0) * 0.25


def _normalize_eigenvalue_pair(
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, tuple[int, int, int]]:
    occupied_flat, occupied_grid_shape = normalize_eigenvalues(occupied_eigenvalues)
    target_flat, target_grid_shape = normalize_eigenvalues(target_eigenvalues)
    if occupied_grid_shape != target_grid_shape:
        raise ValueError("occupied and target eigenvalue grids must share the same shape")
    return occupied_flat, target_flat, occupied_grid_shape


def _normalize_sorted_triangle_energies(energies: npt.ArrayLike) -> FloatArray:
    values = np.asarray(energies, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"expected three sorted energies, got shape {values.shape!r}")
    if not np.all(np.isfinite(values)):
        raise ValueError("triangle energies must be finite")

    adjusted = values.copy()
    if np.any(np.diff(adjusted) < 0.0):
        raise ValueError("triangle energies must be sorted in nondecreasing order")
    for index in range(1, adjusted.size):
        if adjusted[index] <= adjusted[index - 1]:
            adjusted[index] = np.nextafter(adjusted[index - 1], np.inf)
    return adjusted


def _triangle_affine_coefficients(energies: FloatArray) -> FloatArray:
    coefficients = np.full((3, 3), np.nan, dtype=np.float64)
    for column, energy in enumerate(energies):
        mask = np.arange(3) != column
        coefficients[mask, column] = -energy / (energies[mask] - energy)
    return coefficients


def _unflatten_pair_band_last(values: FloatArray, grid_shape: tuple[int, int, int]) -> FloatArray:
    target_band_count = values.shape[1]
    source_band_count = values.shape[2]
    reshaped = values.reshape((grid_shape[2], grid_shape[1], grid_shape[0], target_band_count, source_band_count))
    return np.transpose(reshaped, (2, 1, 0, 3, 4))
