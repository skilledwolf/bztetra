from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._grids import FloatArray
from ._grids import interpolate_local_values
from ._grids import interpolated_tetrahedron_energies
from ._grids import normalize_eigenvalues
from ._grids import normalize_energy_samples
from .formulas import small_tetrahedron_cut
from .formulas import triangle_cut
from .geometry import IntegrationMesh
from .geometry import TetraMethod
from .geometry import build_integration_mesh


def density_of_states_weights(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    eig_flat, energy_grid_shape = normalize_eigenvalues(eigenvalues)
    sample_energies = normalize_energy_samples(energies)
    mesh = build_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    tetra_band_energies = interpolated_tetrahedron_energies(mesh, eig_flat)
    local_weights = _dos_weights_on_local_mesh(mesh, tetra_band_energies, sample_energies)
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_energy_band_last(output_flat, mesh.weight_grid_shape)


def dos(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    return density_of_states_weights(
        reciprocal_vectors,
        eigenvalues,
        energies,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )


def integrated_density_of_states_weights(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    eig_flat, energy_grid_shape = normalize_eigenvalues(eigenvalues)
    sample_energies = normalize_energy_samples(energies)
    mesh = build_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    tetra_band_energies = interpolated_tetrahedron_energies(mesh, eig_flat)
    local_weights = _intdos_weights_on_local_mesh(mesh, tetra_band_energies, sample_energies)
    output_flat = interpolate_local_values(mesh, local_weights)
    return _unflatten_energy_band_last(output_flat, mesh.weight_grid_shape)


def intdos(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    return integrated_density_of_states_weights(
        reciprocal_vectors,
        eigenvalues,
        energies,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )


def _dos_weights_on_local_mesh(
    mesh: IntegrationMesh,
    tetra_band_energies: FloatArray,
    sample_energies: FloatArray,
) -> FloatArray:
    band_count = tetra_band_energies.shape[2]
    local_weights = np.zeros((mesh.local_point_count, sample_energies.shape[0], band_count), dtype=np.float64)

    for tetrahedron_index in range(mesh.tetrahedron_count):
        local_points = mesh.local_point_indices[tetrahedron_index]
        for band_index in range(band_count):
            sorted_order = np.argsort(tetra_band_energies[tetrahedron_index, :, band_index])
            sorted_energies = tetra_band_energies[tetrahedron_index, sorted_order, band_index]
            for energy_index, energy in enumerate(sample_energies):
                vertex_weights = _dos_vertex_weights(sorted_energies, sorted_order, energy)
                point_weights = vertex_weights @ mesh.tetrahedron_weight_matrix
                np.add.at(local_weights[:, energy_index, band_index], local_points, point_weights)

    local_weights /= float(6 * np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return local_weights


def _intdos_weights_on_local_mesh(
    mesh: IntegrationMesh,
    tetra_band_energies: FloatArray,
    sample_energies: FloatArray,
) -> FloatArray:
    band_count = tetra_band_energies.shape[2]
    local_weights = np.zeros((mesh.local_point_count, sample_energies.shape[0], band_count), dtype=np.float64)

    for tetrahedron_index in range(mesh.tetrahedron_count):
        local_points = mesh.local_point_indices[tetrahedron_index]
        for band_index in range(band_count):
            sorted_order = np.argsort(tetra_band_energies[tetrahedron_index, :, band_index])
            sorted_energies = tetra_band_energies[tetrahedron_index, sorted_order, band_index]
            for energy_index, energy in enumerate(sample_energies):
                vertex_weights = _intdos_vertex_weights(sorted_energies, sorted_order, energy)
                point_weights = vertex_weights @ mesh.tetrahedron_weight_matrix
                np.add.at(local_weights[:, energy_index, band_index], local_points, point_weights)

    local_weights /= float(6 * np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return local_weights


def _dos_vertex_weights(
    sorted_energies: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sample_energy: float,
) -> FloatArray:
    weights = np.zeros(4, dtype=np.float64)
    shifted = sorted_energies - sample_energy

    if sorted_energies[0] <= sample_energy <= sorted_energies[1]:
        _accumulate_triangle_cut(weights, sorted_order, "a1", shifted)
        return weights

    if sorted_energies[1] <= sample_energy <= sorted_energies[2]:
        for kind in ("b1", "b2"):
            _accumulate_triangle_cut(weights, sorted_order, kind, shifted)
        return weights

    if sorted_energies[2] <= sample_energy <= sorted_energies[3]:
        _accumulate_triangle_cut(weights, sorted_order, "c1", shifted)

    return weights


def _intdos_vertex_weights(
    sorted_energies: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sample_energy: float,
) -> FloatArray:
    weights = np.zeros(4, dtype=np.float64)
    shifted = sorted_energies - sample_energy

    if sorted_energies[0] <= sample_energy <= sorted_energies[1]:
        _accumulate_small_tetra_cut(weights, sorted_order, "a1", shifted)
        return weights

    if sorted_energies[1] <= sample_energy <= sorted_energies[2]:
        for kind in ("b1", "b2", "b3"):
            _accumulate_small_tetra_cut(weights, sorted_order, kind, shifted)
        return weights

    if sorted_energies[2] <= sample_energy <= sorted_energies[3]:
        for kind in ("c1", "c2", "c3"):
            _accumulate_small_tetra_cut(weights, sorted_order, kind, shifted)
        return weights

    if sorted_energies[3] <= sample_energy:
        weights[:] = 0.25

    return weights


def _accumulate_triangle_cut(
    weights: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    kind: str,
    shifted_energies: FloatArray,
) -> None:
    cut = triangle_cut(kind, shifted_energies)
    weights[sorted_order] += cut.volume_factor * cut.coefficients.sum(axis=0) / 3.0


def _accumulate_small_tetra_cut(
    weights: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    kind: str,
    shifted_energies: FloatArray,
) -> None:
    cut = small_tetrahedron_cut(kind, shifted_energies)
    weights[sorted_order] += cut.volume_factor * cut.coefficients.sum(axis=0) * 0.25


def _unflatten_energy_band_last(values: FloatArray, grid_shape: tuple[int, int, int]) -> FloatArray:
    energy_count = values.shape[1]
    band_count = values.shape[2]
    reshaped = values.reshape((grid_shape[2], grid_shape[1], grid_shape[0], energy_count, band_count))
    return np.transpose(reshaped, (3, 2, 1, 0, 4))
