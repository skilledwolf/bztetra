from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._grids import FloatArray
from ._grids import interpolate_local_values
from ._grids import interpolated_tetrahedron_energies
from ._grids import normalize_eigenvalues
from .formulas import small_tetrahedron_cut
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
    band_count = tetra_band_energies.shape[2]
    local_weights = np.zeros((mesh.local_point_count, band_count), dtype=np.float64)

    for tetrahedron_index in range(mesh.tetrahedron_count):
        local_points = mesh.local_point_indices[tetrahedron_index]
        for band_index in range(band_count):
            sorted_order = np.argsort(tetra_band_energies[tetrahedron_index, :, band_index])
            sorted_energies = tetra_band_energies[tetrahedron_index, sorted_order, band_index] - fermi_energy
            vertex_weights = _occupation_vertex_weights(sorted_energies, sorted_order)
            point_weights = vertex_weights @ mesh.tetrahedron_weight_matrix
            np.add.at(local_weights[:, band_index], local_points, point_weights)

    local_weights /= float(6 * np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return local_weights


def _occupation_vertex_weights(sorted_energies: FloatArray, sorted_order: npt.NDArray[np.int64]) -> FloatArray:
    weights = np.zeros(4, dtype=np.float64)

    if sorted_energies[0] <= 0.0 < sorted_energies[1]:
        _accumulate_cut(weights, sorted_order, "a1", sorted_energies)
        return weights

    if sorted_energies[1] <= 0.0 < sorted_energies[2]:
        for kind in ("b1", "b2", "b3"):
            _accumulate_cut(weights, sorted_order, kind, sorted_energies)
        return weights

    if sorted_energies[2] <= 0.0 < sorted_energies[3]:
        for kind in ("c1", "c2", "c3"):
            _accumulate_cut(weights, sorted_order, kind, sorted_energies)
        return weights

    if sorted_energies[3] <= 0.0:
        weights[:] = 0.25

    return weights


def _accumulate_cut(
    weights: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    kind: str,
    sorted_energies: FloatArray,
) -> None:
    cut = small_tetrahedron_cut(kind, sorted_energies)
    weights[sorted_order] += cut.volume_factor * cut.coefficients.sum(axis=0) * 0.25


def _unflatten_band_last(values: FloatArray, grid_shape: tuple[int, int, int]) -> FloatArray:
    band_count = values.shape[1]
    reshaped = values.reshape((grid_shape[2], grid_shape[1], grid_shape[0], band_count))
    return np.transpose(reshaped, (2, 1, 0, 3))
