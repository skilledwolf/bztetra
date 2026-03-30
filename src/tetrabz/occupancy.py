from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .formulas import small_tetrahedron_cut
from .geometry import build_integration_mesh
from .geometry import trilinear_interpolation_indices
from .geometry import TetraMethod


FloatArray = npt.NDArray[np.float64]


def occupation_weights(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    weight_grid_shape: tuple[int, int, int] | None = None,
    *,
    method: int | TetraMethod = "optimized",
    fermi_energy: float = 0.0,
) -> FloatArray:
    """Compute occupation weights for `theta(fermi_energy - eigenvalues)`."""

    eig = _normalize_eigenvalues(eigenvalues)
    mesh = build_integration_mesh(
        reciprocal_vectors,
        eig.shape[:3],
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    eig_flat = _flatten_eigenvalues(eig)
    local_weights = _occupation_weights_flat(mesh, eig_flat, fermi_energy=float(fermi_energy))
    weight_grid = mesh.weight_grid_shape
    weights_flat = (
        _interpolate_local_weights(local_weights, weight_grid, mesh.fractional_kpoints)
        if mesh.interpolation_required
        else local_weights
    )
    return _reshape_weights(weights_flat, weight_grid)


def solve_fermi_energy(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    nelec: float,
    weight_grid_shape: tuple[int, int, int] | None = None,
    *,
    method: int | TetraMethod = "optimized",
    tolerance: float = 1.0e-10,
    max_iterations: int = 300,
) -> tuple[float, FloatArray]:
    """Solve the Fermi energy with a bisection search and return the weights."""

    eig = _normalize_eigenvalues(eigenvalues)
    mesh = build_integration_mesh(
        reciprocal_vectors,
        eig.shape[:3],
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    eig_flat = _flatten_eigenvalues(eig)

    lower = float(np.min(eig_flat))
    upper = float(np.max(eig_flat))
    local_weights: FloatArray | None = None
    fermi_energy = 0.5 * (lower + upper)

    for _ in range(max_iterations):
        fermi_energy = 0.5 * (lower + upper)
        local_weights = _occupation_weights_flat(mesh, eig_flat, fermi_energy=fermi_energy)
        electron_count = float(np.sum(local_weights))
        if abs(electron_count - nelec) < tolerance:
            break
        if electron_count < nelec:
            lower = fermi_energy
        else:
            upper = fermi_energy
    else:
        raise RuntimeError("solve_fermi_energy did not converge")

    assert local_weights is not None
    weight_grid = mesh.weight_grid_shape
    weights_flat = (
        _interpolate_local_weights(local_weights, weight_grid, mesh.fractional_kpoints)
        if mesh.interpolation_required
        else local_weights
    )
    return fermi_energy, _reshape_weights(weights_flat, weight_grid)


def occ(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    weight_grid_shape: tuple[int, int, int] | None = None,
    *,
    method: int | TetraMethod = "optimized",
    fermi_energy: float = 0.0,
) -> FloatArray:
    """Compatibility alias for the occupation weights."""

    return occupation_weights(
        reciprocal_vectors,
        eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
        fermi_energy=fermi_energy,
    )


def fermieng(
    reciprocal_vectors: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    nelec: float,
    weight_grid_shape: tuple[int, int, int] | None = None,
    *,
    method: int | TetraMethod = "optimized",
    tolerance: float = 1.0e-10,
    max_iterations: int = 300,
) -> tuple[float, FloatArray]:
    """Compatibility alias for the Fermi-energy solve."""

    return solve_fermi_energy(
        reciprocal_vectors,
        eigenvalues,
        nelec,
        weight_grid_shape=weight_grid_shape,
        method=method,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _occupation_weights_flat(
    mesh,
    eig_flat: FloatArray,
    *,
    fermi_energy: float,
) -> FloatArray:
    band_count = eig_flat.shape[0]
    local_weights = np.zeros((band_count, mesh.local_point_count), dtype=np.float64)
    smoothing = mesh.tetrahedron_weight_matrix
    normalization = float(mesh.tetrahedron_count)

    for global_points, local_points in zip(mesh.global_point_indices, mesh.local_point_indices, strict=True):
        smoothed = eig_flat[:, global_points] @ smoothing.T
        for band_index in range(band_count):
            shifted = smoothed[band_index] - fermi_energy
            order = np.argsort(shifted, kind="mergesort")
            sorted_energies = shifted[order]
            vertex_weights = _occupation_vertex_weights(sorted_energies)
            if not np.any(vertex_weights):
                continue

            contribution = vertex_weights @ smoothing
            reordered = np.empty_like(vertex_weights)
            reordered[order] = vertex_weights
            np.add.at(local_weights[band_index], local_points, reordered @ smoothing)

    return local_weights / normalization


def _occupation_vertex_weights(sorted_energies: FloatArray) -> FloatArray:
    weights = np.zeros(4, dtype=np.float64)
    regularized = _regularize_sorted_energies(sorted_energies)

    if sorted_energies[0] <= 0.0 < sorted_energies[1]:
        cut = small_tetrahedron_cut("a1", regularized)
        weights += 0.25 * cut.volume_factor * cut.coefficients.sum(axis=0)
    elif sorted_energies[1] <= 0.0 < sorted_energies[2]:
        for kind in ("b1", "b2", "b3"):
            cut = small_tetrahedron_cut(kind, regularized)
            weights += 0.25 * cut.volume_factor * cut.coefficients.sum(axis=0)
    elif sorted_energies[2] <= 0.0 < sorted_energies[3]:
        for kind in ("c1", "c2", "c3"):
            cut = small_tetrahedron_cut(kind, regularized)
            weights += 0.25 * cut.volume_factor * cut.coefficients.sum(axis=0)
    elif sorted_energies[3] <= 0.0:
        weights.fill(0.25)

    return weights


def _interpolate_local_weights(
    local_weights: FloatArray,
    weight_grid_shape: tuple[int, int, int],
    fractional_kpoints: FloatArray | None,
) -> FloatArray:
    if fractional_kpoints is None:
        raise ValueError("fractional_kpoints are required for interpolation")

    interpolated = np.zeros((local_weights.shape[0], int(np.prod(weight_grid_shape))), dtype=np.float64)
    for local_index, kpoint in enumerate(fractional_kpoints):
        indices, factors = trilinear_interpolation_indices(weight_grid_shape, kpoint)
        values = local_weights[:, local_index]
        for index, factor in zip(indices, factors, strict=True):
            interpolated[:, index] += values * factor
    return interpolated


def _reshape_weights(weights_flat: FloatArray, grid_shape: tuple[int, int, int]) -> FloatArray:
    band_count = weights_flat.shape[0]
    return (
        weights_flat.reshape((band_count, *grid_shape), order="F")
        .transpose(1, 2, 3, 0)
        .copy()
    )


def _flatten_eigenvalues(eigenvalues: FloatArray) -> FloatArray:
    band_count = eigenvalues.shape[3]
    return np.ascontiguousarray(
        eigenvalues.transpose(3, 0, 1, 2).reshape((band_count, -1), order="F"),
        dtype=np.float64,
    )


def _normalize_eigenvalues(eigenvalues: npt.ArrayLike) -> FloatArray:
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(f"expected eigenvalues with shape (nx, ny, nz, nbands), got {values.shape!r}")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be finite")
    return values


def _regularize_sorted_energies(sorted_energies: FloatArray) -> FloatArray:
    regularized = np.asarray(sorted_energies, dtype=np.float64).copy()
    scale = max(1.0, float(np.max(np.abs(regularized))))
    step = np.finfo(np.float64).eps * scale * 16.0
    for index in range(1, regularized.size):
        if regularized[index] <= regularized[index - 1]:
            regularized[index] = regularized[index - 1] + step
    return regularized
