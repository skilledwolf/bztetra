from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .geometry import IntegrationMesh
from .geometry import trilinear_interpolation_indices


FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


def normalize_eigenvalues(eigenvalues: npt.ArrayLike) -> tuple[FloatArray, tuple[int, int, int]]:
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("expected eigenvalues with shape (nx, ny, nz, nbands)")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be finite")

    energy_grid_shape = tuple(int(item) for item in values.shape[:3])
    band_count = values.shape[3]
    flattened = np.ascontiguousarray(np.transpose(values, (2, 1, 0, 3))).reshape(-1, band_count)
    return flattened, energy_grid_shape


def normalize_energy_samples(energies: npt.ArrayLike) -> FloatArray:
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("expected a one-dimensional energy grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("energy samples must be finite")
    return values


def normalize_complex_energy_samples(energies: npt.ArrayLike) -> ComplexArray:
    values = np.asarray(energies, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("expected a one-dimensional complex energy grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("complex energy samples must be finite")
    return values


def interpolated_tetrahedron_energies(mesh: IntegrationMesh, eig_flat: FloatArray) -> FloatArray:
    tetrahedron_count = mesh.tetrahedron_count
    band_count = eig_flat.shape[1]
    tetra_band_energies = np.empty((tetrahedron_count, 4, band_count), dtype=np.float64)

    for tetrahedron_index in range(tetrahedron_count):
        tetra_band_energies[tetrahedron_index] = (
            mesh.tetrahedron_weight_matrix @ eig_flat[mesh.global_point_indices[tetrahedron_index]]
        )

    return tetra_band_energies


def interpolate_local_values(mesh: IntegrationMesh, local_values: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    if local_values.shape[0] != mesh.local_point_count:
        raise ValueError("local values must be indexed by the mesh local-point axis")

    if not mesh.interpolation_required:
        return local_values

    if mesh.fractional_kpoints is None:
        raise ValueError("interpolated mesh requires fractional k-points")

    flattened_features = local_values.reshape(local_values.shape[0], -1)
    output_flat = np.zeros(
        (int(np.prod(mesh.weight_grid_shape, dtype=np.int64)), flattened_features.shape[1]),
        dtype=flattened_features.dtype,
    )
    for local_index, kpoint in enumerate(mesh.fractional_kpoints):
        indices, weights = trilinear_interpolation_indices(mesh.weight_grid_shape, kpoint)
        for feature_index in range(flattened_features.shape[1]):
            np.add.at(
                output_flat[:, feature_index],
                indices,
                weights * flattened_features[local_index, feature_index],
            )
    return output_flat.reshape((output_flat.shape[0],) + local_values.shape[1:])
