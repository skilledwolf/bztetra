from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._grids import FloatArray
from ._grids import normalize_eigenvalues


ComplexArray = npt.NDArray[np.complex128]


def _normalize_eigenvalue_pair(
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, tuple[int, int, int]]:
    occupied_flat, occupied_grid_shape = normalize_eigenvalues(occupied_eigenvalues)
    target_flat, target_grid_shape = normalize_eigenvalues(target_eigenvalues)
    if occupied_grid_shape != target_grid_shape:
        raise ValueError("occupied and target eigenvalue grids must share the same shape")
    return occupied_flat, target_flat, occupied_grid_shape


def _unflatten_pair_band_last(values: FloatArray, grid_shape: tuple[int, int, int]) -> FloatArray:
    target_band_count = values.shape[1]
    source_band_count = values.shape[2]
    reshaped = values.reshape(
        (grid_shape[2], grid_shape[1], grid_shape[0], target_band_count, source_band_count)
    )
    return np.transpose(reshaped, (2, 1, 0, 3, 4))


def _unflatten_energy_pair_band_last(
    values: FloatArray, grid_shape: tuple[int, int, int]
) -> FloatArray:
    energy_count = values.shape[1]
    target_band_count = values.shape[2]
    source_band_count = values.shape[3]
    reshaped = values.reshape(
        (
            grid_shape[2],
            grid_shape[1],
            grid_shape[0],
            energy_count,
            target_band_count,
            source_band_count,
        )
    )
    return np.transpose(reshaped, (3, 2, 1, 0, 4, 5))
