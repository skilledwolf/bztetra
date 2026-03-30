from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import numpy.typing as npt


GridShape2D = tuple[int, int]
IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
TriangleMethod = Literal["linear"]

LINEAR_METHOD = 1


@dataclass(frozen=True, slots=True)
class TriangleIntegrationMesh:
    """Serial geometry/indexing state shared by the 2D integration kernels."""

    method: int
    energy_grid_shape: GridShape2D
    weight_grid_shape: GridShape2D
    interpolation_required: bool
    triangle_offsets: IntArray
    global_point_indices: IntArray
    local_point_indices: IntArray
    fractional_kpoints: FloatArray | None
    interpolation_indices: IntArray | None
    interpolation_weights: FloatArray | None

    @property
    def triangle_count(self) -> int:
        return int(self.global_point_indices.shape[0])

    @property
    def local_point_count(self) -> int:
        if self.fractional_kpoints is None:
            return int(np.prod(self.energy_grid_shape, dtype=np.int64))
        return int(self.fractional_kpoints.shape[0])


def triangle_offsets(
    reciprocal_vectors: npt.ArrayLike,
    energy_grid_shape: GridShape2D | tuple[int, int],
) -> IntArray:
    basis = _normalize_reciprocal_vectors(reciprocal_vectors)
    grid = _normalize_grid_shape(energy_grid_shape)
    scaled_basis = basis / np.asarray(grid, dtype=np.float64)[None, :]

    main_diagonal = scaled_basis[:, 0] + scaled_basis[:, 1]
    cross_diagonal = scaled_basis[:, 0] - scaled_basis[:, 1]

    if np.dot(cross_diagonal, cross_diagonal) < np.dot(main_diagonal, main_diagonal):
        return np.array(
            [
                [[1, 0], [0, 0], [0, 1]],
                [[1, 0], [0, 1], [1, 1]],
            ],
            dtype=np.int64,
        )

    return np.array(
        [
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [1, 1], [0, 1]],
        ],
        dtype=np.int64,
    )


def build_integration_mesh(
    reciprocal_vectors: npt.ArrayLike,
    energy_grid_shape: GridShape2D | tuple[int, int],
    weight_grid_shape: GridShape2D | tuple[int, int] | None = None,
    *,
    method: int | TriangleMethod = "linear",
) -> TriangleIntegrationMesh:
    basis = _normalize_reciprocal_vectors(reciprocal_vectors)
    energy_grid = _normalize_grid_shape(energy_grid_shape)
    weight_grid = energy_grid if weight_grid_shape is None else _normalize_grid_shape(weight_grid_shape)
    return _build_integration_mesh_from_normalized_inputs(
        basis,
        energy_grid,
        weight_grid,
        _normalize_method(method),
    )


def cached_integration_mesh(
    reciprocal_vectors: npt.ArrayLike,
    energy_grid_shape: GridShape2D | tuple[int, int],
    weight_grid_shape: GridShape2D | tuple[int, int] | None = None,
    *,
    method: int | TriangleMethod = "linear",
) -> TriangleIntegrationMesh:
    basis = _normalize_reciprocal_vectors(reciprocal_vectors)
    energy_grid = _normalize_grid_shape(energy_grid_shape)
    weight_grid = energy_grid if weight_grid_shape is None else _normalize_grid_shape(weight_grid_shape)
    basis_key = tuple(float(value) for value in basis.reshape(-1))
    return _cached_integration_mesh(
        _normalize_method(method),
        energy_grid,
        weight_grid,
        basis_key,
    )


@lru_cache(maxsize=32)
def _cached_integration_mesh(
    method: int,
    energy_grid_shape: GridShape2D,
    weight_grid_shape: GridShape2D,
    reciprocal_vector_key: tuple[float, ...],
) -> TriangleIntegrationMesh:
    basis = np.asarray(reciprocal_vector_key, dtype=np.float64).reshape(2, 2)
    return _build_integration_mesh_from_normalized_inputs(
        basis,
        energy_grid_shape,
        weight_grid_shape,
        method,
    )


def _build_integration_mesh_from_normalized_inputs(
    reciprocal_vectors: FloatArray,
    energy_grid_shape: GridShape2D,
    weight_grid_shape: GridShape2D,
    method: int,
) -> TriangleIntegrationMesh:
    interpolation_required = energy_grid_shape != weight_grid_shape

    offsets = triangle_offsets(reciprocal_vectors, energy_grid_shape)
    global_indices = _build_global_point_indices(offsets, energy_grid_shape)

    fractional_kpoints: FloatArray | None = None
    interpolation_indices: IntArray | None = None
    interpolation_weights: FloatArray | None = None
    if interpolation_required:
        local_indices, fractional_kpoints = _localize_point_indices(global_indices, energy_grid_shape)
        interpolation_indices, interpolation_weights = _build_interpolation_stencils(
            weight_grid_shape,
            fractional_kpoints,
        )
    else:
        local_indices = global_indices.copy()

    return TriangleIntegrationMesh(
        method=method,
        energy_grid_shape=energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        interpolation_required=interpolation_required,
        triangle_offsets=offsets,
        global_point_indices=global_indices,
        local_point_indices=local_indices,
        fractional_kpoints=fractional_kpoints,
        interpolation_indices=interpolation_indices,
        interpolation_weights=interpolation_weights,
    )


def bilinear_interpolation_indices(
    grid_shape: GridShape2D | tuple[int, int],
    fractional_kpoint: npt.ArrayLike,
) -> tuple[IntArray, FloatArray]:
    grid = _normalize_grid_shape(grid_shape)
    kpoint = np.asarray(fractional_kpoint, dtype=np.float64)
    if kpoint.shape != (2,):
        raise ValueError(f"expected a 2-vector k-point, got shape {kpoint.shape!r}")

    scaled = kpoint * np.asarray(grid, dtype=np.float64)
    lower = np.floor(scaled).astype(np.int64)
    frac = scaled - lower

    indices = np.empty(4, dtype=np.int64)
    weights = np.empty(4, dtype=np.float64)
    cursor = 0
    for dx in (0, 1):
        for dy in (0, 1):
            coords = np.array((lower[0] + dx, lower[1] + dy), dtype=np.int64)
            wrapped = np.mod(coords, np.asarray(grid, dtype=np.int64))
            indices[cursor] = _flatten_index(wrapped, grid)
            weights[cursor] = (
                (frac[0] ** dx) * ((1.0 - frac[0]) ** (1 - dx))
                * (frac[1] ** dy) * ((1.0 - frac[1]) ** (1 - dy))
            )
            cursor += 1

    return indices, weights


def _build_global_point_indices(offsets: IntArray, grid_shape: GridShape2D) -> IntArray:
    nx, ny = grid_shape
    triangle_count = 2 * int(np.prod(grid_shape, dtype=np.int64))
    global_indices = np.empty((triangle_count, 3), dtype=np.int64)

    cursor = 0
    grid = np.asarray(grid_shape, dtype=np.int64)
    for y_index in range(ny):
        for x_index in range(nx):
            origin = np.array((x_index, y_index), dtype=np.int64)
            for triangle in offsets:
                wrapped = np.mod(origin + triangle, grid)
                global_indices[cursor] = wrapped[:, 0] + nx * wrapped[:, 1]
                cursor += 1

    return global_indices


def _localize_point_indices(
    global_indices: IntArray,
    grid_shape: GridShape2D,
) -> tuple[IntArray, FloatArray]:
    point_count = int(np.prod(grid_shape, dtype=np.int64))
    global_to_local = np.full(point_count, -1, dtype=np.int64)
    local_indices = np.empty_like(global_indices)
    local_to_global = np.empty(point_count, dtype=np.int64)

    next_local = 0
    for triangle_index in range(global_indices.shape[0]):
        for point_index in range(global_indices.shape[1]):
            global_index = int(global_indices[triangle_index, point_index])
            local_index = global_to_local[global_index]
            if local_index == -1:
                local_index = next_local
                global_to_local[global_index] = next_local
                local_to_global[next_local] = global_index
                next_local += 1
            local_indices[triangle_index, point_index] = local_index

    fractional_kpoints = np.empty((next_local, 2), dtype=np.float64)
    nx, _ = grid_shape
    for local_index in range(next_local):
        global_index = int(local_to_global[local_index])
        x_index = global_index % nx
        y_index = global_index // nx
        fractional_kpoints[local_index] = (
            x_index / grid_shape[0],
            y_index / grid_shape[1],
        )

    return local_indices, fractional_kpoints


def _build_interpolation_stencils(
    grid_shape: GridShape2D,
    fractional_kpoints: FloatArray,
) -> tuple[IntArray, FloatArray]:
    point_count = fractional_kpoints.shape[0]
    interpolation_indices = np.empty((point_count, 4), dtype=np.int64)
    interpolation_weights = np.empty((point_count, 4), dtype=np.float64)

    for point_index in range(point_count):
        indices, weights = bilinear_interpolation_indices(grid_shape, fractional_kpoints[point_index])
        interpolation_indices[point_index] = indices
        interpolation_weights[point_index] = weights

    return interpolation_indices, interpolation_weights


def _flatten_index(coords: npt.ArrayLike, grid_shape: GridShape2D) -> np.int64:
    x_index, y_index = np.asarray(coords, dtype=np.int64)
    nx, _ = grid_shape
    return np.int64(x_index + nx * y_index)


def _normalize_grid_shape(shape: GridShape2D | tuple[int, int]) -> GridShape2D:
    values = tuple(int(item) for item in shape)
    if len(values) != 2 or any(item <= 0 for item in values):
        raise ValueError(f"expected two positive grid dimensions, got {shape!r}")
    return values


def _normalize_method(method: int | TriangleMethod) -> int:
    if method in (LINEAR_METHOD, "linear"):
        return LINEAR_METHOD
    raise ValueError("bztetra.twod currently supports only the linear triangle method")


def _normalize_reciprocal_vectors(reciprocal_vectors: npt.ArrayLike) -> FloatArray:
    basis = np.asarray(reciprocal_vectors, dtype=np.float64)
    if basis.shape != (2, 2):
        raise ValueError(
            "bztetra.twod expects reciprocal vectors with shape (2, 2), "
            f"got {basis.shape!r}"
        )
    return basis
