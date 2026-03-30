from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import numpy.typing as npt


GridShape = tuple[int, int, int]
IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
TetraMethod = Literal["linear", "optimized"]

LINEAR_METHOD = 1
OPTIMIZED_METHOD = 2

_LINEAR_WEIGHT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
_OPTIMIZED_WEIGHT_MATRIX = np.array(
    [
        [1440, 0, 30, 0, -38, 7, 17, -28, -56, 9, -46, 9, -38, -28, 17, 7, -18, -18, 12, -18],
        [0, 1440, 0, 30, -28, -38, 7, 17, 9, -56, 9, -46, 7, -38, -28, 17, -18, -18, -18, 12],
        [30, 0, 1440, 0, 17, -28, -38, 7, -46, 9, -56, 9, 17, 7, -38, -28, 12, -18, -18, -18],
        [0, 30, 0, 1440, 7, 17, -28, -38, 9, -46, 9, -56, -28, 17, 7, -38, -18, 12, -18, -18],
    ],
    dtype=np.float64,
) / 1260.0


@dataclass(frozen=True, slots=True)
class IntegrationMesh:
    """Serial geometry/indexing state shared by the integration kernels.

    All indices are zero-based, unlike the legacy Fortran implementation.
    `reciprocal_vectors` are still interpreted in the legacy column-vector
    convention for this low-level module so the geometry port stays auditable.
    """

    method: int
    energy_grid_shape: GridShape
    weight_grid_shape: GridShape
    interpolation_required: bool
    tetrahedron_weight_matrix: FloatArray
    tetrahedra_offsets: IntArray
    global_point_indices: IntArray
    local_point_indices: IntArray
    fractional_kpoints: FloatArray | None

    @property
    def tetrahedron_count(self) -> int:
        return int(self.global_point_indices.shape[0])

    @property
    def local_point_count(self) -> int:
        if self.fractional_kpoints is None:
            return int(np.prod(self.energy_grid_shape, dtype=np.int64))
        return int(self.fractional_kpoints.shape[0])


def tetrahedron_weight_matrix(method: int | TetraMethod) -> FloatArray:
    normalized = _normalize_method(method)
    if normalized == LINEAR_METHOD:
        return _LINEAR_WEIGHT_MATRIX.copy()
    return _OPTIMIZED_WEIGHT_MATRIX.copy()


def tetrahedron_offsets(
    reciprocal_vectors: npt.ArrayLike,
    energy_grid_shape: GridShape | tuple[int, int, int],
) -> IntArray:
    grid = _normalize_grid_shape(energy_grid_shape)
    basis = _normalize_reciprocal_vectors(reciprocal_vectors)
    scaled_basis = basis / np.asarray(grid, dtype=np.float64)[None, :]

    diagonal_vectors = np.empty((4, 3), dtype=np.float64)
    diagonal_vectors[0] = -scaled_basis[:, 0] + scaled_basis[:, 1] + scaled_basis[:, 2]
    diagonal_vectors[1] = scaled_basis[:, 0] - scaled_basis[:, 1] + scaled_basis[:, 2]
    diagonal_vectors[2] = scaled_basis[:, 0] + scaled_basis[:, 1] - scaled_basis[:, 2]
    diagonal_vectors[3] = scaled_basis[:, 0] + scaled_basis[:, 1] + scaled_basis[:, 2]
    shortest_diagonal = int(np.argmin(np.einsum("ij,ij->i", diagonal_vectors, diagonal_vectors)))

    start = np.zeros(4, dtype=np.int64)
    start[shortest_diagonal] = 1
    directions = np.eye(4, dtype=np.int64)
    directions[shortest_diagonal, shortest_diagonal] = -1

    offsets = np.empty((6, 20, 3), dtype=np.int64)
    tetrahedron_index = 0
    for axis_1 in range(3):
        for axis_2 in range(3):
            if axis_2 == axis_1:
                continue
            for axis_3 in range(3):
                if axis_3 in {axis_1, axis_2}:
                    continue
                corners = np.empty((20, 3), dtype=np.int64)
                corners[0] = start[:3]
                corners[1] = corners[0] + directions[:3, axis_1]
                corners[2] = corners[1] + directions[:3, axis_2]
                corners[3] = corners[2] + directions[:3, axis_3]

                corners[4] = 2 * corners[0] - corners[1]
                corners[5] = 2 * corners[1] - corners[2]
                corners[6] = 2 * corners[2] - corners[3]
                corners[7] = 2 * corners[3] - corners[0]

                corners[8] = 2 * corners[0] - corners[2]
                corners[9] = 2 * corners[1] - corners[3]
                corners[10] = 2 * corners[2] - corners[0]
                corners[11] = 2 * corners[3] - corners[1]

                corners[12] = 2 * corners[0] - corners[3]
                corners[13] = 2 * corners[1] - corners[0]
                corners[14] = 2 * corners[2] - corners[1]
                corners[15] = 2 * corners[3] - corners[2]

                corners[16] = corners[3] - corners[0] + corners[1]
                corners[17] = corners[0] - corners[1] + corners[2]
                corners[18] = corners[1] - corners[2] + corners[3]
                corners[19] = corners[2] - corners[3] + corners[0]

                offsets[tetrahedron_index] = corners
                tetrahedron_index += 1

    return offsets


def build_integration_mesh(
    reciprocal_vectors: npt.ArrayLike,
    energy_grid_shape: GridShape | tuple[int, int, int],
    weight_grid_shape: GridShape | tuple[int, int, int] | None = None,
    *,
    method: int | TetraMethod = "optimized",
) -> IntegrationMesh:
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
    energy_grid_shape: GridShape | tuple[int, int, int],
    weight_grid_shape: GridShape | tuple[int, int, int] | None = None,
    *,
    method: int | TetraMethod = "optimized",
) -> IntegrationMesh:
    """Return a cached mesh object for internal read-only kernel setup reuse."""
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
    energy_grid_shape: GridShape,
    weight_grid_shape: GridShape,
    reciprocal_vector_key: tuple[float, ...],
) -> IntegrationMesh:
    basis = np.asarray(reciprocal_vector_key, dtype=np.float64).reshape(3, 3)
    return _build_integration_mesh_from_normalized_inputs(
        basis,
        energy_grid_shape,
        weight_grid_shape,
        method,
    )


def _build_integration_mesh_from_normalized_inputs(
    reciprocal_vectors: FloatArray,
    energy_grid_shape: GridShape,
    weight_grid_shape: GridShape,
    method: int,
) -> IntegrationMesh:
    interpolation_required = energy_grid_shape != weight_grid_shape

    offsets = tetrahedron_offsets(reciprocal_vectors, energy_grid_shape)
    global_indices = _build_global_point_indices(offsets, energy_grid_shape)

    fractional_kpoints: FloatArray | None = None
    if interpolation_required:
        local_indices, fractional_kpoints = _localize_point_indices(global_indices, energy_grid_shape)
    else:
        local_indices = global_indices.copy()

    return IntegrationMesh(
        method=method,
        energy_grid_shape=energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        interpolation_required=interpolation_required,
        tetrahedron_weight_matrix=tetrahedron_weight_matrix(method),
        tetrahedra_offsets=offsets,
        global_point_indices=global_indices,
        local_point_indices=local_indices,
        fractional_kpoints=fractional_kpoints,
    )


def trilinear_interpolation_indices(
    grid_shape: GridShape | tuple[int, int, int],
    fractional_kpoint: npt.ArrayLike,
) -> tuple[IntArray, FloatArray]:
    grid = _normalize_grid_shape(grid_shape)
    kpoint = np.asarray(fractional_kpoint, dtype=np.float64)
    if kpoint.shape != (3,):
        raise ValueError(f"expected a 3-vector k-point, got shape {kpoint.shape!r}")

    scaled = kpoint * np.asarray(grid, dtype=np.float64)
    lower = np.floor(scaled).astype(np.int64)
    frac = scaled - lower

    indices = np.empty(8, dtype=np.int64)
    weights = np.empty(8, dtype=np.float64)
    cursor = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                coords = np.array((lower[0] + dx, lower[1] + dy, lower[2] + dz), dtype=np.int64)
                wrapped = np.mod(coords, np.asarray(grid, dtype=np.int64))
                indices[cursor] = _flatten_index(wrapped, grid)
                weights[cursor] = (
                    (frac[0] ** dx) * ((1.0 - frac[0]) ** (1 - dx))
                    * (frac[1] ** dy) * ((1.0 - frac[1]) ** (1 - dy))
                    * (frac[2] ** dz) * ((1.0 - frac[2]) ** (1 - dz))
                )
                cursor += 1

    return indices, weights


def _build_global_point_indices(offsets: IntArray, grid_shape: GridShape) -> IntArray:
    nx, ny, nz = grid_shape
    tetrahedron_count = 6 * int(np.prod(grid_shape, dtype=np.int64))
    global_indices = np.empty((tetrahedron_count, 20), dtype=np.int64)

    cursor = 0
    grid = np.asarray(grid_shape, dtype=np.int64)
    for z_index in range(nz):
        for y_index in range(ny):
            for x_index in range(nx):
                origin = np.array((x_index, y_index, z_index), dtype=np.int64)
                for tetrahedron in offsets:
                    wrapped = np.mod(origin + tetrahedron, grid)
                    global_indices[cursor] = (
                        wrapped[:, 0]
                        + nx * wrapped[:, 1]
                        + nx * ny * wrapped[:, 2]
                    )
                    cursor += 1

    return global_indices


def _localize_point_indices(
    global_indices: IntArray,
    grid_shape: GridShape,
) -> tuple[IntArray, FloatArray]:
    point_count = int(np.prod(grid_shape, dtype=np.int64))
    global_to_local = np.full(point_count, -1, dtype=np.int64)
    local_indices = np.empty_like(global_indices)
    local_to_global = np.empty(point_count, dtype=np.int64)

    next_local = 0
    for tetrahedron_index in range(global_indices.shape[0]):
        for point_index in range(global_indices.shape[1]):
            global_index = int(global_indices[tetrahedron_index, point_index])
            local_index = global_to_local[global_index]
            if local_index == -1:
                local_index = next_local
                global_to_local[global_index] = next_local
                local_to_global[next_local] = global_index
                next_local += 1
            local_indices[tetrahedron_index, point_index] = local_index

    fractional_kpoints = np.empty((next_local, 3), dtype=np.float64)
    nx, ny, _ = grid_shape
    stride_xy = nx * ny
    for local_index in range(next_local):
        global_index = int(local_to_global[local_index])
        x_index = global_index % nx
        y_index = (global_index // nx) % ny
        z_index = global_index // stride_xy
        fractional_kpoints[local_index] = (
            x_index / grid_shape[0],
            y_index / grid_shape[1],
            z_index / grid_shape[2],
        )

    return local_indices, fractional_kpoints


def _flatten_index(coords: npt.ArrayLike, grid_shape: GridShape) -> np.int64:
    x_index, y_index, z_index = np.asarray(coords, dtype=np.int64)
    nx, ny, _ = grid_shape
    return np.int64(x_index + nx * y_index + nx * ny * z_index)


def _normalize_grid_shape(shape: GridShape | tuple[int, int, int]) -> GridShape:
    values = tuple(int(item) for item in shape)
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError(f"expected three positive grid dimensions, got {shape!r}")
    return values


def _normalize_method(method: int | TetraMethod) -> int:
    if method in (LINEAR_METHOD, "linear"):
        return LINEAR_METHOD
    if method in (OPTIMIZED_METHOD, "optimized"):
        return OPTIMIZED_METHOD
    raise ValueError(f"unsupported tetrahedron method {method!r}")


def _normalize_reciprocal_vectors(reciprocal_vectors: npt.ArrayLike) -> FloatArray:
    basis = np.asarray(reciprocal_vectors, dtype=np.float64)
    if basis.shape != (3, 3):
        raise ValueError(f"expected reciprocal vectors with shape (3, 3), got {basis.shape!r}")
    return basis
