from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np


FERMI_ENERGY = 0.5
DEFAULT_RECIPROCAL_VECTORS = np.diag([3.0, 3.0]).astype(np.float64)


def time_best(kernel, *, repeats: int) -> float:
    kernel()
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        kernel()
        best = min(best, time.perf_counter() - start)
    return float(best)


def profile_top_cumulative(kernel, *, limit: int) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    kernel()
    profiler.disable()

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(limit)
    return stream.getvalue().rstrip()


def centered_fractional_kpoint(indices: tuple[int, int], grid_shape: tuple[int, int]) -> np.ndarray:
    grid = np.asarray(grid_shape, dtype=np.int64)
    half_grid = grid // 2
    integer_indices = np.asarray(indices, dtype=np.int64)
    centered = np.mod(integer_indices + half_grid, grid) - half_grid
    return centered.astype(np.float64) / grid.astype(np.float64)


def free_electron_bands_2d(
    reciprocal_vectors: np.ndarray = DEFAULT_RECIPROCAL_VECTORS,
    grid_shape: tuple[int, int] = (48, 48),
) -> np.ndarray:
    eigenvalues = np.empty((*grid_shape, 2), dtype=np.float64)

    nx, ny = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            kvec = reciprocal_vectors @ centered_fractional_kpoint((x_index, y_index), grid_shape)
            band_0 = 0.5 * float(np.dot(kvec, kvec))
            eigenvalues[x_index, y_index, 0] = band_0
            eigenvalues[x_index, y_index, 1] = band_0 + 0.25

    return eigenvalues


def shifted_response_bands_2d(
    reciprocal_vectors: np.ndarray = DEFAULT_RECIPROCAL_VECTORS,
    grid_shape: tuple[int, int] = (48, 48),
) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.empty((*grid_shape, 2), dtype=np.float64)
    target = np.empty((*grid_shape, 2), dtype=np.float64)
    shift = np.array([1.0, 0.0], dtype=np.float64)

    nx, ny = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            kvec = reciprocal_vectors @ centered_fractional_kpoint((x_index, y_index), grid_shape)
            base = 0.5 * float(np.dot(kvec, kvec)) - FERMI_ENERGY
            shifted = kvec + shift
            occupied[x_index, y_index, 0] = base
            occupied[x_index, y_index, 1] = base + 0.25
            target[x_index, y_index, 0] = 0.5 * float(np.dot(shifted, shifted)) - FERMI_ENERGY
            target[x_index, y_index, 1] = base + 0.5

    return occupied, target


def multiband_response_bands_2d(
    reciprocal_vectors: np.ndarray = DEFAULT_RECIPROCAL_VECTORS,
    grid_shape: tuple[int, int] = (48, 48),
    *,
    band_count: int,
    q_vector: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.empty((*grid_shape, band_count), dtype=np.float64)
    target = np.empty((*grid_shape, band_count), dtype=np.float64)
    if q_vector is None:
        q_vector = np.array([1.0, 0.75], dtype=np.float64)
    else:
        q_vector = np.asarray(q_vector, dtype=np.float64)

    nx, ny = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            kvec = reciprocal_vectors @ centered_fractional_kpoint((x_index, y_index), grid_shape)
            base = 0.5 * float(np.dot(kvec, kvec)) - FERMI_ENERGY
            shifted = kvec + q_vector
            target_base = 0.5 * float(np.dot(shifted, shifted)) - FERMI_ENERGY
            for band_index in range(band_count):
                occupied[x_index, y_index, band_index] = base + 0.08 * band_index
                target[x_index, y_index, band_index] = target_base + 0.12 * band_index + 0.35

    return occupied, target


def lindhard_bands_2d(
    reciprocal_vectors: np.ndarray = DEFAULT_RECIPROCAL_VECTORS,
    grid_shape: tuple[int, int] = (48, 48),
    *,
    q_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.empty((*grid_shape, 1), dtype=np.float64)
    target = np.empty((*grid_shape, 1), dtype=np.float64)
    qvec = np.array([q_value, 0.0], dtype=np.float64)

    nx, ny = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            kvec = reciprocal_vectors @ centered_fractional_kpoint((x_index, y_index), grid_shape)
            occupied[x_index, y_index, 0] = 0.5 * float(np.dot(kvec, kvec)) - FERMI_ENERGY
            shifted = kvec + qvec
            target[x_index, y_index, 0] = 0.5 * float(np.dot(shifted, shifted)) - FERMI_ENERGY

    return occupied, target
