from __future__ import annotations

import argparse
import time

import numpy as np

from tetrabz import dbldelta
from tetrabz import dblstep
from tetrabz import dos
from tetrabz import fermigr
from tetrabz import intdos
from tetrabz import occ
from tetrabz import polcmplx
from tetrabz import polstat


FERMI_ENERGY = 0.5


def main() -> None:
    args = _parse_args()
    grid_shape = (args.grid, args.grid, args.grid)
    sample_energies = np.linspace(0.0, 1.25, args.energy_count, dtype=np.float64)
    complex_sample_energies = 1j * np.linspace(0.1, 2.0, args.energy_count, dtype=np.float64)

    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    scalar_bands = _free_electron_bands(reciprocal_vectors, grid_shape)
    response_occupied_bands, response_target_bands = _free_electron_response_bands(reciprocal_vectors, grid_shape)
    occupied_bands, target_bands = _lindhard_bands(reciprocal_vectors, grid_shape, q_value=args.q_value)

    tasks = [
        (
            "occ",
            lambda: occ(
                reciprocal_vectors,
                scalar_bands,
                weight_grid_shape=grid_shape,
                method=args.method,
                fermi_energy=FERMI_ENERGY,
            ),
        ),
        (
            "dos",
            lambda: dos(
                reciprocal_vectors,
                scalar_bands,
                sample_energies,
                weight_grid_shape=grid_shape,
                method=args.method,
            ),
        ),
        (
            "intdos",
            lambda: intdos(
                reciprocal_vectors,
                scalar_bands,
                sample_energies,
                weight_grid_shape=grid_shape,
                method=args.method,
            ),
        ),
        (
            "dblstep",
            lambda: dblstep(
                reciprocal_vectors,
                response_occupied_bands,
                response_target_bands,
                weight_grid_shape=grid_shape,
                method=args.method,
            ),
        ),
        (
            "dbldelta",
            lambda: dbldelta(
                reciprocal_vectors,
                response_occupied_bands,
                response_target_bands,
                weight_grid_shape=grid_shape,
                method=args.method,
            ),
        ),
        (
            "polstat",
            lambda: polstat(
                reciprocal_vectors,
                occupied_bands,
                target_bands,
                weight_grid_shape=grid_shape,
                method=args.method,
            ),
        ),
        (
            "fermigr",
            lambda: fermigr(
                reciprocal_vectors,
                response_occupied_bands,
                response_target_bands,
                sample_energies,
                weight_grid_shape=grid_shape,
                method=args.method,
            ),
        ),
        (
            "polcmplx",
            lambda: polcmplx(
                reciprocal_vectors,
                response_occupied_bands,
                response_target_bands,
                complex_sample_energies,
                weight_grid_shape=grid_shape,
                method=args.method,
            ),
        ),
    ]

    print(f"Benchmark grid: {args.grid}^3")
    print(f"Tetrahedron method: {args.method}")
    print(f"Sample energies: {args.energy_count}")
    print(f"Lindhard q-value: {args.q_value:.6f}")
    print()

    for label, kernel in tasks:
        elapsed = _time_kernel(kernel, repeats=args.repeats)
        print(f"{label:>7s}: {elapsed:.3f} s (best of {args.repeats} warm runs)")


def _time_kernel(kernel, *, repeats: int) -> float:
    kernel()
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        kernel()
        best = min(best, time.perf_counter() - start)
    return float(best)


def _free_electron_bands(
    reciprocal_vectors: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    eigenvalues = np.empty((*grid_shape, 2), dtype=np.float64)

    nx, ny, nz = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = reciprocal_vectors @ _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                band_0 = 0.5 * float(np.dot(kvec, kvec))
                eigenvalues[x_index, y_index, z_index, 0] = band_0
                eigenvalues[x_index, y_index, z_index, 1] = band_0 + 0.25

    return eigenvalues


def _lindhard_bands(
    reciprocal_vectors: np.ndarray,
    grid_shape: tuple[int, int, int],
    *,
    q_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.empty((*grid_shape, 1), dtype=np.float64)
    target = np.empty((*grid_shape, 1), dtype=np.float64)
    qvec = np.array([q_value, 0.0, 0.0], dtype=np.float64)

    nx, ny, nz = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = reciprocal_vectors @ _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                occupied[x_index, y_index, z_index, 0] = 0.5 * float(np.dot(kvec, kvec)) - FERMI_ENERGY
                shifted = kvec + qvec
                target[x_index, y_index, z_index, 0] = 0.5 * float(np.dot(shifted, shifted)) - FERMI_ENERGY

    return occupied, target


def _free_electron_response_bands(
    reciprocal_vectors: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.empty((*grid_shape, 2), dtype=np.float64)
    target = np.empty((*grid_shape, 2), dtype=np.float64)

    nx, ny, nz = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = reciprocal_vectors @ _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                base = 0.5 * float(np.dot(kvec, kvec)) - FERMI_ENERGY
                shifted = kvec.copy()
                shifted[0] = shifted[0] + 1.0
                occupied[x_index, y_index, z_index, 0] = base
                occupied[x_index, y_index, z_index, 1] = base + 0.25
                target[x_index, y_index, z_index, 0] = 0.5 * float(np.dot(shifted, shifted)) - FERMI_ENERGY
                target[x_index, y_index, z_index, 1] = base + 0.5

    return occupied, target


def _centered_fractional_kpoint(
    indices: tuple[int, int, int],
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    grid = np.asarray(grid_shape, dtype=np.int64)
    half_grid = grid // 2
    integer_indices = np.asarray(indices, dtype=np.int64)
    centered = np.mod(integer_indices + half_grid, grid) - half_grid
    return centered.astype(np.float64) / grid.astype(np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the current tetrabz hot paths on a free-electron workload.")
    parser.add_argument("--grid", type=int, default=24, help="Cubic grid size to benchmark (default: 24)")
    parser.add_argument("--energy-count", type=int, default=16, help="Number of DOS sample energies (default: 16)")
    parser.add_argument("--q-value", type=float, default=2.0, help="Momentum transfer for the Lindhard benchmark (default: 2.0)")
    parser.add_argument("--method", choices=("linear", "optimized"), default="optimized", help="Tetrahedron method to benchmark")
    parser.add_argument("--repeats", type=int, default=3, help="How many warm runs to time per kernel (default: 3)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
