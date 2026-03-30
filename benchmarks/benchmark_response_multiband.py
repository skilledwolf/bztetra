from __future__ import annotations

import argparse
import time

import numpy as np

from tetrabz import fermigr
from tetrabz import polcmplx
from tetrabz import prepare_response_problem


FERMI_ENERGY = 0.5


def main() -> None:
    args = _parse_args()
    grid_shape = (args.grid, args.grid, args.grid)
    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    sample_energies = np.linspace(0.0, 1.25, args.energy_count, dtype=np.float64)
    complex_sample_energies = 1j * np.linspace(0.1, 2.0, args.energy_count, dtype=np.float64)
    occupied, target = _synthetic_multiband_response_bands(reciprocal_vectors, grid_shape, band_count=args.bands)
    prepared = prepare_response_problem(
        reciprocal_vectors,
        occupied,
        target,
        weight_grid_shape=grid_shape,
        method=args.method,
    )

    tasks = [
        (
            "fermigr",
            lambda: fermigr(
                reciprocal_vectors,
                occupied,
                target,
                sample_energies,
                weight_grid_shape=grid_shape,
                method=args.method,
            ),
        ),
        (
            "polcmplx",
            lambda: polcmplx(
                reciprocal_vectors,
                occupied,
                target,
                complex_sample_energies,
                weight_grid_shape=grid_shape,
                method=args.method,
            ),
        ),
        (
            "fermigr_prepared",
            lambda: prepared.fermigr(sample_energies),
        ),
        (
            "polcmplx_prepared",
            lambda: prepared.polcmplx(complex_sample_energies),
        ),
    ]

    print(f"Benchmark grid: {args.grid}^3")
    print(f"Response bands: {args.bands} x {args.bands}")
    print(f"Sample energies: {args.energy_count}")
    print(f"Tetrahedron method: {args.method}")
    print(f"Warm repeats: {args.repeats}")
    print()

    for label, kernel in tasks:
        elapsed = _time_kernel(kernel, repeats=args.repeats)
        print(f"{label:>17s}: {elapsed:.3f} s")


def _time_kernel(kernel, *, repeats: int) -> float:
    kernel()
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        kernel()
        best = min(best, time.perf_counter() - start)
    return float(best)


def _synthetic_multiband_response_bands(
    reciprocal_vectors: np.ndarray,
    grid_shape: tuple[int, int, int],
    *,
    band_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.empty((*grid_shape, band_count), dtype=np.float64)
    target = np.empty((*grid_shape, band_count), dtype=np.float64)

    nx, ny, nz = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = reciprocal_vectors @ _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                base = 0.5 * float(np.dot(kvec, kvec)) - FERMI_ENERGY
                for band_index in range(band_count):
                    occupied[x_index, y_index, z_index, band_index] = base + 0.08 * band_index
                    target[x_index, y_index, z_index, band_index] = base + 0.12 * band_index + 0.35

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
    parser = argparse.ArgumentParser(
        description="Benchmark multiband frequency-response workloads for fermigr/polcmplx, including prepared-response reuse."
    )
    parser.add_argument("--grid", type=int, default=16, help="Cubic grid size to benchmark (default: 16)")
    parser.add_argument("--bands", type=int, default=6, help="Number of occupied and target bands (default: 6)")
    parser.add_argument("--energy-count", type=int, default=16, help="Number of frequency samples (default: 16)")
    parser.add_argument("--method", choices=("linear", "optimized"), default="optimized", help="Tetrahedron method to benchmark")
    parser.add_argument("--repeats", type=int, default=3, help="How many warm runs to time per workload (default: 3)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
