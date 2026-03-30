from __future__ import annotations

import argparse
import time

import numpy as np

from bztetra import static_polarization_weights


FERMI_ENERGY = 0.5


def main() -> None:
    args = _parse_args()
    energy_grid_shape = (args.grid, args.grid, args.grid)
    weight_grid_shape = (args.weight_grid, args.weight_grid, args.weight_grid)
    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)

    tasks = []
    for q_value in args.q_values:
        occupied, target = _lindhard_bands(reciprocal_vectors, energy_grid_shape, q_value=q_value)
        tasks.append(
            (
                f"lindhard q={q_value:.3f}",
                lambda occupied_bands=occupied, target_bands=target: static_polarization_weights(
                    reciprocal_vectors,
                    occupied_bands,
                    target_bands,
                    weight_grid_shape=weight_grid_shape,
                    method=args.method,
                ),
            )
        )

    occupied, target = _shifted_multiband_response_bands(reciprocal_vectors, energy_grid_shape)
    tasks.append(
        (
            "multiband shifted",
            lambda: static_polarization_weights(
                reciprocal_vectors,
                occupied,
                target,
                weight_grid_shape=weight_grid_shape,
                method=args.method,
            ),
        )
    )

    print(f"Energy grid: {args.grid}^3")
    print(f"Weight grid: {args.weight_grid}^3")
    print(f"Tetrahedron method: {args.method}")
    print(f"Warm repeats: {args.repeats}")
    print(f"q sweep: {', '.join(f'{value:.3f}' for value in args.q_values)}")
    print()

    for label, kernel in tasks:
        elapsed = _time_kernel(kernel, repeats=args.repeats)
        print(f"{label:>18s}: {elapsed:.3f} s")


def _time_kernel(kernel, *, repeats: int) -> float:
    kernel()
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        kernel()
        best = min(best, time.perf_counter() - start)
    return float(best)


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
                target_kvec = kvec + qvec
                target[x_index, y_index, z_index, 0] = 0.5 * float(np.dot(target_kvec, target_kvec)) - FERMI_ENERGY

    return occupied, target


def _shifted_multiband_response_bands(
    reciprocal_vectors: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.empty((*grid_shape, 2), dtype=np.float64)
    target = np.empty((*grid_shape, 2), dtype=np.float64)
    shift = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    nx, ny, nz = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = reciprocal_vectors @ _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                base = 0.5 * float(np.dot(kvec, kvec)) - FERMI_ENERGY
                shifted = kvec + shift
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


def _parse_q_values(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one q value")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark static polarization workloads, including small-q, Kohn-anomaly, and multiband cases."
    )
    parser.add_argument("--grid", type=int, default=24, help="Cubic energy grid size to benchmark (default: 24)")
    parser.add_argument(
        "--weight-grid",
        type=int,
        default=None,
        help="Cubic weight grid size to benchmark (default: match --grid)",
    )
    parser.add_argument("--method", choices=("linear", "optimized"), default="optimized", help="Tetrahedron method to benchmark")
    parser.add_argument("--repeats", type=int, default=3, help="How many warm runs to time per workload (default: 3)")
    parser.add_argument(
        "--q-values",
        type=_parse_q_values,
        default=(0.125, 2.0),
        help="Comma-separated Lindhard q values to benchmark (default: 0.125,2.0)",
    )
    args = parser.parse_args()
    if args.weight_grid is None:
        args.weight_grid = args.grid
    return args


if __name__ == "__main__":
    main()
