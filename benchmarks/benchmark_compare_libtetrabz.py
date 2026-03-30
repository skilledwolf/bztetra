from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import libtetrabz
import numpy as np

import bztetra


FloatArray = np.ndarray
ComplexArray = np.ndarray


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    legacy_seconds: float
    port_seconds: float
    prepared_seconds: float | None
    max_abs_diff: float
    notes: str = ""


def main() -> None:
    args = _parse_args()
    grid_shape = (args.grid, args.grid, args.grid)

    bvec, scalar_eigenvalues = _free_electron_case(grid_shape)
    _, occupied_eigenvalues, target_eigenvalues = _free_electron_response_case(grid_shape)
    prepared_response = bztetra.prepare_response_evaluator(
        bvec,
        occupied_eigenvalues - 0.5,
        target_eigenvalues - 0.5,
        weight_grid_shape=grid_shape,
        method="optimized",
    )

    dos_energies = _legacy_dos_energy_points()
    fermi_golden_rule_energies = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float64)
    complex_frequency_energies = np.array([-2.0 + 1.0j, 0.0 + 2.0j, 1.0 - 0.5j], dtype=np.complex128)
    electrons_per_spin = _legacy_electron_count_per_spin(bvec)

    results = [
        _benchmark_array_case(
            "occupation_weights",
            lambda: libtetrabz.occ(bvec, scalar_eigenvalues - 0.5),
            lambda: bztetra.occupation_weights(
                bvec,
                scalar_eigenvalues - 0.5,
                weight_grid_shape=grid_shape,
                method="optimized",
            ),
            repeats=args.repeats,
        ),
        _benchmark_fermieng_case(
            bvec,
            scalar_eigenvalues,
            electrons_per_spin,
            grid_shape=grid_shape,
            repeats=args.repeats,
        ),
        _benchmark_array_case(
            "density_of_states_weights",
            lambda: libtetrabz.dos(bvec, scalar_eigenvalues, dos_energies),
            lambda: bztetra.density_of_states_weights(
                bvec,
                scalar_eigenvalues,
                dos_energies,
                weight_grid_shape=grid_shape,
                method="optimized",
            ),
            port_normalizer=_normalize_port_dos_output,
            repeats=args.repeats,
        ),
        _benchmark_array_case(
            "integrated_density_of_states_weights",
            lambda: libtetrabz.intdos(bvec, scalar_eigenvalues, dos_energies),
            lambda: bztetra.integrated_density_of_states_weights(
                bvec,
                scalar_eigenvalues,
                dos_energies,
                weight_grid_shape=grid_shape,
                method="optimized",
            ),
            port_normalizer=_normalize_port_dos_output,
            repeats=args.repeats,
        ),
        _benchmark_array_case(
            "phase_space_overlap_weights",
            lambda: libtetrabz.dblstep(bvec, occupied_eigenvalues - 0.5, target_eigenvalues - 0.5),
            lambda: bztetra.phase_space_overlap_weights(
                bvec,
                occupied_eigenvalues - 0.5,
                target_eigenvalues - 0.5,
                weight_grid_shape=grid_shape,
                method="optimized",
            ),
            port_normalizer=_swap_pair_axes,
            prepared_callable=prepared_response.phase_space_overlap_weights,
            prepared_normalizer=_swap_pair_axes,
            repeats=args.repeats,
        ),
        _benchmark_array_case(
            "nesting_function_weights",
            lambda: libtetrabz.dbldelta(bvec, occupied_eigenvalues - 0.5, target_eigenvalues - 0.5),
            lambda: bztetra.nesting_function_weights(
                bvec,
                occupied_eigenvalues - 0.5,
                target_eigenvalues - 0.5,
                weight_grid_shape=grid_shape,
                method="optimized",
            ),
            port_normalizer=_swap_pair_axes,
            prepared_callable=prepared_response.nesting_function_weights,
            prepared_normalizer=_swap_pair_axes,
            repeats=args.repeats,
        ),
        _benchmark_array_case(
            "static_polarization_weights",
            lambda: libtetrabz.polstat(bvec, occupied_eigenvalues - 0.5, target_eigenvalues - 0.5),
            lambda: bztetra.static_polarization_weights(
                bvec,
                occupied_eigenvalues - 0.5,
                target_eigenvalues - 0.5,
                weight_grid_shape=grid_shape,
                method="optimized",
            ),
            port_normalizer=_swap_pair_axes,
            prepared_callable=prepared_response.static_polarization_weights,
            prepared_normalizer=_swap_pair_axes,
            repeats=args.repeats,
        ),
        _benchmark_array_case(
            "fermi_golden_rule_weights",
            lambda: libtetrabz.fermigr(bvec, occupied_eigenvalues - 0.5, target_eigenvalues - 0.5, fermi_golden_rule_energies),
            lambda: bztetra.fermi_golden_rule_weights(
                bvec,
                occupied_eigenvalues - 0.5,
                target_eigenvalues - 0.5,
                fermi_golden_rule_energies,
                weight_grid_shape=grid_shape,
                method="optimized",
            ),
            port_normalizer=_normalize_port_frequency_output,
            prepared_callable=lambda: prepared_response.fermi_golden_rule_weights(fermi_golden_rule_energies),
            prepared_normalizer=_normalize_port_frequency_output,
            repeats=args.repeats,
        ),
        _benchmark_array_case(
            "complex_frequency_polarization_weights",
            lambda: libtetrabz.polcmplx(bvec, occupied_eigenvalues - 0.5, target_eigenvalues - 0.5, complex_frequency_energies),
            lambda: bztetra.complex_frequency_polarization_weights(
                bvec,
                occupied_eigenvalues - 0.5,
                target_eigenvalues - 0.5,
                complex_frequency_energies,
                weight_grid_shape=grid_shape,
                method="optimized",
            ),
            port_normalizer=_normalize_port_frequency_output,
            prepared_callable=lambda: prepared_response.complex_frequency_polarization_weights(complex_frequency_energies),
            prepared_normalizer=_normalize_port_frequency_output,
            repeats=args.repeats,
        ),
    ]

    print(f"Comparison grid: {args.grid}^3")
    print(f"Warm repeats: {args.repeats}")
    print("Legacy reference: libtetrabz 0.1.2")
    print()
    print(
        f"{'case':<38} {'legacy(s)':>10} {'bztetra(s)':>11} {'prepared(s)':>12} "
        f"{'speedup':>9} {'max|diff|':>12}"
    )
    for result in results:
        prepared = "-" if result.prepared_seconds is None else f"{result.prepared_seconds:>12.3f}"
        speedup = result.legacy_seconds / result.port_seconds
        print(
            f"{result.name:<38} {result.legacy_seconds:>10.3f} {result.port_seconds:>11.3f} "
            f"{prepared} {speedup:>9.2f} {result.max_abs_diff:>12.3e}"
        )
        if result.notes:
            print(f"  note: {result.notes}")


def _benchmark_array_case(
    name: str,
    legacy_callable,
    port_callable,
    *,
    port_normalizer=lambda array: array,
    prepared_callable=None,
    prepared_normalizer=lambda array: array,
    repeats: int,
) -> BenchmarkResult:
    legacy_output = legacy_callable()
    port_output = port_callable()
    prepared_output = None if prepared_callable is None else prepared_callable()

    legacy_seconds = _time_best(legacy_callable, repeats=repeats)
    port_seconds = _time_best(port_callable, repeats=repeats)
    prepared_seconds = None if prepared_callable is None else _time_best(prepared_callable, repeats=repeats)

    normalized_legacy = np.asarray(legacy_output)
    normalized_port = np.asarray(port_normalizer(port_output))
    if normalized_legacy.shape != normalized_port.shape:
        raise ValueError(f"{name}: shape mismatch {normalized_legacy.shape!r} vs {normalized_port.shape!r}")

    if prepared_output is not None:
        normalized_prepared = np.asarray(prepared_normalizer(prepared_output))
        if normalized_legacy.shape != normalized_prepared.shape:
            raise ValueError(
                f"{name}: prepared shape mismatch {normalized_legacy.shape!r} vs {normalized_prepared.shape!r}"
            )
        np.testing.assert_allclose(normalized_port, normalized_prepared, rtol=1.0e-10, atol=1.0e-12)

    max_abs_diff = float(np.max(np.abs(normalized_legacy - normalized_port)))
    return BenchmarkResult(
        name=name,
        legacy_seconds=legacy_seconds,
        port_seconds=port_seconds,
        prepared_seconds=prepared_seconds,
        max_abs_diff=max_abs_diff,
    )


def _benchmark_fermieng_case(
    bvec: np.ndarray,
    eigenvalues: np.ndarray,
    electrons_per_spin: float,
    *,
    grid_shape: tuple[int, int, int],
    repeats: int,
) -> BenchmarkResult:
    legacy_ef, legacy_weights, legacy_iterations = libtetrabz.fermieng(bvec, eigenvalues, electrons_per_spin)
    port_result = bztetra.solve_fermi_energy(
        bvec,
        eigenvalues,
        electrons_per_spin,
        weight_grid_shape=grid_shape,
        method="optimized",
    )

    legacy_seconds = _time_best(lambda: libtetrabz.fermieng(bvec, eigenvalues, electrons_per_spin), repeats=repeats)
    port_seconds = _time_best(
        lambda: bztetra.solve_fermi_energy(
            bvec,
            eigenvalues,
            electrons_per_spin,
            weight_grid_shape=grid_shape,
            method="optimized",
        ),
        repeats=repeats,
    )

    weight_diff = float(np.max(np.abs(np.asarray(legacy_weights) - np.asarray(port_result.weights))))
    ef_diff = abs(float(legacy_ef) - float(port_result.fermi_energy))
    notes = (
        f"|ef diff|={ef_diff:.3e}, legacy_iter={int(legacy_iterations)}, "
        f"tetrabz_iter={int(port_result.iterations)}"
    )
    return BenchmarkResult(
        name="solve_fermi_energy",
        legacy_seconds=legacy_seconds,
        port_seconds=port_seconds,
        prepared_seconds=None,
        max_abs_diff=weight_diff,
        notes=notes,
    )


def _time_best(callable_, *, repeats: int) -> float:
    callable_()
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        best = min(best, time.perf_counter() - start)
    return float(best)


def _normalize_port_dos_output(array: FloatArray) -> FloatArray:
    return np.transpose(np.asarray(array), (1, 2, 3, 4, 0))


def _normalize_port_frequency_output(array: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(array), (1, 2, 3, 5, 4, 0))


def _swap_pair_axes(array: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(array), (0, 1, 2, 4, 3))


def _legacy_electron_count_per_spin(bvec: np.ndarray) -> float:
    vbz = float(np.linalg.det(bvec))
    return (4.0 * np.pi / 3.0 + np.sqrt(2.0) * np.pi / 3.0) / vbz


def _legacy_dos_energy_points() -> FloatArray:
    radii = 0.2 * np.arange(1, 6, dtype=np.float64)
    return 0.5 * radii * radii


def _free_electron_case(grid_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    bvec = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    return bvec, _make_eigenvalues(bvec, grid_shape)


def _free_electron_response_case(grid_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bvec, eig1 = _free_electron_case(grid_shape)
    eig2 = _make_response_eigenvalues(bvec, grid_shape)
    return bvec, eig1, eig2


def _make_eigenvalues(bvec: np.ndarray, grid_shape: tuple[int, int, int]) -> FloatArray:
    nx, ny, nz = grid_shape
    eigenvalues = np.empty((nx, ny, nz, 2), dtype=np.float64)
    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = bvec @ _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                base = 0.5 * float(np.dot(kvec, kvec))
                eigenvalues[x_index, y_index, z_index, 0] = base
                eigenvalues[x_index, y_index, z_index, 1] = base + 0.25
    return eigenvalues


def _make_response_eigenvalues(bvec: np.ndarray, grid_shape: tuple[int, int, int]) -> FloatArray:
    nx, ny, nz = grid_shape
    eigenvalues = np.empty((nx, ny, nz, 2), dtype=np.float64)
    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = bvec @ _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                base = 0.5 * float(np.dot(kvec, kvec))
                shifted = kvec.copy()
                shifted[0] += 1.0
                eigenvalues[x_index, y_index, z_index, 0] = 0.5 * float(np.dot(shifted, shifted))
                eigenvalues[x_index, y_index, z_index, 1] = base + 0.5
    return eigenvalues


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
        description="Benchmark bztetra against the published libtetrabz wrapper on the legacy free-electron workloads."
    )
    parser.add_argument("--grid", type=int, default=8, help="Cubic grid size used for the comparison workloads (default: 8)")
    parser.add_argument("--repeats", type=int, default=3, help="How many warm runs to time per implementation (default: 3)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
