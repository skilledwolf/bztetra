from __future__ import annotations

import argparse

import numpy as np

from bztetra.twod import complex_frequency_polarization_weights
from bztetra.twod import density_of_states_weights
from bztetra.twod import fermi_golden_rule_weights
from bztetra.twod import integrated_density_of_states_weights
from bztetra.twod import nesting_function_weights
from bztetra.twod import occupation_weights
from bztetra.twod import phase_space_overlap_weights
from bztetra.twod import prepare_response_evaluator
from bztetra.twod import static_polarization_weights

try:
    from benchmarks._twod_benchmark_support import DEFAULT_RECIPROCAL_VECTORS
    from benchmarks._twod_benchmark_support import free_electron_bands_2d
    from benchmarks._twod_benchmark_support import lindhard_bands_2d
    from benchmarks._twod_benchmark_support import profile_top_cumulative
    from benchmarks._twod_benchmark_support import shifted_response_bands_2d
    from benchmarks._twod_benchmark_support import time_best
except ImportError:  # pragma: no cover - direct script execution
    from _twod_benchmark_support import DEFAULT_RECIPROCAL_VECTORS
    from _twod_benchmark_support import free_electron_bands_2d
    from _twod_benchmark_support import lindhard_bands_2d
    from _twod_benchmark_support import profile_top_cumulative
    from _twod_benchmark_support import shifted_response_bands_2d
    from _twod_benchmark_support import time_best


FERMI_ENERGY = 0.5


def main() -> None:
    args = _parse_args()
    grid_shape = (args.grid, args.grid)
    reciprocal_vectors = DEFAULT_RECIPROCAL_VECTORS
    scalar_bands = free_electron_bands_2d(reciprocal_vectors, grid_shape)
    response_occupied_bands, response_target_bands = shifted_response_bands_2d(reciprocal_vectors, grid_shape)
    occupied_bands, target_bands = lindhard_bands_2d(reciprocal_vectors, grid_shape, q_value=args.q_value)
    sample_energies = np.linspace(0.0, 1.25, args.energy_count, dtype=np.float64)
    complex_sample_energies = 1j * np.linspace(0.1, 2.0, args.energy_count, dtype=np.float64)
    prepared_response = prepare_response_evaluator(
        reciprocal_vectors,
        response_occupied_bands,
        response_target_bands,
        weight_grid_shape=grid_shape,
        method="linear",
    )

    tasks = [
        (
            "occupation_weights",
            lambda: occupation_weights(
                reciprocal_vectors,
                scalar_bands,
                weight_grid_shape=grid_shape,
                method="linear",
                fermi_energy=FERMI_ENERGY,
            ),
        ),
        (
            "density_of_states_weights",
            lambda: density_of_states_weights(
                reciprocal_vectors,
                scalar_bands,
                sample_energies,
                weight_grid_shape=grid_shape,
                method="linear",
            ),
        ),
        (
            "integrated_density_of_states_weights",
            lambda: integrated_density_of_states_weights(
                reciprocal_vectors,
                scalar_bands,
                sample_energies,
                weight_grid_shape=grid_shape,
                method="linear",
            ),
        ),
        (
            "phase_space_overlap_weights",
            lambda: phase_space_overlap_weights(
                reciprocal_vectors,
                response_occupied_bands,
                response_target_bands,
                weight_grid_shape=grid_shape,
                method="linear",
            ),
        ),
        (
            "nesting_function_weights",
            lambda: nesting_function_weights(
                reciprocal_vectors,
                response_occupied_bands,
                response_target_bands,
                weight_grid_shape=grid_shape,
                method="linear",
            ),
        ),
        (
            "static_polarization_weights",
            lambda: static_polarization_weights(
                reciprocal_vectors,
                occupied_bands,
                target_bands,
                weight_grid_shape=grid_shape,
                method="linear",
            ),
        ),
        (
            "fermi_golden_rule_weights",
            lambda: fermi_golden_rule_weights(
                reciprocal_vectors,
                response_occupied_bands,
                response_target_bands,
                sample_energies,
                weight_grid_shape=grid_shape,
                method="linear",
            ),
        ),
        (
            "complex_frequency_polarization_weights",
            lambda: complex_frequency_polarization_weights(
                reciprocal_vectors,
                response_occupied_bands,
                response_target_bands,
                complex_sample_energies,
                weight_grid_shape=grid_shape,
                method="linear",
            ),
        ),
        (
            "fermi_golden_rule_weights_prepared",
            lambda: prepared_response.fermi_golden_rule_weights(sample_energies),
        ),
        (
            "complex_frequency_polarization_weights_prepared",
            lambda: prepared_response.complex_frequency_polarization_weights(complex_sample_energies),
        ),
    ]

    print(f"Benchmark grid: {args.grid}^2")
    print("Triangle method: linear")
    print(f"Sample energies: {args.energy_count}")
    print(f"Lindhard q-value: {args.q_value:.6f}")
    print()

    for label, kernel in tasks:
        elapsed = time_best(kernel, repeats=args.repeats)
        print(f"{label:>7s}: {elapsed:.3f} s (best of {args.repeats} warm runs)")

    if args.profile:
        _emit_profiles(tasks, args.profile_task, args.profile_limit)


def _emit_profiles(tasks, profile_task: str, profile_limit: int) -> None:
    profile_targets = {profile_task}
    if profile_task == "all":
        profile_targets = {label for label, _ in tasks}

    print()
    for label, kernel in tasks:
        if label not in profile_targets:
            continue
        print(f"[profile] {label}")
        print(profile_top_cumulative(kernel, limit=profile_limit))
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the current bztetra.twod hot paths on free-electron workloads."
    )
    parser.add_argument("--grid", type=int, default=64, help="Square grid size to benchmark (default: 64)")
    parser.add_argument("--energy-count", type=int, default=16, help="Number of DOS/frequency sample energies (default: 16)")
    parser.add_argument("--q-value", type=float, default=2.0, help="Momentum transfer for the Lindhard benchmark (default: 2.0)")
    parser.add_argument("--repeats", type=int, default=3, help="How many warm runs to time per kernel (default: 3)")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print cProfile top-cumulative statistics for the selected public kernel(s)",
    )
    parser.add_argument(
        "--profile-task",
        default="complex_frequency_polarization_weights",
        help="Comma-separated benchmark labels to profile, or 'all' (default: complex_frequency_polarization_weights)",
    )
    parser.add_argument(
        "--profile-limit",
        type=int,
        default=20,
        help="How many cumulative cProfile rows to show per selected kernel (default: 20)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
