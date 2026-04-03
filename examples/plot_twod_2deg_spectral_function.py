from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np

from bztetra.twod import prepare_response_sweep_evaluator

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required for this example. Install with `pip install -e '.[plot]'`.") from exc


DEFAULT_OUTPUT = Path("build/review_plots/twod_2deg_spectral_function.png")
GRID_SHAPE = (64, 64)
Q_VALUES = np.linspace(0.0, 2.5, 97, dtype=np.float64)
OMEGA_VALUES = np.linspace(0.0, 5.5, 501, dtype=np.float64)
LINECUT_Q_VALUES = np.array([0.5, 1.0, 2.0], dtype=np.float64)
FERMI_ENERGY = 0.5
FERMI_WAVEVECTOR = 1.0


def main() -> None:
    args = _parse_args()
    evaluation_start = time.perf_counter()
    spectral_map = compute_spectral_map(workers=args.workers)
    evaluation_elapsed = time.perf_counter() - evaluation_start
    figure = build_figure(spectral_map)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    peak_index = np.unravel_index(int(np.argmax(spectral_map)), spectral_map.shape)
    peak_omega = OMEGA_VALUES[peak_index[0]]
    peak_q = Q_VALUES[peak_index[1]]
    print(f"Wrote plot to {output_path}")
    print(f"Evaluated spectral map in {evaluation_elapsed:.3f} s")
    print(f"Used q-batch workers: {args.workers}")
    print(f"Peak spectral weight at q={peak_q:.3f}, omega={peak_omega:.3f}")


def compute_spectral_map(*, workers: int) -> np.ndarray:
    reciprocal_vectors, occupied, target_batch = build_shifted_free_electron_targets(Q_VALUES)
    sweep = prepare_response_sweep_evaluator(
        reciprocal_vectors,
        occupied,
        weight_grid_shape=GRID_SHAPE,
        method="linear",
    )
    contracted = sweep.fermi_golden_rule_observables_batch(
        target_batch,
        OMEGA_VALUES,
        workers=workers,
    )
    return contracted.T * abs(np.linalg.det(reciprocal_vectors))


def build_shifted_free_electron_targets(q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal_vectors = np.diag([3.0, 3.0]).astype(np.float64)
    occupied = np.empty((*GRID_SHAPE, 1), dtype=np.float64)
    target_batch = np.empty((q_values.size, *GRID_SHAPE, 1), dtype=np.float64)

    nx, ny = GRID_SHAPE
    for x_index in range(nx):
        for y_index in range(ny):
            kvec = reciprocal_vectors @ _centered_fractional_kpoint((x_index, y_index), GRID_SHAPE)
            kx = float(kvec[0])
            ky = float(kvec[1])
            occupied[x_index, y_index, 0] = 0.5 * float(np.dot(kvec, kvec)) - FERMI_ENERGY
            for q_index, q_value in enumerate(q_values):
                shifted_kx = kx + float(q_value)
                target_batch[q_index, x_index, y_index, 0] = 0.5 * (
                    shifted_kx * shifted_kx + ky * ky
                ) - FERMI_ENERGY

    return reciprocal_vectors, occupied, target_batch


def particle_hole_continuum_edges(q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q_values, dtype=np.float64)
    lower = np.maximum(0.0, 0.5 * q * q - q * FERMI_WAVEVECTOR)
    upper = 0.5 * q * q + q * FERMI_WAVEVECTOR
    return lower, upper


def build_figure(spectral_map: np.ndarray):
    figure, (map_axis, line_axis) = plt.subplots(
        1,
        2,
        figsize=(11.6, 5.8),
        gridspec_kw={"width_ratios": (1.6, 1.0)},
    )

    normalized_map = spectral_map / max(float(np.max(spectral_map)), 1.0e-16)
    lower_edge, upper_edge = particle_hole_continuum_edges(Q_VALUES)

    image = map_axis.imshow(
        normalized_map,
        origin="lower",
        aspect="auto",
        extent=(Q_VALUES[0], Q_VALUES[-1], OMEGA_VALUES[0], OMEGA_VALUES[-1]),
        cmap="magma",
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0),
        interpolation="nearest",
    )
    map_axis.plot(Q_VALUES, lower_edge, color="#E9D8A6", linewidth=1.3, linestyle="--", label=r"$\omega_-(q)$")
    map_axis.plot(Q_VALUES, upper_edge, color="#94D2BD", linewidth=1.3, linestyle="--", label=r"$\omega_+(q)$")
    map_axis.set_title("2D Free-Electron Spectral Weight Map")
    map_axis.set_xlabel(r"Momentum transfer $q_x / k_F$")
    map_axis.set_ylabel(r"Energy transfer $\omega$")
    map_axis.set_xlim(Q_VALUES[0], Q_VALUES[-1])
    map_axis.set_ylim(OMEGA_VALUES[0], OMEGA_VALUES[-1])
    map_axis.legend(loc="upper left", fontsize=8)
    map_axis.annotate(
        "particle-hole continuum",
        xy=(1.7, 2.5),
        xytext=(0.95, 4.2),
        arrowprops={"arrowstyle": "->", "color": "#F4A261"},
        color="#F4A261",
    )

    colorbar = figure.colorbar(image, ax=map_axis, pad=0.02)
    colorbar.set_label("Normalized spectral weight")

    peak_scale = max(float(np.max(spectral_map)), 1.0e-16)
    for q_value in LINECUT_Q_VALUES:
        q_index = int(np.argmin(np.abs(Q_VALUES - q_value)))
        line_axis.plot(
            OMEGA_VALUES,
            spectral_map[:, q_index] / peak_scale,
            linewidth=2.1,
            label=rf"$q_x / k_F = {Q_VALUES[q_index]:.2f}$",
        )

    line_axis.set_title("Selected Line Cuts")
    line_axis.set_xlabel(r"Energy transfer $\omega$")
    line_axis.set_ylabel("Spectral weight / max")
    line_axis.set_xlim(OMEGA_VALUES[0], OMEGA_VALUES[-1])
    line_axis.set_ylim(-0.01, 1.05)
    line_axis.grid(alpha=0.22)
    line_axis.legend(loc="upper right", fontsize=8)

    return figure


def _centered_fractional_kpoint(
    indices: tuple[int, int],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.asarray(grid_shape, dtype=np.int64)
    half_grid = grid // 2
    integer_indices = np.asarray(indices, dtype=np.int64)
    centered = np.mod(integer_indices + half_grid, grid) - half_grid
    return centered.astype(np.float64) / grid.astype(np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the 2D free-electron spectral-weight map S(q_x, omega) using the contracted triangle-response path."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the plot image (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of q-points to evaluate in parallel "
            f"(default: 1, detected cpu count: {os.cpu_count() or 1})"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
