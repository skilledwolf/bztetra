from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bztetra.twod import phase_space_overlap_weights

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required for this example. Install with `pip install -e '.[plot]'`.") from exc


DEFAULT_OUTPUT = Path("build/review_plots/twod_phase_space_overlap.png")
GRID_SHAPE = (128, 128)
Q_VALUES = np.linspace(0.0, 2.4, 25, dtype=np.float64)
FERMI_ENERGY = 0.5
FERMI_WAVEVECTOR = 1.0


def main() -> None:
    args = _parse_args()
    overlap_curve = compute_overlap_curve()
    exact_curve = exact_overlap_curve(Q_VALUES)

    normalized = overlap_curve / half_fermi_area()
    exact_normalized = exact_curve / half_fermi_area()

    figure = build_figure(normalized, exact_normalized)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    max_error = np.max(np.abs(normalized - exact_normalized))
    print(f"Wrote plot to {output_path}")
    print(f"Max |normalized overlap - exact|: {max_error:.6e}")


def compute_overlap_curve() -> np.ndarray:
    values = np.empty(Q_VALUES.size, dtype=np.float64)
    for index, q_value in enumerate(Q_VALUES):
        reciprocal_vectors, occupied, target = build_shifted_free_electron_bands(float(q_value))
        weights = phase_space_overlap_weights(
            reciprocal_vectors,
            occupied,
            target,
            weight_grid_shape=GRID_SHAPE,
            method="linear",
        )
        values[index] = weights[..., 0, 0].sum() * np.linalg.det(reciprocal_vectors)
    return values


def build_shifted_free_electron_bands(q_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal_vectors = np.diag([3.0, 3.0]).astype(np.float64)
    occupied = np.empty((*GRID_SHAPE, 1), dtype=np.float64)
    target = np.empty((*GRID_SHAPE, 1), dtype=np.float64)
    qvec = np.array([q_value, 0.0], dtype=np.float64)

    nx, ny = GRID_SHAPE
    for x_index in range(nx):
        for y_index in range(ny):
            kvec = reciprocal_vectors @ _centered_fractional_kpoint((x_index, y_index), GRID_SHAPE)
            occupied[x_index, y_index, 0] = 0.5 * float(np.dot(kvec, kvec)) - FERMI_ENERGY
            shifted = kvec + qvec
            target[x_index, y_index, 0] = 0.5 * float(np.dot(shifted, shifted)) - FERMI_ENERGY

    return reciprocal_vectors, occupied, target


def exact_overlap_curve(q_values: np.ndarray) -> np.ndarray:
    values = np.zeros_like(q_values)
    active = q_values < 2.0 * FERMI_WAVEVECTOR
    reduced = q_values[active] / (2.0 * FERMI_WAVEVECTOR)
    values[active] = np.arccos(reduced) - reduced * np.sqrt(1.0 - reduced * reduced)
    return values


def half_fermi_area() -> float:
    return 0.5 * np.pi * FERMI_WAVEVECTOR * FERMI_WAVEVECTOR


def build_figure(overlap_curve: np.ndarray, exact_curve: np.ndarray):
    figure, axis = plt.subplots(figsize=(9.0, 5.8))

    axis.plot(Q_VALUES, exact_curve, color="#111111", linewidth=2.8, label="Exact circular-segment area")
    axis.scatter(Q_VALUES, overlap_curve, color="#0A9396", s=24, label="bztetra.twod (128^2)")
    axis.axvline(2.0, color="#AE2012", linewidth=1.2, linestyle=":")
    axis.grid(alpha=0.2)

    axis.set_title("2D Free-Electron Phase-Space Overlap")
    axis.set_xlabel(r"Momentum transfer $q / k_F$")
    axis.set_ylabel("Overlap / half Fermi-disk area")
    axis.set_xlim(0.0, 2.4)
    axis.set_ylim(-0.02, 1.05)
    axis.legend(loc="upper right", fontsize=9)
    axis.annotate(
        "same-spectrum limit\nreturns half the occupied disk",
        xy=(0.0, 1.0),
        xytext=(0.45, 0.82),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#555555",
    )
    axis.annotate(
        "phase space closes at $2k_F$",
        xy=(2.0, 0.0),
        xytext=(1.45, 0.22),
        arrowprops={"arrowstyle": "->", "color": "#AE2012"},
        color="#AE2012",
    )

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
        description="Plot the 2D free-electron dblstep phase-space overlap review curve."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the plot image (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
