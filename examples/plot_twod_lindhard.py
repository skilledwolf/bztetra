from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bztetra.twod import density_of_states_weights
from bztetra.twod import static_polarization_weights

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required for this example. Install with `pip install -e '.[plot]'`.") from exc


DEFAULT_OUTPUT = Path("build/review_plots/twod_lindhard.png")
GRID_SHAPE = (128, 128)
Q_VALUES = np.linspace(0.0, 4.0, 33, dtype=np.float64)
FERMI_ENERGY = 0.5


def main() -> None:
    args = _parse_args()
    curve = compute_lindhard_curve()
    exact_curve = exact_lindhard_curve(Q_VALUES)

    figure = build_figure(curve, exact_curve)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    max_error = np.max(np.abs(curve - exact_curve))
    print(f"Wrote plot to {output_path}")
    print(f"Max |normalized Lindhard - exact|: {max_error:.6e}")


def compute_lindhard_curve() -> np.ndarray:
    reciprocal_vectors, base_band, _ = build_shifted_free_electron_bands(0.0)
    density_of_states = density_of_states_weights(
        reciprocal_vectors,
        base_band,
        np.array([0.0], dtype=np.float64),
        weight_grid_shape=GRID_SHAPE,
        method="linear",
    )
    density_of_states_at_fermi = density_of_states[..., 0].sum() * np.linalg.det(reciprocal_vectors)

    values = np.empty(Q_VALUES.size, dtype=np.float64)
    values[0] = 1.0

    for index in range(1, Q_VALUES.size):
        reciprocal_vectors, occupied, target = build_shifted_free_electron_bands(float(Q_VALUES[index]))
        weights = static_polarization_weights(
            reciprocal_vectors,
            occupied,
            target,
            weight_grid_shape=GRID_SHAPE,
            method="linear",
        )
        # The kernel is the occupied-to-empty contribution; the full symmetric
        # free-electron Lindhard curve doubles it for q != 0.
        values[index] = (
            2.0
            * weights[..., 0, 0].sum()
            * np.linalg.det(reciprocal_vectors)
            / density_of_states_at_fermi
        )

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


def exact_lindhard_curve(q_values: np.ndarray) -> np.ndarray:
    values = np.ones_like(q_values)
    active = q_values > 2.0
    values[active] = 1.0 - np.sqrt(1.0 - (2.0 / q_values[active]) ** 2)
    return values


def build_figure(curve: np.ndarray, exact_curve: np.ndarray):
    figure, axis = plt.subplots(figsize=(9.0, 5.8))

    axis.plot(Q_VALUES, exact_curve, color="#111111", linewidth=2.8, label="Exact 2D Lindhard")
    axis.scatter(Q_VALUES, curve, color="#CA6702", s=24, label="bztetra.twod (128^2)")
    axis.axvline(2.0, color="#AE2012", linewidth=1.2, linestyle=":")
    axis.axhline(1.0, color="#777777", linewidth=1.0, linestyle=":")
    axis.grid(alpha=0.2)

    axis.set_title("2D Free-Electron Static Lindhard Function")
    axis.set_xlabel(r"Momentum transfer $q / k_F$")
    axis.set_ylabel(r"$\chi_0(q) / N(E_F)$")
    axis.set_xlim(0.0, 4.0)
    axis.set_ylim(-0.02, 1.05)
    axis.legend(loc="upper right", fontsize=9)
    axis.annotate(
        "constant compressibility plateau",
        xy=(1.0, 1.0),
        xytext=(0.75, 0.82),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#555555",
    )
    axis.annotate(
        "2D Kohn cusp at $2k_F$",
        xy=(2.0, 1.0),
        xytext=(2.35, 0.72),
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
        description="Plot the normalized 2D free-electron Lindhard review curve."
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
