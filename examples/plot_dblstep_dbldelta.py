from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tetrabz import dbldelta
from tetrabz import dblstep

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required for this example. Install with `pip install -e '.[plot]'`.") from exc


DEFAULT_OUTPUT = Path("build/review_plots/dblstep_dbldelta.png")
GRID_SHAPE = (32, 32, 32)
Q_DBLSTEP = np.linspace(0.0, 2.5, 26, dtype=np.float64)
Q_DBLDELTA = np.linspace(0.35, 2.5, 24, dtype=np.float64)
FERMI_ENERGY = 0.5
FERMI_WAVEVECTOR = 1.0


def main() -> None:
    args = _parse_args()

    dblstep_curve = compute_curve(dblstep, Q_DBLSTEP)
    dbldelta_curve = compute_curve(dbldelta, Q_DBLDELTA)

    dblstep_exact = exact_dblstep_curve(Q_DBLSTEP)
    dbldelta_exact = exact_dbldelta_curve(Q_DBLDELTA)

    dblstep_normalized = dblstep_curve / half_fermi_volume()
    dblstep_exact_normalized = dblstep_exact / half_fermi_volume()
    dbldelta_normalized = normalize_dbldelta_curve(Q_DBLDELTA, dbldelta_curve)
    dbldelta_exact_normalized = normalize_dbldelta_curve(Q_DBLDELTA, dbldelta_exact)

    figure = build_figure(
        dblstep_normalized,
        dblstep_exact_normalized,
        dbldelta_normalized,
        dbldelta_exact_normalized,
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    dblstep_error = np.max(np.abs(dblstep_normalized - dblstep_exact_normalized))
    dbldelta_mask = Q_DBLDELTA < 1.85
    dbldelta_error = np.max(np.abs(dbldelta_normalized[dbldelta_mask] - dbldelta_exact_normalized[dbldelta_mask]))
    print(f"Wrote plot to {output_path}")
    print(f"Max |normalized dblstep - exact|: {dblstep_error:.6e}")
    print(f"Max |normalized dbldelta - exact| for q < 1.85: {dbldelta_error:.6e}")


def compute_curve(kernel, q_values: np.ndarray) -> np.ndarray:
    values = np.empty(q_values.size, dtype=np.float64)

    for index, q_value in enumerate(q_values):
        reciprocal_vectors, occupied, target = build_shifted_free_electron_bands(GRID_SHAPE, float(q_value))
        weights = kernel(
            reciprocal_vectors,
            occupied,
            target,
            weight_grid_shape=GRID_SHAPE,
            method="optimized",
        )
        values[index] = weights[..., 0, 0].sum() * np.linalg.det(reciprocal_vectors)

    return values


def build_shifted_free_electron_bands(
    grid_shape: tuple[int, int, int],
    q_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
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

    return reciprocal_vectors, occupied, target


def exact_dblstep_curve(q_values: np.ndarray) -> np.ndarray:
    values = np.zeros_like(q_values)
    active = q_values < 2.0 * FERMI_WAVEVECTOR
    q_active = q_values[active]
    values[active] = np.pi * (4.0 + q_active) * (2.0 - q_active) ** 2 / 24.0
    return values


def exact_dbldelta_curve(q_values: np.ndarray) -> np.ndarray:
    values = np.zeros_like(q_values)
    active = (q_values > 0.0) & (q_values < 2.0 * FERMI_WAVEVECTOR)
    values[active] = 2.0 * np.pi / q_values[active]
    return values


def half_fermi_volume() -> float:
    return 2.0 * np.pi / 3.0


def normalize_dbldelta_curve(q_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(values)
    active = q_values > 0.0
    normalized[active] = q_values[active] * values[active] / (2.0 * np.pi)
    return normalized


def build_figure(
    dblstep_curve: np.ndarray,
    dblstep_exact: np.ndarray,
    dbldelta_curve: np.ndarray,
    dbldelta_exact: np.ndarray,
):
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharex=False)

    axes[0].plot(Q_DBLSTEP, dblstep_exact, color="#111111", linewidth=2.6, label="Exact overlap fraction")
    axes[0].scatter(Q_DBLSTEP, dblstep_curve, color="#0A9396", s=24, label="tetrabz (32^3)")
    axes[0].axvline(2.0, color="#AE2012", linewidth=1.2, linestyle=":")
    axes[0].grid(alpha=0.2)
    axes[0].set_title("dblstep")
    axes[0].set_xlabel(r"Momentum transfer $q / k_F$")
    axes[0].set_ylabel("Normalized overlap weight")
    axes[0].set_xlim(0.0, 2.5)
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].annotate(
        "half the overlap volume\nof two unit Fermi spheres",
        xy=(0.8, dblstep_exact[np.argmin(np.abs(Q_DBLSTEP - 0.8))]),
        xytext=(1.05, 0.77),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#555555",
    )

    axes[1].plot(Q_DBLDELTA, dbldelta_exact, color="#111111", linewidth=2.6, label="Exact normalized shell weight")
    axes[1].scatter(Q_DBLDELTA, dbldelta_curve, color="#CA6702", s=24, label="tetrabz (32^3)")
    axes[1].axvline(2.0, color="#AE2012", linewidth=1.2, linestyle=":")
    axes[1].grid(alpha=0.2)
    axes[1].set_title("dbldelta")
    axes[1].set_xlabel(r"Momentum transfer $q / k_F$")
    axes[1].set_ylabel(r"$q\,W(q)/(2\pi)$")
    axes[1].set_xlim(0.3, 2.5)
    axes[1].set_ylim(-0.05, 1.35)
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].annotate(
        "ideal shell-intersection weight\nstays near 1 below $2k_F$",
        xy=(1.0, 1.0),
        xytext=(1.2, 1.18),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#555555",
    )

    figure.suptitle("Free-Electron Phase-Space Response for dblstep and dbldelta")
    return figure


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
        description="Plot free-electron phase-space review curves for dblstep and dbldelta."
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
