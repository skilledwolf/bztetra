from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tetrabz import fermigr

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required for this example. Install with `pip install -e '.[plot]'`.") from exc


DEFAULT_OUTPUT = Path("build/review_plots/fermigr.png")
GRID_SHAPE = (16, 16, 16)
ENERGY_SWEEP = np.linspace(0.1, 1.1, 41, dtype=np.float64)
EXACT_ENERGIES = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float64)
FERMI_ENERGY = 0.5


def main() -> None:
    args = _parse_args()

    reciprocal_vectors, occupied, target = build_response_bands(GRID_SHAPE)
    metric = build_weight_metric(reciprocal_vectors, GRID_SHAPE)
    dense_curve = compute_weighted_curve(reciprocal_vectors, occupied, target, metric, ENERGY_SWEEP)
    exact_points = exact_fermigr_points()

    figure = build_figure(dense_curve, exact_points)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    exact_curve = compute_weighted_curve(reciprocal_vectors, occupied, target, metric, EXACT_ENERGIES)
    print(f"Wrote plot to {output_path}")
    print(f"Max |channel 0->0 - exact points|: {np.max(np.abs(exact_curve[:, 0, 0] - exact_points[:, 0, 0])):.6e}")
    print(f"Max |channel 1->0 - exact points|: {np.max(np.abs(exact_curve[:, 0, 1] - exact_points[:, 0, 1])):.6e}")


def compute_weighted_curve(
    reciprocal_vectors: np.ndarray,
    occupied: np.ndarray,
    target: np.ndarray,
    metric: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    weights = fermigr(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        weight_grid_shape=GRID_SHAPE,
        method="optimized",
    )
    weighted = (weights * metric[None, ..., None, None]).sum(axis=(1, 2, 3))
    return weighted * np.linalg.det(reciprocal_vectors)


def exact_fermigr_points() -> np.ndarray:
    values = np.zeros((EXACT_ENERGIES.size, 2, 2), dtype=np.float64)
    values[:, 0, 0] = np.array(
        [
            4.0 * np.pi / 9.0,
            1295.0 * np.pi / 2592.0,
            15.0 * np.pi / 32.0,
        ],
        dtype=np.float64,
    )
    values[:, 0, 1] = np.array(
        [
            5183.0 * np.pi / 41472.0,
            4559.0 * np.pi / 41472.0,
            0.0,
        ],
        dtype=np.float64,
    )
    return values


def build_response_bands(grid_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
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

    return reciprocal_vectors, occupied, target


def build_weight_metric(reciprocal_vectors: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
    metric = np.empty(grid_shape, dtype=np.float64)
    nx, ny, nz = grid_shape

    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = reciprocal_vectors @ _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                metric[x_index, y_index, z_index] = float(np.dot(kvec, kvec))

    return metric


def build_figure(weighted_curve: np.ndarray, exact_points: np.ndarray):
    figure, axis = plt.subplots(figsize=(10.0, 6.5))

    axis.plot(ENERGY_SWEEP, weighted_curve[:, 0, 0], color="#0A9396", linewidth=2.6, label="Port 0 -> 0 (16^3)")
    axis.plot(ENERGY_SWEEP, weighted_curve[:, 0, 1], color="#AE2012", linewidth=2.6, label="Port 1 -> 0 (16^3)")
    axis.scatter(EXACT_ENERGIES, exact_points[:, 0, 0], color="#005F73", s=34, marker="o", label="Exact points 0 -> 0")
    axis.scatter(EXACT_ENERGIES, exact_points[:, 0, 1], color="#9B2226", s=34, marker="s", label="Exact points 1 -> 0")

    axis.axvline(1.0, color="#777777", linewidth=1.0, linestyle=":")
    axis.grid(alpha=0.2)
    axis.set_title(r"Free-Electron Fermi-Golden-Rule Weights With $k^2$ Matrix Element")
    axis.set_xlabel(r"Energy transfer $\omega$")
    axis.set_ylabel(r"Integrated weight")
    axis.legend(loc="upper right", fontsize=9)
    axis.annotate(
        "upper threshold in the toy model",
        xy=(1.0, exact_points[-1, 0, 0]),
        xytext=(0.72, 1.26),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#555555",
    )
    axis.annotate(
        "only the target band 0 channels stay active",
        xy=(0.62, weighted_curve[20, 0, 1]),
        xytext=(0.22, 0.56),
        arrowprops={"arrowstyle": "->", "color": "#9B2226"},
        color="#9B2226",
    )

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
        description="Plot the free-electron Fermi-golden-rule review figure for the current fermigr port."
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
