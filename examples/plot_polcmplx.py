from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tetrabz import polcmplx

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required for this example. Install with `pip install -e '.[plot]'`.") from exc


DEFAULT_OUTPUT = Path("build/review_plots/polcmplx.png")
GRID_SHAPE = (16, 16, 16)
REAL_FREQUENCIES = np.linspace(-2.5, 1.5, 121, dtype=np.float64)
BROADENING = 0.5
FERMI_ENERGY = 0.5


def main() -> None:
    args = _parse_args()

    reciprocal_vectors, occupied, target = build_response_bands(GRID_SHAPE)
    metric = build_weight_metric(reciprocal_vectors, GRID_SHAPE)
    complex_energies = REAL_FREQUENCIES + 1j * BROADENING
    computed = compute_weighted_curve(reciprocal_vectors, occupied, target, metric, complex_energies)
    exact = exact_interband_curve(complex_energies)

    figure = build_figure(computed, exact)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote plot to {output_path}")
    print(f"Max |source 0 -> target 1 - exact|: {np.max(np.abs(computed[:, 1, 0] - exact[:, 1, 0])):.6e}")
    print(f"Max |source 1 -> target 1 - exact|: {np.max(np.abs(computed[:, 1, 1] - exact[:, 1, 1])):.6e}")


def compute_weighted_curve(
    reciprocal_vectors: np.ndarray,
    occupied: np.ndarray,
    target: np.ndarray,
    metric: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    weights = polcmplx(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        weight_grid_shape=GRID_SHAPE,
        method="optimized",
    )
    weighted = (weights * metric[None, ..., None, None]).sum(axis=(1, 2, 3))
    return weighted * np.linalg.det(reciprocal_vectors)


def exact_interband_curve(energies: np.ndarray) -> np.ndarray:
    values = np.zeros((energies.size, 2, 2), dtype=np.complex128)
    values[:, 1, 0] = (8.0 * np.pi) / (5.0 * (1.0 + 2.0 * energies))
    values[:, 1, 1] = (np.sqrt(8.0) * np.pi) / (5.0 * (1.0 + 4.0 * energies))
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


def build_figure(computed: np.ndarray, exact: np.ndarray):
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)

    channel_colors = {
        "0_to_1": "#0A9396",
        "1_to_1": "#AE2012",
    }

    axes[0].plot(REAL_FREQUENCIES, exact[:, 1, 0].real, color=channel_colors["0_to_1"], linewidth=2.4, label="Exact Re (0 -> 1)")
    axes[0].plot(REAL_FREQUENCIES, exact[:, 1, 1].real, color=channel_colors["1_to_1"], linewidth=2.4, label="Exact Re (1 -> 1)")
    axes[0].scatter(REAL_FREQUENCIES[::6], computed[::6, 1, 0].real, color="#005F73", s=20, label="Port Re (0 -> 1)")
    axes[0].scatter(REAL_FREQUENCIES[::6], computed[::6, 1, 1].real, color="#9B2226", s=20, marker="s", label="Port Re (1 -> 1)")
    axes[0].set_ylabel(r"$\Re\, \chi(\omega + i\eta)$")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(REAL_FREQUENCIES, -exact[:, 1, 0].imag, color=channel_colors["0_to_1"], linewidth=2.4, label="Exact -Im (0 -> 1)")
    axes[1].plot(REAL_FREQUENCIES, -exact[:, 1, 1].imag, color=channel_colors["1_to_1"], linewidth=2.4, label="Exact -Im (1 -> 1)")
    axes[1].scatter(REAL_FREQUENCIES[::6], -computed[::6, 1, 0].imag, color="#005F73", s=20, label="Port -Im (0 -> 1)")
    axes[1].scatter(REAL_FREQUENCIES[::6], -computed[::6, 1, 1].imag, color="#9B2226", s=20, marker="s", label="Port -Im (1 -> 1)")
    axes[1].set_ylabel(r"$-\Im\, \chi(\omega + i\eta)$")
    axes[1].set_xlabel(r"Real frequency $\omega$")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right", fontsize=9)

    figure.suptitle(r"Free-Electron Complex Response Along $\omega + i\eta$ With Exact Interband Channels")
    axes[0].annotate(
        "broadening keeps the poles finite",
        xy=(-0.5, exact[np.argmin(np.abs(REAL_FREQUENCIES + 0.5)), 1, 0].real),
        xytext=(-1.7, 0.15),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#555555",
    )
    axes[1].annotate(
        r"interband absorption peak near $\omega \approx -0.25$",
        xy=(-0.25, -exact[np.argmin(np.abs(REAL_FREQUENCIES + 0.25)), 1, 1].imag),
        xytext=(-1.95, 0.44),
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
        description="Plot the complex-frequency free-electron response review figure for the current polcmplx port."
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
