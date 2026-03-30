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


DEFAULT_OUTPUT = Path("build/review_plots/polcmplx_matsubara.png")
GRID_SHAPE = (16, 16, 16)
# Start above zero: the full tensor includes intraband channels with a pole at z = 0.
MATSUBARA_FREQUENCIES = np.linspace(0.1, 2.5, 81, dtype=np.float64)
FERMI_ENERGY = 0.5


def main() -> None:
    args = _parse_args()

    reciprocal_vectors, occupied, target = build_response_bands(GRID_SHAPE)
    metric = build_weight_metric(reciprocal_vectors, GRID_SHAPE)
    sample_energies = 1j * MATSUBARA_FREQUENCIES

    weighted = compute_weighted_curve(reciprocal_vectors, occupied, target, metric, sample_energies)
    exact_channels = exact_constant_gap_channels(sample_energies)

    figure = build_figure(weighted, exact_channels)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    channel_01_error = np.max(np.abs(weighted[:, 1, 0] - exact_channels[:, 0]))
    channel_11_error = np.max(np.abs(weighted[:, 1, 1] - exact_channels[:, 1]))
    print(f"Wrote plot to {output_path}")
    print(f"Max |0 -> 1 channel - exact|: {channel_01_error:.6e}")
    print(f"Max |1 -> 1 channel - exact|: {channel_11_error:.6e}")


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


def exact_constant_gap_channels(energies: np.ndarray) -> np.ndarray:
    samples = np.asarray(energies, dtype=np.complex128)
    values = np.empty((samples.size, 2), dtype=np.complex128)
    values[:, 0] = 8.0 * np.pi / (5.0 * (1.0 + 2.0 * samples))
    values[:, 1] = np.sqrt(8.0) * np.pi / (5.0 * (1.0 + 4.0 * samples))
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


def build_figure(weighted_curve: np.ndarray, exact_channels: np.ndarray):
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharex=True)

    channels = (
        ("0 -> 1", weighted_curve[:, 1, 0], exact_channels[:, 0], "#0A9396"),
        ("1 -> 1", weighted_curve[:, 1, 1], exact_channels[:, 1], "#AE2012"),
    )
    parts = (
        ("Real Part", np.real, axes[0]),
        ("Imaginary Part", np.imag, axes[1]),
    )

    for title, projector, axis in parts:
        for label, numerical, exact, color in channels:
            axis.plot(
                MATSUBARA_FREQUENCIES,
                projector(numerical),
                color=color,
                linewidth=2.6,
                label=f"Port {label}",
            )
            axis.plot(
                MATSUBARA_FREQUENCIES,
                projector(exact),
                color=color,
                linewidth=1.4,
                linestyle="--",
                label=f"Exact {label}",
            )

        axis.grid(alpha=0.2)
        axis.set_title(title)
        axis.set_xlabel(r"Matsubara frequency $\omega_n$")

    axes[0].set_ylabel(r"Integrated polarization weight")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].legend(loc="lower left", fontsize=9)

    figure.suptitle(r"Free-Electron Complex Polarization on the Imaginary-Frequency Axis")
    axes[0].annotate(
        "constant-gap interband channels\ndecay as $1 / (\\Delta + i\\omega_n)$",
        xy=(0.75, np.real(exact_channels[24, 0])),
        xytext=(1.15, 2.4),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#555555",
    )
    axes[1].annotate(
        "causal response stays smooth\non the Matsubara axis",
        xy=(1.0, np.imag(exact_channels[32, 1])),
        xytext=(1.45, -0.65),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#555555",
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
        description="Plot the free-electron complex polarization review figure on the Matsubara axis."
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
