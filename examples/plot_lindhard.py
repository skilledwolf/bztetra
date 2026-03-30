from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tetrabz import density_of_states_weights
from tetrabz import static_polarization_weights

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required for this example. Install with `pip install -e '.[plot]'`.") from exc


DEFAULT_OUTPUT = Path("build/review_plots/lindhard.png")
LEGACY_DATA_DIR = Path(__file__).resolve().parents[1] / "libtetra_original" / "example"
FERMI_ENERGY = 0.5
GRID_SHAPE = (8, 8, 8)
Q_VALUES = np.linspace(0.0, 4.0, 31, dtype=np.float64)


def main() -> None:
    args = _parse_args()

    linear_curve = compute_lindhard_curve(GRID_SHAPE, "linear")
    optimized_curve = compute_lindhard_curve(GRID_SHAPE, "optimized")
    legacy_linear = _load_legacy_dataset("lindhard1_8.dat")
    legacy_optimized = _load_legacy_dataset("lindhard2_8.dat")
    dense_q = np.linspace(0.0, 4.0, 400, dtype=np.float64)
    exact_curve = exact_lindhard_curve(dense_q)

    figure = build_figure(
        dense_q,
        exact_curve,
        linear_curve,
        optimized_curve,
        legacy_linear,
        legacy_optimized,
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote plot to {output_path}")
    print(f"Max |current linear - legacy linear|: {np.max(np.abs(linear_curve - legacy_linear[:, 1])):.6e}")
    print(f"Max |current optimized - legacy optimized|: {np.max(np.abs(optimized_curve - legacy_optimized[:, 1])):.6e}")
    print(f"Current optimized q=0 value: {optimized_curve[0]:.6f}")
    print(f"Current optimized q=2 value: {optimized_curve[np.argmin(np.abs(Q_VALUES - 2.0))]:.6f}")


def compute_lindhard_curve(grid_shape: tuple[int, int, int], method: str) -> np.ndarray:
    values = np.empty(Q_VALUES.size, dtype=np.float64)

    for index, q_value in enumerate(Q_VALUES):
        reciprocal_vectors, eig1, eig2 = build_lindhard_bands(grid_shape, q_value)
        if index == 0:
            weights = density_of_states_weights(
                reciprocal_vectors,
                eig1,
                np.array([0.0], dtype=np.float64),
                weight_grid_shape=grid_shape,
                method=method,
            )
            values[index] = weights.sum() * np.linalg.det(reciprocal_vectors) / (4.0 * np.pi)
        else:
            weights = static_polarization_weights(
                reciprocal_vectors,
                eig1,
                eig2,
                weight_grid_shape=grid_shape,
                method=method,
            )
            values[index] = 2.0 * weights.sum() * np.linalg.det(reciprocal_vectors) / (4.0 * np.pi)

    return values


def build_lindhard_bands(grid_shape: tuple[int, int, int], q_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    eig1 = np.empty((*grid_shape, 1), dtype=np.float64)
    eig2 = np.empty((*grid_shape, 1), dtype=np.float64)
    qvec = np.array([q_value, 0.0, 0.0], dtype=np.float64)

    nx, ny, nz = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = reciprocal_vectors @ _centered_fractional_kpoint((x_index, y_index, z_index), grid_shape)
                eig1[x_index, y_index, z_index, 0] = 0.5 * float(np.dot(kvec, kvec)) - FERMI_ENERGY
                shifted = kvec + qvec
                eig2[x_index, y_index, z_index, 0] = 0.5 * float(np.dot(shifted, shifted)) - FERMI_ENERGY

    return reciprocal_vectors, eig1, eig2


def exact_lindhard_curve(q_values: np.ndarray) -> np.ndarray:
    values = np.empty_like(q_values)
    q_zero = np.isclose(q_values, 0.0)
    q_kohn = np.isclose(q_values, 2.0)
    regular = ~(q_zero | q_kohn)

    values[q_zero] = 1.0
    values[q_kohn] = 0.5

    q_nonzero = q_values[regular]
    values[regular] = 0.5 + 0.5 / q_nonzero * (1.0 - 0.25 * q_nonzero * q_nonzero) * np.log(
        np.abs((q_nonzero + 2.0) / (q_nonzero - 2.0))
    )
    return values


def build_figure(
    dense_q: np.ndarray,
    exact_curve: np.ndarray,
    linear_curve: np.ndarray,
    optimized_curve: np.ndarray,
    legacy_linear: np.ndarray,
    legacy_optimized: np.ndarray,
):
    figure, axis = plt.subplots(figsize=(10.0, 6.5))

    axis.plot(dense_q, exact_curve, color="#111111", linewidth=2.8, label="Exact Lindhard")
    axis.plot(
        legacy_linear[:, 0],
        legacy_linear[:, 1],
        color="#8A6D1D",
        linewidth=1.4,
        linestyle="--",
        label="Legacy linear (8^3)",
    )
    axis.plot(
        legacy_optimized[:, 0],
        legacy_optimized[:, 1],
        color="#005F73",
        linewidth=1.4,
        linestyle="--",
        label="Legacy optimized (8^3)",
    )
    axis.scatter(Q_VALUES, linear_curve, color="#E9C46A", s=22, marker="s", label="Port linear (8^3)")
    axis.scatter(Q_VALUES, optimized_curve, color="#0A9396", s=22, marker="o", label="Port optimized (8^3)")

    axis.axvline(2.0, color="#AE2012", linewidth=1.2, linestyle=":")
    axis.axhline(1.0, color="#777777", linewidth=1.0, linestyle=":")
    axis.grid(alpha=0.2)

    axis.set_title("3D Free-Electron Static Lindhard Function")
    axis.set_xlabel(r"$q/k_F$")
    axis.set_ylabel(r"$\chi_0(q) / N(E_F)$")
    axis.set_xlim(0.0, 4.0)
    axis.set_ylim(0.05, 1.05)
    axis.legend(loc="upper right", fontsize=9)
    axis.annotate(
        "Kohn anomaly at 2$k_F$",
        xy=(2.0, exact_lindhard_curve(np.array([2.0]))[0]),
        xytext=(2.35, 0.72),
        arrowprops={"arrowstyle": "->", "color": "#AE2012"},
        color="#AE2012",
    )
    axis.annotate(
        "compressibility limit",
        xy=(0.0, 1.0),
        xytext=(0.4, 0.97),
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


def _load_legacy_dataset(filename: str) -> np.ndarray:
    return np.loadtxt(LEGACY_DATA_DIR / filename, dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the static Lindhard response review figure inspired by the legacy libtetrabz example."
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
