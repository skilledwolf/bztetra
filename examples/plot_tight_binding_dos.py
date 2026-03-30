from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bztetra import density_of_states_weights
from bztetra import solve_fermi_energy
from bztetra import integrated_density_of_states_weights

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised as a runtime dependency check
    raise SystemExit("matplotlib is required for this example. Install with `pip install -e '.[plot]'`.") from exc


ENERGY_MIN = -3.0
ENERGY_MAX = 3.0
ENERGY_COUNT = 100
COARSE_GRID = (8, 8, 8)
DEFAULT_OUTPUT = Path("build/review_plots/tight_binding_dos.png")
LEGACY_DATA_DIR = Path(__file__).resolve().parents[1] / "libtetra_original" / "example"


def main() -> None:
    args = _parse_args()

    reciprocal_vectors = np.eye(3, dtype=np.float64)
    sample_energies = np.linspace(ENERGY_MIN, ENERGY_MAX, ENERGY_COUNT, dtype=np.float64)

    coarse_eigenvalues = build_cubic_tight_binding_band(COARSE_GRID)
    linear_dos, linear_intdos = _compute_spectra(reciprocal_vectors, coarse_eigenvalues, sample_energies, "linear")
    optimized_dos, optimized_intdos = _compute_spectra(
        reciprocal_vectors,
        coarse_eigenvalues,
        sample_energies,
        "optimized",
    )
    fermi_solution = solve_fermi_energy(
        reciprocal_vectors,
        coarse_eigenvalues,
        electrons_per_spin=0.5,
        weight_grid_shape=COARSE_GRID,
        method="optimized",
    )

    legacy_converged = _load_legacy_dataset("dos40.dat")
    legacy_linear = _load_legacy_dataset("dos1_8.dat")
    legacy_optimized = _load_legacy_dataset("dos2_8.dat")
    legacy_converged_intdos = cumulative_trapezoid(legacy_converged[:, 0], legacy_converged[:, 1])
    legacy_converged_intdos /= legacy_converged_intdos[-1]

    figure = build_figure(
        sample_energies,
        linear_dos,
        optimized_dos,
        linear_intdos,
        optimized_intdos,
        legacy_converged,
        legacy_linear,
        legacy_optimized,
        legacy_converged_intdos,
        fermi_solution.fermi_energy,
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote plot to {output_path}")
    print(f"Optimized 8x8x8 half-filling Fermi energy: {fermi_solution.fermi_energy:.6f}")
    print(f"Max |optimized port - legacy optimized|: {np.max(np.abs(optimized_dos - legacy_optimized[:, 1])):.6e}")
    print(f"Max |linear port - legacy linear|: {np.max(np.abs(linear_dos - legacy_linear[:, 1])):.6e}")
    print(f"Reference integrated DOS endpoint after normalization: {legacy_converged_intdos[-1]:.6f}")


def build_cubic_tight_binding_band(grid_shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = grid_shape
    eigenvalues = np.empty((nx, ny, nz, 1), dtype=np.float64)

    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                kvec = 2.0 * np.pi * (
                    np.array((x_index, y_index, z_index), dtype=np.float64) - 0.5 * np.array(grid_shape, dtype=np.float64)
                ) / np.array(grid_shape, dtype=np.float64)
                eigenvalues[x_index, y_index, z_index, 0] = -np.cos(kvec).sum()

    return eigenvalues


def cumulative_trapezoid(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    increments = 0.5 * (y_values[1:] + y_values[:-1]) * np.diff(x_values)
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(increments)))


def build_figure(
    sample_energies: np.ndarray,
    linear_dos: np.ndarray,
    optimized_dos: np.ndarray,
    linear_intdos: np.ndarray,
    optimized_intdos: np.ndarray,
    legacy_converged: np.ndarray,
    legacy_linear: np.ndarray,
    legacy_optimized: np.ndarray,
    legacy_converged_intdos: np.ndarray,
    fermi_energy: float,
):
    figure, (axis_dos, axis_intdos) = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)

    axis_dos.plot(legacy_converged[:, 0], legacy_converged[:, 1], color="#111111", linewidth=2.8, label="Legacy converged (40^3)")
    axis_dos.plot(legacy_linear[:, 0], legacy_linear[:, 1], color="#8A6D1D", linewidth=1.6, linestyle="--", label="Legacy linear (8^3)")
    axis_dos.plot(
        legacy_optimized[:, 0],
        legacy_optimized[:, 1],
        color="#005F73",
        linewidth=1.6,
        linestyle="--",
        label="Legacy optimized (8^3)",
    )
    axis_dos.scatter(sample_energies, linear_dos, color="#E9C46A", s=18, marker="s", label="Port linear (8^3)")
    axis_dos.scatter(sample_energies, optimized_dos, color="#0A9396", s=18, marker="o", label="Port optimized (8^3)")

    axis_intdos.plot(
        legacy_converged[:, 0],
        legacy_converged_intdos,
        color="#111111",
        linewidth=2.8,
        label="Integrated legacy converged DOS",
    )
    axis_intdos.plot(sample_energies, linear_intdos, color="#E9C46A", linewidth=1.8, label="Port linear intDOS (8^3)")
    axis_intdos.plot(
        sample_energies,
        optimized_intdos,
        color="#0A9396",
        linewidth=1.8,
        label="Port optimized intDOS (8^3)",
    )

    for axis in (axis_dos, axis_intdos):
        axis.axvline(-3.0, color="#999999", linewidth=1.0, linestyle=":")
        axis.axvline(3.0, color="#999999", linewidth=1.0, linestyle=":")
        axis.axvline(0.0, color="#BB3E03", linewidth=1.0, linestyle=":")
        axis.grid(alpha=0.18)

    axis_intdos.axhline(0.5, color="#BB3E03", linewidth=1.0, linestyle=":")
    axis_intdos.axvline(fermi_energy, color="#AE2012", linewidth=1.4, linestyle="--")

    axis_dos.set_title("Cubic Tight-Binding DOS: coarse tetrahedra against the legacy converged line")
    axis_dos.set_ylabel("DOS [1/t]")
    axis_dos.legend(loc="upper left", ncol=2, fontsize=9)
    axis_dos.annotate(
        "band edges",
        xy=(3.0, legacy_converged[-1, 1]),
        xytext=(2.1, 0.14),
        arrowprops={"arrowstyle": "->", "color": "#666666"},
        color="#444444",
    )
    axis_dos.annotate(
        "half filling near E = 0",
        xy=(0.0, legacy_converged[len(legacy_converged) // 2, 1]),
        xytext=(-2.2, 0.22),
        arrowprops={"arrowstyle": "->", "color": "#AE2012"},
        color="#AE2012",
    )

    axis_intdos.set_xlabel("Energy [t]")
    axis_intdos.set_ylabel("Integrated DOS")
    axis_intdos.set_ylim(-0.02, 1.02)
    axis_intdos.legend(loc="upper left", fontsize=9)
    axis_intdos.text(
        0.02,
        0.06,
        f"optimized 8^3 EF = {fermi_energy:.4f}\nexact half-filling EF = 0",
        transform=axis_intdos.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )

    return figure


def _compute_spectra(
    reciprocal_vectors: np.ndarray,
    eigenvalues: np.ndarray,
    sample_energies: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    weight_grid_shape = tuple(int(item) for item in eigenvalues.shape[:3])
    dos_weights = density_of_states_weights(
        reciprocal_vectors,
        eigenvalues,
        sample_energies,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    intdos_weights = integrated_density_of_states_weights(
        reciprocal_vectors,
        eigenvalues,
        sample_energies,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return dos_weights.sum(axis=(1, 2, 3, 4)), intdos_weights.sum(axis=(1, 2, 3, 4))


def _load_legacy_dataset(filename: str) -> np.ndarray:
    return np.loadtxt(LEGACY_DATA_DIR / filename, dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the cubic tight-binding DOS/intDOS review figure inspired by the legacy libtetrabz example."
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
