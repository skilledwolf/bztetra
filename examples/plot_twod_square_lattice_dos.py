from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bztetra.twod import density_of_states_weights
from bztetra.twod import integrated_density_of_states_weights


def square_lattice_band(grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    reciprocal_vectors = 2.0 * np.pi * np.eye(2, dtype=np.float64)
    nx, ny = grid_shape
    eigenvalues = np.empty((nx, ny, 1), dtype=np.float64)

    for x_index in range(nx):
        for y_index in range(ny):
            kfrac = np.array([x_index / nx, y_index / ny], dtype=np.float64) - 0.5
            kcart = reciprocal_vectors @ kfrac
            eigenvalues[x_index, y_index, 0] = -2.0 * (np.cos(kcart[0]) + np.cos(kcart[1]))

    return reciprocal_vectors, eigenvalues


def main() -> None:
    reciprocal_vectors, eigenvalues = square_lattice_band((128, 128))
    energies = np.linspace(-4.0, 4.0, 401, dtype=np.float64)

    dos_curve = density_of_states_weights(reciprocal_vectors, eigenvalues, energies).sum(axis=(1, 2, 3))
    intdos_curve = integrated_density_of_states_weights(
        reciprocal_vectors,
        eigenvalues,
        energies,
    ).sum(axis=(1, 2, 3))

    output_dir = Path("build/review_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "twod_square_lattice_dos.png"

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), constrained_layout=True)

    axes[0].plot(energies, dos_curve, color="#005F73", lw=2.2)
    axes[0].axvline(0.0, color="#BB3E03", lw=1.2, ls="--", label="van Hove energy")
    axes[0].set_ylabel("DOS / BZ area")
    axes[0].set_title("2D Square-Lattice DOS from bztetra.twod")
    axes[0].legend(frameon=False)

    axes[1].plot(energies, intdos_curve, color="#0A9396", lw=2.2)
    axes[1].set_xlabel("Energy")
    axes[1].set_ylabel("Integrated DOS / BZ area")

    figure.savefig(output_path, dpi=200)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
