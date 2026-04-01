from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from bztetra.twod import prepare_response_evaluator

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.colors import TwoSlopeNorm
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required for this example. Install with `pip install -e '.[plot]'`.") from exc


DEFAULT_OUTPUT = Path("build/review_plots/twod_2deg_retarded_response.png")
GRID_SHAPE = (64, 64)
Q_VALUES = np.linspace(0.0, 2.5, 65, dtype=np.float64)
OMEGA_VALUES = np.linspace(0.0, 8.0, 241, dtype=np.float64)
LINECUT_Q_VALUES = np.array([0.5, 1.0, 2.0], dtype=np.float64)
VALIDATION_Q = 1.0
FERMI_ENERGY = 0.5
FERMI_WAVEVECTOR = 1.0


def main() -> None:
    args = _parse_args()
    evaluation_start = time.perf_counter()
    spectral_map, real_map, validation = compute_response_maps()
    evaluation_elapsed = time.perf_counter() - evaluation_start
    figure = build_figure(spectral_map, real_map, validation)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    spectral_peak_index = np.unravel_index(int(np.argmax(spectral_map)), spectral_map.shape)
    peak_omega = OMEGA_VALUES[spectral_peak_index[0]]
    peak_q = Q_VALUES[spectral_peak_index[1]]
    validation_error = np.max(np.abs(validation["reconstructed_real"] - validation["direct_real"]))
    print(f"Wrote plot to {output_path}")
    print(f"Evaluated retarded-response map in {evaluation_elapsed:.3f} s")
    print(f"Peak spectral weight at q={peak_q:.3f}, omega={peak_omega:.3f}")
    print(
        "Validation at "
        f"q={validation['q_value']:.3f}: "
        f"max |Re chi_KK - Re chi_direct(-omega)| = {validation_error:.3e}"
    )


def compute_response_maps() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    spectral_map = np.empty((OMEGA_VALUES.size, Q_VALUES.size), dtype=np.float64)
    real_map = np.empty((OMEGA_VALUES.size, Q_VALUES.size), dtype=np.float64)
    validation_q_index = int(np.argmin(np.abs(Q_VALUES - VALIDATION_Q)))
    validation_payload: dict[str, np.ndarray | float] = {}

    for q_index, q_value in enumerate(Q_VALUES):
        reciprocal_vectors, occupied, target = build_shifted_free_electron_bands(float(q_value))
        evaluator = prepare_response_evaluator(
            reciprocal_vectors,
            occupied,
            target,
            weight_grid_shape=GRID_SHAPE,
            method="linear",
        )
        response = evaluator.retarded_response_observables(OMEGA_VALUES)
        prefactor = abs(np.linalg.det(reciprocal_vectors))
        spectral_map[:, q_index] = response.imag * prefactor / np.pi
        real_map[:, q_index] = response.real * prefactor

        if q_index == validation_q_index:
            direct_real = (
                evaluator.complex_frequency_polarization_observables((-OMEGA_VALUES).astype(np.complex128)).real
                * prefactor
            )
            validation_payload = {
                "q_value": float(q_value),
                "reconstructed_real": real_map[:, q_index].copy(),
                "direct_real": direct_real,
            }

    return spectral_map, real_map, validation_payload


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


def particle_hole_continuum_edges(q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q_values, dtype=np.float64)
    lower = np.maximum(0.0, 0.5 * q * q - q * FERMI_WAVEVECTOR)
    upper = 0.5 * q * q + q * FERMI_WAVEVECTOR
    return lower, upper


def build_figure(
    spectral_map: np.ndarray,
    real_map: np.ndarray,
    validation: dict[str, np.ndarray | float],
):
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
    spectral_map_axis, real_map_axis = axes[0]
    spectral_line_axis, validation_axis = axes[1]

    lower_edge, upper_edge = particle_hole_continuum_edges(Q_VALUES)

    normalized_spectral = spectral_map / max(float(np.max(spectral_map)), 1.0e-16)
    spectral_image = spectral_map_axis.imshow(
        normalized_spectral,
        origin="lower",
        aspect="auto",
        extent=(Q_VALUES[0], Q_VALUES[-1], OMEGA_VALUES[0], OMEGA_VALUES[-1]),
        cmap="magma",
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0),
        interpolation="nearest",
    )
    spectral_map_axis.plot(Q_VALUES, lower_edge, color="#E9D8A6", linewidth=1.2, linestyle="--")
    spectral_map_axis.plot(Q_VALUES, upper_edge, color="#94D2BD", linewidth=1.2, linestyle="--")
    spectral_map_axis.set_title(r"2D 2DEG Spectral Weight $S(q_x,\omega)$")
    spectral_map_axis.set_xlabel(r"Momentum transfer $q_x / k_F$")
    spectral_map_axis.set_ylabel(r"Energy transfer $\omega$")
    figure.colorbar(spectral_image, ax=spectral_map_axis, pad=0.02, label="Normalized spectral weight")

    real_limit = max(float(np.max(np.abs(real_map))), 1.0e-16)
    real_image = real_map_axis.imshow(
        real_map,
        origin="lower",
        aspect="auto",
        extent=(Q_VALUES[0], Q_VALUES[-1], OMEGA_VALUES[0], OMEGA_VALUES[-1]),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-real_limit, vcenter=0.0, vmax=real_limit),
        interpolation="nearest",
    )
    real_map_axis.plot(Q_VALUES, lower_edge, color="#1D3557", linewidth=1.1, linestyle=":")
    real_map_axis.plot(Q_VALUES, upper_edge, color="#1D3557", linewidth=1.1, linestyle=":")
    real_map_axis.set_title(r"Reconstructed $\mathrm{Re}\,\Pi(-\omega + i0^+)$")
    real_map_axis.set_xlabel(r"Momentum transfer $q_x / k_F$")
    real_map_axis.set_ylabel(r"Energy transfer $\omega$")
    figure.colorbar(real_image, ax=real_map_axis, pad=0.02, label=r"$\mathrm{Re}\,\Pi$")

    line_scale = max(float(np.max(spectral_map)), 1.0e-16)
    for q_value in LINECUT_Q_VALUES:
        q_index = int(np.argmin(np.abs(Q_VALUES - q_value)))
        spectral_line_axis.plot(
            OMEGA_VALUES,
            spectral_map[:, q_index] / line_scale,
            linewidth=2.0,
            label=rf"$q_x / k_F = {Q_VALUES[q_index]:.2f}$",
        )

    spectral_line_axis.set_title("Selected Spectral Line Cuts")
    spectral_line_axis.set_xlabel(r"Energy transfer $\omega$")
    spectral_line_axis.set_ylabel(r"$S(q_x,\omega)$ / max")
    spectral_line_axis.set_xlim(OMEGA_VALUES[0], OMEGA_VALUES[-1])
    spectral_line_axis.set_ylim(-0.01, 1.05)
    spectral_line_axis.grid(alpha=0.22)
    spectral_line_axis.legend(loc="upper right", fontsize=8)

    validation_axis.plot(
        OMEGA_VALUES,
        validation["reconstructed_real"],
        color="#005F73",
        linewidth=2.2,
        label="KK reconstruction",
    )
    validation_axis.plot(
        OMEGA_VALUES,
        validation["direct_real"],
        color="#AE2012",
        linewidth=1.6,
        linestyle="--",
        label=r"direct $\Pi(-\omega + 0j)$",
    )
    validation_axis.set_title(
        rf"Real-Part Validation at $q_x / k_F = {float(validation['q_value']):.2f}$"
    )
    validation_axis.set_xlabel(r"Energy transfer $\omega$")
    validation_axis.set_ylabel(r"$\mathrm{Re}\,\Pi$")
    validation_axis.set_xlim(OMEGA_VALUES[0], OMEGA_VALUES[-1])
    validation_axis.grid(alpha=0.22)
    validation_axis.legend(loc="upper right", fontsize=8)

    figure.suptitle("2D Free-Electron Dynamic Response From Contracted Triangle Kernels", fontsize=14)
    figure.tight_layout()
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
        description=(
            "Plot the 2D free-electron dynamic response using the contracted triangle "
            "spectral kernel and automatic real-part reconstruction."
        )
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
