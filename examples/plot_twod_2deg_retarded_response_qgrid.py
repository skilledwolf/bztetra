from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np

from bztetra.twod import prepare_response_sweep_evaluator

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.colors import TwoSlopeNorm
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required for this example. Install with `pip install -e '.[plot]'`.") from exc


DEFAULT_OUTPUT = Path("build/review_plots/twod_2deg_retarded_response_qgrid.png")
DEFAULT_K_GRID_SHAPE = (64, 64)
DEFAULT_NQX = 65
DEFAULT_NQY = 65
DEFAULT_NOMEGA = 501
DEFAULT_Q_MAX = 2.5
DEFAULT_OMEGA_MAX = 8.0
DEFAULT_OMEGA_SLICE = 1.0
DEFAULT_VALIDATION_Q = (1.0, 0.75)
FERMI_ENERGY = 0.5


def main() -> None:
    args = _parse_args()
    grid_shape = (args.nkx, args.nky)
    qx_values = np.linspace(-args.qmax, args.qmax, args.nqx, dtype=np.float64)
    qy_values = np.linspace(-args.qmax, args.qmax, args.nqy, dtype=np.float64)
    requested_omega_values = np.linspace(0.0, args.omega_max, args.nomega, dtype=np.float64)

    evaluation_start = time.perf_counter()
    spectral_cube, real_cube, omega_values, validation = compute_response_grid(
        grid_shape=grid_shape,
        qx_values=qx_values,
        qy_values=qy_values,
        omega_values=requested_omega_values,
        workers=args.workers,
        validation_q=(args.validation_qx, args.validation_qy),
    )
    evaluation_elapsed = time.perf_counter() - evaluation_start
    figure = build_figure(
        spectral_cube=spectral_cube,
        real_cube=real_cube,
        qx_values=qx_values,
        qy_values=qy_values,
        omega_values=omega_values,
        omega_slice=args.omega_slice,
        validation=validation,
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    peak_index = np.unravel_index(int(np.argmax(spectral_cube)), spectral_cube.shape)
    peak_omega = omega_values[peak_index[0]]
    peak_qy = qy_values[peak_index[1]]
    peak_qx = qx_values[peak_index[2]]
    validation_error = np.max(np.abs(validation["reconstructed_real"] - validation["direct_real"]))
    print(f"Wrote plot to {output_path}")
    print(f"Evaluated full retarded-response grid in {evaluation_elapsed:.3f} s")
    print(
        "Grid sizes: "
        f"nkx={grid_shape[0]}, nky={grid_shape[1]}, "
        f"nqx={qx_values.size}, nqy={qy_values.size}, nomega={omega_values.size}"
    )
    if not np.isclose(omega_values[-1], requested_omega_values[-1]):
        print(
            "Adjusted omega_max to cover the full q-grid support: "
            f"{requested_omega_values[-1]:.3f} -> {omega_values[-1]:.3f}"
        )
    print(f"Used q-batch workers: {args.workers}")
    print(f"Peak spectral weight at qx={peak_qx:.3f}, qy={peak_qy:.3f}, omega={peak_omega:.3f}")
    print(
        "Validation at "
        f"qx={validation['qx_value']:.3f}, qy={validation['qy_value']:.3f}: "
        f"max |Re chi_KK - Re chi_direct(-omega)| = {validation_error:.3e}"
    )


def compute_response_grid(
    *,
    grid_shape: tuple[int, int],
    qx_values: np.ndarray,
    qy_values: np.ndarray,
    omega_values: np.ndarray,
    workers: int,
    validation_q: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    reciprocal_vectors, occupied, target_batch = build_shifted_free_electron_targets(
        grid_shape=grid_shape,
        qx_values=qx_values,
        qy_values=qy_values,
    )
    support_upper_bound = max(
        float(omega_values[-1]),
        float(np.max(target_batch[..., 0]) - np.min(occupied[..., 0])),
    )
    if support_upper_bound > float(omega_values[-1]):
        omega_values = np.linspace(float(omega_values[0]), support_upper_bound, omega_values.size, dtype=np.float64)

    prefactor = abs(np.linalg.det(reciprocal_vectors))
    sweep = prepare_response_sweep_evaluator(
        reciprocal_vectors,
        occupied,
        weight_grid_shape=grid_shape,
        method="linear",
    )
    responses = sweep.retarded_response_observables_batch(
        target_batch,
        omega_values,
        workers=workers,
    )
    flat_spectral = np.stack([response.imag for response in responses], axis=1) * prefactor / np.pi
    flat_real = np.stack([response.real for response in responses], axis=1) * prefactor
    spectral_cube = flat_spectral.reshape((omega_values.size, qy_values.size, qx_values.size))
    real_cube = flat_real.reshape((omega_values.size, qy_values.size, qx_values.size))

    validation_qx_index = int(np.argmin(np.abs(qx_values - validation_q[0])))
    validation_qy_index = int(np.argmin(np.abs(qy_values - validation_q[1])))
    validation_flat_index = validation_qy_index * qx_values.size + validation_qx_index
    validation_evaluator = sweep.prepare_target_evaluator(target_batch[validation_flat_index])
    direct_real = (
        validation_evaluator.complex_frequency_polarization_observables((-omega_values).astype(np.complex128)).real
        * prefactor
    )
    validation_payload = {
        "qx_value": float(qx_values[validation_qx_index]),
        "qy_value": float(qy_values[validation_qy_index]),
        "reconstructed_real": real_cube[:, validation_qy_index, validation_qx_index].copy(),
        "direct_real": direct_real,
    }
    return spectral_cube, real_cube, omega_values, validation_payload


def build_shifted_free_electron_targets(
    *,
    grid_shape: tuple[int, int],
    qx_values: np.ndarray,
    qy_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal_vectors = np.diag([3.0, 3.0]).astype(np.float64)
    kx, ky = _centered_cartesian_kgrid(reciprocal_vectors, grid_shape)
    occupied = (0.5 * (kx * kx + ky * ky) - FERMI_ENERGY)[..., None]

    qx_grid, qy_grid = np.meshgrid(qx_values, qy_values, indexing="xy")
    qx_flat = qx_grid.reshape(-1, 1, 1)
    qy_flat = qy_grid.reshape(-1, 1, 1)
    target_batch = 0.5 * ((kx[None, :, :] + qx_flat) ** 2 + (ky[None, :, :] + qy_flat) ** 2) - FERMI_ENERGY
    return reciprocal_vectors, occupied, target_batch[..., None]


def build_figure(
    *,
    spectral_cube: np.ndarray,
    real_cube: np.ndarray,
    qx_values: np.ndarray,
    qy_values: np.ndarray,
    omega_values: np.ndarray,
    omega_slice: float,
    validation: dict[str, np.ndarray | float],
):
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.4))
    spectral_q_axis, real_q_axis = axes[0]
    spectral_line_axis, validation_axis = axes[1]

    omega_slice_index = int(np.argmin(np.abs(omega_values - omega_slice)))
    omega_slice_value = float(omega_values[omega_slice_index])
    qy_zero_index = int(np.argmin(np.abs(qy_values)))

    spectral_q_slice = spectral_cube[omega_slice_index]
    normalized_spectral_q = spectral_q_slice / max(float(np.max(spectral_q_slice)), 1.0e-16)
    spectral_q_image = spectral_q_axis.imshow(
        normalized_spectral_q,
        origin="lower",
        aspect="equal",
        extent=(qx_values[0], qx_values[-1], qy_values[0], qy_values[-1]),
        cmap="magma",
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0),
        interpolation="nearest",
    )
    spectral_q_axis.set_title(rf"$S(q_x, q_y, \omega)$ at $\omega = {omega_slice_value:.2f}$")
    spectral_q_axis.set_xlabel(r"Momentum transfer $q_x / k_F$")
    spectral_q_axis.set_ylabel(r"Momentum transfer $q_y / k_F$")
    figure.colorbar(spectral_q_image, ax=spectral_q_axis, pad=0.02, label="Normalized spectral weight")

    real_q_slice = real_cube[omega_slice_index]
    real_limit = max(float(np.max(np.abs(real_q_slice))), 1.0e-16)
    real_q_image = real_q_axis.imshow(
        real_q_slice,
        origin="lower",
        aspect="equal",
        extent=(qx_values[0], qx_values[-1], qy_values[0], qy_values[-1]),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-real_limit, vcenter=0.0, vmax=real_limit),
        interpolation="nearest",
    )
    real_q_axis.set_title(rf"$\mathrm{{Re}}\,\Pi(q_x, q_y, -\omega + i0^+)$ at $\omega = {omega_slice_value:.2f}$")
    real_q_axis.set_xlabel(r"Momentum transfer $q_x / k_F$")
    real_q_axis.set_ylabel(r"Momentum transfer $q_y / k_F$")
    figure.colorbar(real_q_image, ax=real_q_axis, pad=0.02, label=r"$\mathrm{Re}\,\Pi$")

    normalized_linecut = spectral_cube[:, qy_zero_index, :] / max(float(np.max(spectral_cube)), 1.0e-16)
    spectral_line_image = spectral_line_axis.imshow(
        normalized_linecut,
        origin="lower",
        aspect="auto",
        extent=(qx_values[0], qx_values[-1], omega_values[0], omega_values[-1]),
        cmap="magma",
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0),
        interpolation="nearest",
    )
    spectral_line_axis.set_title(r"$q_y = 0$ slice through the full $q_x$-$q_y$-$\omega$ cube")
    spectral_line_axis.set_xlabel(r"Momentum transfer $q_x / k_F$")
    spectral_line_axis.set_ylabel(r"Energy transfer $\omega$")
    figure.colorbar(spectral_line_image, ax=spectral_line_axis, pad=0.02, label="Normalized spectral weight")

    validation_axis.plot(
        omega_values,
        validation["reconstructed_real"],
        color="#005F73",
        linewidth=2.2,
        label="KK reconstruction",
    )
    validation_axis.plot(
        omega_values,
        validation["direct_real"],
        color="#AE2012",
        linewidth=1.6,
        linestyle="--",
        label=r"direct $\Pi(-\omega + 0j)$",
    )
    validation_axis.set_title(
        "Validation at "
        rf"$q_x / k_F = {float(validation['qx_value']):.2f}$, "
        rf"$q_y / k_F = {float(validation['qy_value']):.2f}$"
    )
    validation_axis.set_xlabel(r"Energy transfer $\omega$")
    validation_axis.set_ylabel(r"$\mathrm{Re}\,\Pi$")
    validation_axis.set_xlim(omega_values[0], omega_values[-1])
    validation_axis.grid(alpha=0.22)
    validation_axis.legend(loc="upper right", fontsize=8)

    figure.suptitle("2D Free-Electron Retarded Response on a Full Momentum Grid", fontsize=14)
    figure.tight_layout()
    return figure


def _centered_cartesian_kgrid(
    reciprocal_vectors: np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    fractional_x = _centered_fractional_axis(grid_shape[0])
    fractional_y = _centered_fractional_axis(grid_shape[1])
    fractional_grid = np.stack(np.meshgrid(fractional_x, fractional_y, indexing="ij"), axis=-1)
    cartesian_grid = fractional_grid @ reciprocal_vectors.T
    return cartesian_grid[..., 0], cartesian_grid[..., 1]


def _centered_fractional_axis(point_count: int) -> np.ndarray:
    integer_axis = np.arange(point_count, dtype=np.int64)
    half_count = point_count // 2
    centered = np.mod(integer_axis + half_count, point_count) - half_count
    return centered.astype(np.float64) / float(point_count)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark and plot the 2D free-electron retarded response on a full "
            "(q_x, q_y, omega) grid using the contracted sweep evaluator."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the plot image (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--nkx", type=int, default=DEFAULT_K_GRID_SHAPE[0], help="Number of k-grid points along x")
    parser.add_argument("--nky", type=int, default=DEFAULT_K_GRID_SHAPE[1], help="Number of k-grid points along y")
    parser.add_argument("--nqx", type=int, default=DEFAULT_NQX, help="Number of q-grid points along x")
    parser.add_argument("--nqy", type=int, default=DEFAULT_NQY, help="Number of q-grid points along y")
    parser.add_argument("--nomega", type=int, default=DEFAULT_NOMEGA, help="Number of sampled frequencies")
    parser.add_argument("--qmax", type=float, default=DEFAULT_Q_MAX, help="Maximum |q_x| and |q_y| on the plot grid")
    parser.add_argument("--omega-max", type=float, default=DEFAULT_OMEGA_MAX, help="Maximum sampled frequency")
    parser.add_argument(
        "--omega-slice",
        type=float,
        default=DEFAULT_OMEGA_SLICE,
        help="Energy at which to plot the (q_x, q_y) slices",
    )
    parser.add_argument(
        "--validation-qx",
        type=float,
        default=DEFAULT_VALIDATION_Q[0],
        help="Requested q_x point for the direct complex-kernel validation cut",
    )
    parser.add_argument(
        "--validation-qy",
        type=float,
        default=DEFAULT_VALIDATION_Q[1],
        help="Requested q_y point for the direct complex-kernel validation cut",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of q-points to evaluate in parallel "
            f"(default: 1, detected cpu count: {os.cpu_count() or 1})"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
