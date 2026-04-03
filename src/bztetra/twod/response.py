from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit

from ..causality import reconstruct_retarded_response
from ..causality import RetardedResponse
from .._grids import ComplexArray
from .._grids import normalize_complex_energy_samples
from .._grids import normalize_energy_samples
from ._grids import FloatArray
from ._grids import interpolate_local_values
from ._grids import interpolated_triangle_energies
from ._grids import normalize_eigenvalues
from ._response_kernels import _complex_polarization_weights_on_local_mesh_numba
from ._response_kernels import _complex_polarization_weights_on_local_mesh_pair_parallel_numba
from ._response_kernels import _complex_polarization_observables_on_local_mesh_numba
from ._response_kernels import _complex_polarization_observables_on_local_mesh_pair_parallel_numba
from ._response_kernels import _fermi_golden_rule_weights_on_local_mesh_numba
from ._response_kernels import _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba
from ._response_kernels import _fermi_golden_rule_observables_on_local_mesh_numba
from ._response_kernels import _fermi_golden_rule_observables_on_local_mesh_pair_parallel_numba
from ._response_kernels import _nesting_function_weights_on_local_mesh_numba
from ._response_kernels import _phase_space_overlap_weights_on_local_mesh_numba
from ._response_kernels import _static_polarization_observables_on_local_mesh_numba
from ._response_kernels import _static_polarization_observables_on_local_mesh_pair_parallel_numba
from ._response_kernels import _static_polarization_weights_on_local_mesh_numba
from ._response_kernels import _static_polarization_weights_on_local_mesh_pair_parallel_numba
from .geometry import TriangleIntegrationMesh
from .geometry import TriangleMethod
from .geometry import cached_integration_mesh


PAIR_PARALLEL_THRESHOLD = 16
PAIR_PARALLEL_TARGET_THRESHOLD = 4


@dataclass(slots=True)
class PreparedResponseEvaluator:
    """Reusable setup for repeated source-to-target 2D response evaluations.

    Static methods return `(wx, wy, ntarget, nsource)` arrays with the last
    axes ordered `(target_band, source_band)`. Frequency-dependent methods
    return `(nenergy, wx, wy, ntarget, nsource)` arrays.
    """

    mesh: TriangleIntegrationMesh
    occupied_triangles: FloatArray
    target_triangles: FloatArray

    def phase_space_overlap_weights(self) -> FloatArray:
        """Evaluate the double-step phase-space overlap. Replaces `libtetrabz_dblstep`."""

        local_weights = _phase_space_overlap_weights_on_local_mesh(
            self.mesh,
            self.occupied_triangles,
            self.target_triangles,
        )
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def nesting_function_weights(self) -> FloatArray:
        """Evaluate the double-delta nesting weights. Replaces `libtetrabz_dbldelta`."""

        local_weights = _nesting_function_weights_on_local_mesh(
            self.mesh,
            self.occupied_triangles,
            self.target_triangles,
        )
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def static_polarization_weights(self) -> FloatArray:
        """Evaluate the static polarization weights. Replaces `libtetrabz_polstat`."""

        local_weights = _static_polarization_weights_on_local_mesh(
            self.mesh,
            self.occupied_triangles,
            self.target_triangles,
        )
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def static_polarization_observables(
        self,
        *,
        matrix_elements: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.generic]:
        """Evaluate directly contracted 2D static observables.

        `matrix_elements` may have shape `(..., wx, wy, ntarget, nsource)`,
        `(..., nlocal, ntarget, nsource)`, or `(..., ntarget, nsource)`. The
        leading axes are returned after contraction. When omitted, the response
        is fully contracted with unit weights and the result is a scalar.
        """

        local_matrix_elements, channel_shape = _normalize_response_matrix_elements(
            self.mesh,
            matrix_elements,
            self.target_triangles.shape[2],
            self.occupied_triangles.shape[2],
        )
        contracted = _static_polarization_observables_on_local_mesh(
            self.mesh,
            self.occupied_triangles,
            self.target_triangles,
            local_matrix_elements,
        )
        return _reshape_static_contracted_observables(contracted, channel_shape)

    def fermi_golden_rule_weights(self, energies: npt.ArrayLike) -> FloatArray:
        """Evaluate real-frequency transition weights. Replaces `libtetrabz_fermigr`."""

        sample_energies = normalize_energy_samples(energies)
        local_weights = _fermi_golden_rule_weights_on_local_mesh(
            self.mesh,
            self.occupied_triangles,
            self.target_triangles,
            sample_energies,
        )
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_energy_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def complex_frequency_polarization_weights(self, energies: npt.ArrayLike) -> ComplexArray:
        """Evaluate the complex-frequency polarization weights. Replaces `libtetrabz_polcmplx`."""

        sample_energies = normalize_complex_energy_samples(energies)
        local_weights = _complex_polarization_weights_on_local_mesh(
            self.mesh,
            self.occupied_triangles,
            self.target_triangles,
            sample_energies,
        )
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_energy_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def fermi_golden_rule_observables(
        self,
        energies: npt.ArrayLike,
        *,
        matrix_elements: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.generic]:
        """Evaluate directly contracted 2D real-frequency observables.

        `matrix_elements` may have shape `(..., wx, wy, ntarget, nsource)`,
        `(..., nlocal, ntarget, nsource)`, or `(..., ntarget, nsource)`. The
        leading axes are returned after the energy axis. When omitted, the
        response is fully contracted with unit weights and the result has shape
        `(nenergy,)`.
        """

        sample_energies = normalize_energy_samples(energies)
        local_matrix_elements, channel_shape = _normalize_response_matrix_elements(
            self.mesh,
            matrix_elements,
            self.target_triangles.shape[2],
            self.occupied_triangles.shape[2],
        )
        contracted = _fermi_golden_rule_observables_on_local_mesh(
            self.mesh,
            self.occupied_triangles,
            self.target_triangles,
            sample_energies,
            local_matrix_elements,
        )
        return _reshape_contracted_observables(contracted, channel_shape)

    def complex_frequency_polarization_observables(
        self,
        energies: npt.ArrayLike,
        *,
        matrix_elements: npt.ArrayLike | None = None,
    ) -> ComplexArray:
        """Evaluate directly contracted 2D complex-frequency observables.

        `matrix_elements` may have shape `(..., wx, wy, ntarget, nsource)`,
        `(..., nlocal, ntarget, nsource)`, or `(..., ntarget, nsource)`. The
        leading axes are returned after the energy axis. When omitted, the
        response is fully contracted with unit weights and the result has shape
        `(nenergy,)`.
        """

        sample_energies = normalize_complex_energy_samples(energies)
        local_matrix_elements, channel_shape = _normalize_response_matrix_elements(
            self.mesh,
            matrix_elements,
            self.target_triangles.shape[2],
            self.occupied_triangles.shape[2],
        )
        contracted = _complex_polarization_observables_on_local_mesh(
            self.mesh,
            self.occupied_triangles,
            self.target_triangles,
            sample_energies,
            local_matrix_elements,
        )
        return _reshape_contracted_observables(contracted, channel_shape)

    def transition_energy_bounds(self) -> tuple[float, float]:
        """Return conservative bounds for occupied-to-empty transition energies."""

        lower_bound, upper_bound = _transition_energy_bounds_numba(
            self.occupied_triangles,
            self.target_triangles,
        )
        return max(0.0, lower_bound), max(0.0, upper_bound)

    def retarded_response_observables(
        self,
        energies: npt.ArrayLike,
        *,
        matrix_elements: npt.ArrayLike | None = None,
        assume_hermitian: bool = False,
    ) -> RetardedResponse:
        """Reconstruct the retarded response from the fast real-frequency path.

        This convenience method evaluates the imaginary part from
        `fermi_golden_rule_observables`, computes the zero-frequency anchor with
        `static_polarization_observables`, derives transition-energy support
        bounds from the prepared source/target bands, and then applies the
        automatic Kramers-Kronig reconstruction. By default, this reconstructs
        the real-axis continuation associated with positive transition energies,
        which matches `complex_frequency_polarization_observables(-omega + 0j)`.
        """

        sample_energies = normalize_energy_samples(energies)
        spectral_weights = self.fermi_golden_rule_observables(
            sample_energies,
            matrix_elements=matrix_elements,
        )
        static_anchor = self.static_polarization_observables(
            matrix_elements=matrix_elements,
        )
        support_bounds = self.transition_energy_bounds()
        return reconstruct_retarded_response(
            sample_energies,
            np.pi * spectral_weights,
            static_anchor=static_anchor,
            support_bounds=support_bounds if support_bounds[1] > support_bounds[0] else None,
            assume_hermitian=assume_hermitian,
        )


@dataclass(slots=True)
class PreparedResponseSweepEvaluator:
    """Reusable 2D response setup for many targets on the same source mesh.

    This evaluator caches the integration mesh and occupied-band triangle
    energies once, then reuses them across a batch of target-band response
    evaluations such as dense `q` sweeps.
    """

    mesh: TriangleIntegrationMesh
    occupied_triangles: FloatArray

    def prepare_target_evaluator(self, target_eigenvalues: npt.ArrayLike) -> PreparedResponseEvaluator:
        """Prepare a single source-to-target evaluator that reuses cached source state."""

        target_flat, target_grid = normalize_eigenvalues(target_eigenvalues)
        if target_grid != self.mesh.energy_grid_shape:
            raise ValueError(
                "target_eigenvalues must share the prepared 2D energy-grid shape, "
                f"got {target_grid!r} and expected {self.mesh.energy_grid_shape!r}"
            )
        target_triangles = interpolated_triangle_energies(self.mesh, target_flat)
        return PreparedResponseEvaluator(
            mesh=self.mesh,
            occupied_triangles=self.occupied_triangles,
            target_triangles=target_triangles,
        )

    def static_polarization_observables_batch(
        self,
        target_eigenvalues_batch: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
        *,
        matrix_elements: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...] | None = None,
        workers: int = 1,
    ) -> npt.NDArray[np.generic]:
        """Evaluate contracted static observables for many targets."""

        return np.stack(
            self._run_target_batch(
                target_eigenvalues_batch,
                workers=workers,
                matrix_elements=matrix_elements,
                evaluate=lambda evaluator, target_matrix_elements: evaluator.static_polarization_observables(
                    matrix_elements=target_matrix_elements,
                ),
            ),
            axis=0,
        )

    def fermi_golden_rule_observables_batch(
        self,
        target_eigenvalues_batch: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
        energies: npt.ArrayLike,
        *,
        matrix_elements: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...] | None = None,
        workers: int = 1,
    ) -> npt.NDArray[np.generic]:
        """Evaluate contracted real-frequency observables for many targets."""

        sample_energies = normalize_energy_samples(energies)
        return np.stack(
            self._run_target_batch(
                target_eigenvalues_batch,
                workers=workers,
                matrix_elements=matrix_elements,
                evaluate=lambda evaluator, target_matrix_elements: evaluator.fermi_golden_rule_observables(
                    sample_energies,
                    matrix_elements=target_matrix_elements,
                ),
            ),
            axis=0,
        )

    def complex_frequency_polarization_observables_batch(
        self,
        target_eigenvalues_batch: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
        energies: npt.ArrayLike,
        *,
        matrix_elements: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...] | None = None,
        workers: int = 1,
    ) -> ComplexArray:
        """Evaluate contracted complex-frequency observables for many targets."""

        sample_energies = normalize_complex_energy_samples(energies)
        return np.stack(
            self._run_target_batch(
                target_eigenvalues_batch,
                workers=workers,
                matrix_elements=matrix_elements,
                evaluate=lambda evaluator, target_matrix_elements: evaluator.complex_frequency_polarization_observables(
                    sample_energies,
                    matrix_elements=target_matrix_elements,
                ),
            ),
            axis=0,
        )

    def retarded_response_observables_batch(
        self,
        target_eigenvalues_batch: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
        energies: npt.ArrayLike,
        *,
        matrix_elements: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...] | None = None,
        workers: int = 1,
        assume_hermitian: bool = False,
    ) -> tuple[RetardedResponse, ...]:
        """Reconstruct retarded responses for many targets."""

        sample_energies = normalize_energy_samples(energies)
        return self._run_target_batch(
            target_eigenvalues_batch,
            workers=workers,
            matrix_elements=matrix_elements,
            evaluate=lambda evaluator, target_matrix_elements: evaluator.retarded_response_observables(
                sample_energies,
                matrix_elements=target_matrix_elements,
                assume_hermitian=assume_hermitian,
            ),
        )

    def _run_target_batch(
        self,
        target_eigenvalues_batch: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
        *,
        workers: int,
        matrix_elements: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...] | None,
        evaluate,
    ) -> tuple[object, ...]:
        target_batch = _normalize_target_eigenvalue_batch(target_eigenvalues_batch)
        matrix_element_batch = _normalize_batched_optional_inputs(matrix_elements, len(target_batch))
        worker_count = _normalize_worker_count(workers)

        if worker_count == 1 or len(target_batch) == 1:
            return tuple(
                evaluate(self.prepare_target_evaluator(target_values), target_matrix_elements)
                for target_values, target_matrix_elements in zip(target_batch, matrix_element_batch, strict=True)
            )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _evaluate_sweep_target,
                    self,
                    target_values,
                    target_matrix_elements,
                    evaluate,
                )
                for target_values, target_matrix_elements in zip(target_batch, matrix_element_batch, strict=True)
            ]
            return tuple(future.result() for future in futures)


def prepare_response_sweep_evaluator(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> PreparedResponseSweepEvaluator:
    """Prepare reusable 2D source-state response data for many target sweeps."""

    occupied_flat, occupied_grid = normalize_eigenvalues(occupied_eigenvalues)
    mesh = cached_integration_mesh(
        reciprocal_vectors,
        occupied_grid,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    occupied_triangles = interpolated_triangle_energies(mesh, occupied_flat)
    return PreparedResponseSweepEvaluator(
        mesh=mesh,
        occupied_triangles=occupied_triangles,
    )


def prepare_response_evaluator(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> PreparedResponseEvaluator:
    """Prepare reusable 2D response state for repeated source-to-target sweeps.

    Both eigenvalue inputs must have shape `(nx, ny, nbands)`. The prepared
    evaluator reuses mesh and triangle setup across static, real-frequency,
    and complex-frequency response calls.
    """

    occupied_flat, occupied_grid = normalize_eigenvalues(occupied_eigenvalues)
    target_flat, target_grid = normalize_eigenvalues(target_eigenvalues)
    if occupied_grid != target_grid:
        raise ValueError(
            "occupied_eigenvalues and target_eigenvalues must share the same "
            f"2D energy-grid shape, got {occupied_grid!r} and {target_grid!r}"
        )

    mesh = cached_integration_mesh(
        reciprocal_vectors,
        occupied_grid,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    occupied_triangles = interpolated_triangle_energies(mesh, occupied_flat)
    target_triangles = interpolated_triangle_energies(mesh, target_flat)
    return PreparedResponseEvaluator(
        mesh=mesh,
        occupied_triangles=occupied_triangles,
        target_triangles=target_triangles,
    )


def phase_space_overlap_weights(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> FloatArray:
    """Evaluate the 2D double-step phase-space overlap.

    The result has shape `(wx, wy, ntarget, nsource)` with the last axes
    ordered `(target_band, source_band)`. Replaces `libtetrabz_dblstep`.
    """

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.phase_space_overlap_weights()


def nesting_function_weights(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> FloatArray:
    """Evaluate the 2D double-delta nesting function.

    The result has shape `(wx, wy, ntarget, nsource)` with the last axes
    ordered `(target_band, source_band)`. Replaces `libtetrabz_dbldelta`.
    """

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.nesting_function_weights()


def static_polarization_weights(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> FloatArray:
    """Evaluate the 2D static polarization function.

    The result has shape `(wx, wy, ntarget, nsource)` with the last axes
    ordered `(target_band, source_band)`. Replaces `libtetrabz_polstat`.
    """

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.static_polarization_weights()


def static_polarization_observables(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    matrix_elements: npt.ArrayLike | None = None,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> npt.NDArray[np.generic]:
    """Evaluate directly contracted 2D static observables."""

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.static_polarization_observables(
        matrix_elements=matrix_elements,
    )


def fermi_golden_rule_weights(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> FloatArray:
    """Evaluate 2D real-frequency transition weights.

    `energies` must be one-dimensional. The result has shape
    `(nenergy, wx, wy, ntarget, nsource)` with the last axes ordered
    `(target_band, source_band)`. Replaces `libtetrabz_fermigr`.
    """

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.fermi_golden_rule_weights(energies)


def complex_frequency_polarization_weights(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> ComplexArray:
    """Evaluate the 2D complex-frequency polarization function.

    `energies` must be one-dimensional. The result has shape
    `(nenergy, wx, wy, ntarget, nsource)` with the last axes ordered
    `(target_band, source_band)`. Replaces `libtetrabz_polcmplx`.
    """

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.complex_frequency_polarization_weights(energies)


def fermi_golden_rule_observables(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    matrix_elements: npt.ArrayLike | None = None,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> npt.NDArray[np.generic]:
    """Evaluate directly contracted 2D real-frequency observables.

    `matrix_elements` may have shape `(..., wx, wy, ntarget, nsource)`,
    `(..., nlocal, ntarget, nsource)`, or `(..., ntarget, nsource)`. The
    leading axes are returned after the energy axis. When omitted, the
    response is fully contracted with unit weights and the result has shape
    `(nenergy,)`.
    """

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.fermi_golden_rule_observables(
        energies,
        matrix_elements=matrix_elements,
    )


def complex_frequency_polarization_observables(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    matrix_elements: npt.ArrayLike | None = None,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
) -> ComplexArray:
    """Evaluate directly contracted 2D complex-frequency observables.

    `matrix_elements` may have shape `(..., wx, wy, ntarget, nsource)`,
    `(..., nlocal, ntarget, nsource)`, or `(..., ntarget, nsource)`. The
    leading axes are returned after the energy axis. When omitted, the
    response is fully contracted with unit weights and the result has shape
    `(nenergy,)`.
    """

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.complex_frequency_polarization_observables(
        energies,
        matrix_elements=matrix_elements,
    )


def retarded_response_observables(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    matrix_elements: npt.ArrayLike | None = None,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
    assume_hermitian: bool = False,
) -> RetardedResponse:
    """Reconstruct the retarded response from real-frequency spectral weights."""

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.retarded_response_observables(
        energies,
        matrix_elements=matrix_elements,
        assume_hermitian=assume_hermitian,
    )


def fermi_golden_rule_observables_batch(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues_batch: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
    energies: npt.ArrayLike,
    *,
    matrix_elements: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...] | None = None,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
    workers: int = 1,
) -> npt.NDArray[np.generic]:
    """Evaluate contracted real-frequency observables for many targets."""

    sweep = prepare_response_sweep_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return sweep.fermi_golden_rule_observables_batch(
        target_eigenvalues_batch,
        energies,
        matrix_elements=matrix_elements,
        workers=workers,
    )


def retarded_response_observables_batch(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues_batch: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
    energies: npt.ArrayLike,
    *,
    matrix_elements: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...] | None = None,
    weight_grid_shape: tuple[int, int] | None = None,
    method: int | TriangleMethod = "linear",
    workers: int = 1,
    assume_hermitian: bool = False,
) -> tuple[RetardedResponse, ...]:
    """Reconstruct retarded responses for many targets."""

    sweep = prepare_response_sweep_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return sweep.retarded_response_observables_batch(
        target_eigenvalues_batch,
        energies,
        matrix_elements=matrix_elements,
        workers=workers,
        assume_hermitian=assume_hermitian,
    )


def _phase_space_overlap_weights_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    occupied_triangles: FloatArray,
    target_triangles: FloatArray,
) -> FloatArray:
    return _phase_space_overlap_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        mesh.local_point_count,
        _triangle_area(mesh),
    )


def _nesting_function_weights_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    occupied_triangles: FloatArray,
    target_triangles: FloatArray,
) -> FloatArray:
    return _nesting_function_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        mesh.local_point_count,
        _triangle_area(mesh),
    )


def _static_polarization_weights_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    occupied_triangles: FloatArray,
    target_triangles: FloatArray,
) -> FloatArray:
    pair_count = occupied_triangles.shape[2] * target_triangles.shape[2]
    if pair_count >= PAIR_PARALLEL_THRESHOLD and target_triangles.shape[2] >= PAIR_PARALLEL_TARGET_THRESHOLD:
        return _static_polarization_weights_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            occupied_triangles,
            target_triangles,
            mesh.local_point_count,
            _triangle_area(mesh),
        )
    return _static_polarization_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        mesh.local_point_count,
        _triangle_area(mesh),
    )


def _static_polarization_observables_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    occupied_triangles: FloatArray,
    target_triangles: FloatArray,
    local_matrix_elements: npt.NDArray[np.generic],
) -> npt.NDArray[np.generic]:
    pair_count = occupied_triangles.shape[2] * target_triangles.shape[2]
    if pair_count >= PAIR_PARALLEL_THRESHOLD and target_triangles.shape[2] >= PAIR_PARALLEL_TARGET_THRESHOLD:
        return _static_polarization_observables_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            occupied_triangles,
            target_triangles,
            local_matrix_elements,
            _triangle_area(mesh),
        )
    return _static_polarization_observables_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        local_matrix_elements,
        _triangle_area(mesh),
    )


def _fermi_golden_rule_weights_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    occupied_triangles: FloatArray,
    target_triangles: FloatArray,
    sample_energies: FloatArray,
) -> FloatArray:
    pair_count = occupied_triangles.shape[2] * target_triangles.shape[2]
    sample_energies_sorted = bool(np.all(sample_energies[1:] >= sample_energies[:-1]))
    if pair_count >= PAIR_PARALLEL_THRESHOLD and target_triangles.shape[2] >= PAIR_PARALLEL_TARGET_THRESHOLD:
        return _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            occupied_triangles,
            target_triangles,
            sample_energies,
            sample_energies_sorted,
            mesh.local_point_count,
            _triangle_area(mesh),
        )
    return _fermi_golden_rule_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        sample_energies,
        sample_energies_sorted,
        mesh.local_point_count,
        _triangle_area(mesh),
    )


def _complex_polarization_weights_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    occupied_triangles: FloatArray,
    target_triangles: FloatArray,
    sample_energies: ComplexArray,
) -> ComplexArray:
    pair_count = occupied_triangles.shape[2] * target_triangles.shape[2]
    if pair_count >= PAIR_PARALLEL_THRESHOLD and target_triangles.shape[2] >= PAIR_PARALLEL_TARGET_THRESHOLD:
        return _complex_polarization_weights_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            occupied_triangles,
            target_triangles,
            sample_energies,
            mesh.local_point_count,
            _triangle_area(mesh),
        )
    return _complex_polarization_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        sample_energies,
        mesh.local_point_count,
        _triangle_area(mesh),
    )


def _fermi_golden_rule_observables_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    occupied_triangles: FloatArray,
    target_triangles: FloatArray,
    sample_energies: FloatArray,
    local_matrix_elements: npt.NDArray[np.generic],
) -> npt.NDArray[np.generic]:
    pair_count = occupied_triangles.shape[2] * target_triangles.shape[2]
    sample_energies_sorted = bool(np.all(sample_energies[1:] >= sample_energies[:-1]))
    if pair_count >= PAIR_PARALLEL_THRESHOLD and target_triangles.shape[2] >= PAIR_PARALLEL_TARGET_THRESHOLD:
        return _fermi_golden_rule_observables_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            occupied_triangles,
            target_triangles,
            sample_energies,
            sample_energies_sorted,
            local_matrix_elements,
            _triangle_area(mesh),
        )
    return _fermi_golden_rule_observables_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        sample_energies,
        sample_energies_sorted,
        local_matrix_elements,
        _triangle_area(mesh),
    )


def _complex_polarization_observables_on_local_mesh(
    mesh: TriangleIntegrationMesh,
    occupied_triangles: FloatArray,
    target_triangles: FloatArray,
    sample_energies: ComplexArray,
    local_matrix_elements: npt.NDArray[np.generic],
) -> ComplexArray:
    pair_count = occupied_triangles.shape[2] * target_triangles.shape[2]
    if pair_count >= PAIR_PARALLEL_THRESHOLD and target_triangles.shape[2] >= PAIR_PARALLEL_TARGET_THRESHOLD:
        return _complex_polarization_observables_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            occupied_triangles,
            target_triangles,
            sample_energies,
            local_matrix_elements,
            _triangle_area(mesh),
        )
    return _complex_polarization_observables_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        sample_energies,
        local_matrix_elements,
        _triangle_area(mesh),
    )


def _triangle_area(mesh: TriangleIntegrationMesh) -> float:
    return 0.5 / float(np.prod(mesh.energy_grid_shape, dtype=np.int64))


def _unflatten_pair_band_last(values: FloatArray, grid_shape: tuple[int, int]) -> FloatArray:
    target_band_count = values.shape[1]
    source_band_count = values.shape[2]
    reshaped = values.reshape((grid_shape[1], grid_shape[0], target_band_count, source_band_count))
    return np.transpose(reshaped, (1, 0, 2, 3))


def _unflatten_energy_pair_band_last(
    values: npt.NDArray[np.generic],
    grid_shape: tuple[int, int],
) -> npt.NDArray[np.generic]:
    energy_count = values.shape[1]
    target_band_count = values.shape[2]
    source_band_count = values.shape[3]
    reshaped = values.reshape(
        (grid_shape[1], grid_shape[0], energy_count, target_band_count, source_band_count)
    )
    return np.transpose(reshaped, (2, 1, 0, 3, 4))


def _normalize_response_matrix_elements(
    mesh: TriangleIntegrationMesh,
    matrix_elements: npt.ArrayLike | None,
    target_band_count: int,
    source_band_count: int,
) -> tuple[npt.NDArray[np.generic], tuple[int, ...]]:
    if matrix_elements is None:
        local = np.ones(
            (mesh.local_point_count, target_band_count, source_band_count, 1),
            dtype=np.float64,
        )
        return local, ()

    if not np.issubdtype(np.asarray(matrix_elements).dtype, np.number):
        raise ValueError("matrix_elements must be numeric")

    dtype = np.complex128 if np.iscomplexobj(matrix_elements) else np.float64
    values = np.asarray(matrix_elements, dtype=dtype)
    expected_local_tail = (mesh.local_point_count, target_band_count, source_band_count)
    expected_grid_tail = mesh.weight_grid_shape + (target_band_count, source_band_count)

    if values.ndim >= 3 and values.shape[-3:] == expected_local_tail:
        channel_shape = tuple(int(item) for item in values.shape[:-3])
        channel_count = _flattened_channel_count(channel_shape)
        flattened = np.ascontiguousarray(
            values.reshape((channel_count, mesh.local_point_count, target_band_count, source_band_count))
        )
        return np.ascontiguousarray(np.transpose(flattened, (1, 2, 3, 0))), channel_shape

    if values.ndim >= 4 and values.shape[-4:] == expected_grid_tail:
        channel_shape = tuple(int(item) for item in values.shape[:-4])
        channel_count = _flattened_channel_count(channel_shape)
        point_count = int(np.prod(mesh.weight_grid_shape, dtype=np.int64))
        flattened = np.ascontiguousarray(
            values.reshape(
                (
                    channel_count,
                    mesh.weight_grid_shape[0],
                    mesh.weight_grid_shape[1],
                    target_band_count,
                    source_band_count,
                )
            )
        )
        flattened = np.transpose(flattened, (0, 2, 1, 3, 4)).reshape(
            (channel_count, point_count, target_band_count, source_band_count)
        )
        grid_matrix_elements = np.ascontiguousarray(np.transpose(flattened, (1, 2, 3, 0)))
        if not mesh.interpolation_required:
            return grid_matrix_elements, channel_shape
        if mesh.interpolation_indices is None or mesh.interpolation_weights is None:
            raise ValueError("interpolated mesh requires cached interpolation stencils")
        local = np.zeros(
            (mesh.local_point_count, target_band_count, source_band_count, channel_count),
            dtype=dtype,
        )
        _pull_weight_grid_matrix_elements_to_local_points_numba(
            mesh.interpolation_indices,
            mesh.interpolation_weights,
            grid_matrix_elements,
            local,
        )
        return local, channel_shape

    if values.ndim >= 2 and values.shape[-2:] == (target_band_count, source_band_count):
        channel_shape = tuple(int(item) for item in values.shape[:-2])
        channel_count = _flattened_channel_count(channel_shape)
        flattened = np.ascontiguousarray(values.reshape((channel_count, target_band_count, source_band_count)))
        local = np.empty(
            (mesh.local_point_count, target_band_count, source_band_count, channel_count),
            dtype=dtype,
        )
        local[:] = np.transpose(flattened, (1, 2, 0))
        return local, channel_shape

    raise ValueError(
        "matrix_elements must have shape (..., wx, wy, ntarget, nsource), "
        "(..., nlocal, ntarget, nsource), or (..., ntarget, nsource)"
    )


def _flattened_channel_count(channel_shape: tuple[int, ...]) -> int:
    return int(np.prod(channel_shape, dtype=np.int64)) if channel_shape else 1


def _reshape_contracted_observables(
    values: npt.NDArray[np.generic],
    channel_shape: tuple[int, ...],
) -> npt.NDArray[np.generic]:
    if not channel_shape:
        return values[:, 0]
    return values.reshape((values.shape[0],) + channel_shape)


def _reshape_static_contracted_observables(
    values: npt.NDArray[np.generic],
    channel_shape: tuple[int, ...],
) -> npt.NDArray[np.generic]:
    if not channel_shape:
        return values[0]
    return values.reshape(channel_shape)


def _normalize_target_eigenvalue_batch(
    target_eigenvalues_batch: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
) -> tuple[npt.ArrayLike, ...]:
    if isinstance(target_eigenvalues_batch, np.ndarray):
        values = np.asarray(target_eigenvalues_batch, dtype=np.float64)
        if values.ndim != 4:
            raise ValueError(
                "target_eigenvalues_batch must have shape (nq, nx, ny, nbands) "
                "or be a sequence of (nx, ny, nbands) arrays"
            )
        return tuple(values[index] for index in range(values.shape[0]))

    batch = tuple(target_eigenvalues_batch)
    if not batch:
        raise ValueError("target_eigenvalues_batch must contain at least one target")
    return batch


def _normalize_batched_optional_inputs(
    values: npt.ArrayLike | list[npt.ArrayLike] | tuple[npt.ArrayLike, ...] | None,
    batch_count: int,
) -> tuple[npt.ArrayLike | None, ...]:
    if values is None:
        return (None,) * batch_count
    if isinstance(values, (list, tuple)):
        if len(values) != batch_count:
            raise ValueError(
                "per-target batch input must have the same length as target_eigenvalues_batch, "
                f"got {len(values)} and {batch_count}"
            )
        return tuple(values)
    return (values,) * batch_count


def _normalize_worker_count(workers: int) -> int:
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be at least 1")
    return worker_count


def _evaluate_sweep_target(
    sweep: PreparedResponseSweepEvaluator,
    target_eigenvalues: npt.ArrayLike,
    matrix_elements: npt.ArrayLike | None,
    evaluate,
):
    return evaluate(
        sweep.prepare_target_evaluator(target_eigenvalues),
        matrix_elements,
    )


@njit(cache=True)
def _pull_weight_grid_matrix_elements_to_local_points_numba(
    interpolation_indices,
    interpolation_weights,
    grid_matrix_elements,
    local_matrix_elements,
) -> None:
    target_band_count = grid_matrix_elements.shape[1]
    source_band_count = grid_matrix_elements.shape[2]
    channel_count = grid_matrix_elements.shape[3]

    for local_index in range(interpolation_indices.shape[0]):
        for stencil_index in range(interpolation_indices.shape[1]):
            weight = interpolation_weights[local_index, stencil_index]
            if weight == 0.0:
                continue
            grid_index = interpolation_indices[local_index, stencil_index]
            for target_band_index in range(target_band_count):
                for source_band_index in range(source_band_count):
                    for channel_index in range(channel_count):
                        local_matrix_elements[
                            local_index,
                            target_band_index,
                            source_band_index,
                            channel_index,
                        ] += (
                            weight
                            * grid_matrix_elements[
                                grid_index,
                                target_band_index,
                                source_band_index,
                                channel_index,
                            ]
                        )


@njit(cache=True)
def _transition_energy_bounds_numba(
    occupied_triangles: FloatArray,
    target_triangles: FloatArray,
) -> tuple[float, float]:
    lower_bound = np.inf
    upper_bound = -np.inf

    for triangle_index in range(occupied_triangles.shape[0]):
        for vertex_index in range(occupied_triangles.shape[1]):
            for source_band_index in range(occupied_triangles.shape[2]):
                occupied_value = occupied_triangles[triangle_index, vertex_index, source_band_index]
                for target_band_index in range(target_triangles.shape[2]):
                    delta = (
                        target_triangles[triangle_index, vertex_index, target_band_index]
                        - occupied_value
                    )
                    if delta < lower_bound:
                        lower_bound = delta
                    if delta > upper_bound:
                        upper_bound = delta

    return lower_bound, upper_bound
