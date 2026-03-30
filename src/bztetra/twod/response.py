from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._grids import ComplexArray
from .._grids import normalize_complex_energy_samples
from .._grids import normalize_energy_samples
from ._grids import FloatArray
from ._grids import interpolate_local_values
from ._grids import interpolated_triangle_energies
from ._grids import normalize_eigenvalues
from ._response_kernels import _complex_polarization_weights_on_local_mesh_numba
from ._response_kernels import _complex_polarization_weights_on_local_mesh_pair_parallel_numba
from ._response_kernels import _fermi_golden_rule_weights_on_local_mesh_numba
from ._response_kernels import _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba
from ._response_kernels import _nesting_function_weights_on_local_mesh_numba
from ._response_kernels import _phase_space_overlap_weights_on_local_mesh_numba
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
