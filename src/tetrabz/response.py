from __future__ import annotations

from dataclasses import dataclass

import numpy.typing as npt

from ._grids import FloatArray
from ._grids import interpolate_local_values
from ._grids import interpolated_tetrahedron_energies
from ._grids import normalize_complex_energy_samples
from ._grids import normalize_energy_samples
from ._response_common import ComplexArray
from ._response_common import _normalize_eigenvalue_pair
from ._response_common import _unflatten_energy_pair_band_last
from ._response_common import _unflatten_pair_band_last
from ._response_frequency import _complex_polarization_weights_on_local_mesh
from ._response_frequency import _fermi_golden_rule_weights_on_local_mesh
from ._response_static import _double_delta_weights_on_local_mesh
from ._response_static import _double_step_weights_on_local_mesh
from ._response_static import _static_polarization_weights_on_local_mesh
from .geometry import IntegrationMesh
from .geometry import TetraMethod
from .geometry import cached_integration_mesh


@dataclass(slots=True)
class PreparedResponseEvaluator:
    """Reusable setup for repeated source-to-target response evaluations.

    Static methods return `(wx, wy, wz, ntarget, nsource)` arrays with the last
    axes ordered `(target_band, source_band)`. Frequency-dependent methods return
    `(nenergy, wx, wy, wz, ntarget, nsource)` arrays.
    """

    mesh: IntegrationMesh
    occupied_tetra: FloatArray
    target_tetra: FloatArray

    def phase_space_overlap_weights(self) -> FloatArray:
        """Evaluate the double-step phase-space overlap. Replaces `libtetrabz_dblstep`."""

        local_weights = _double_step_weights_on_local_mesh(
            self.mesh,
            self.occupied_tetra,
            self.target_tetra,
        )
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def nesting_function_weights(self) -> FloatArray:
        """Evaluate the double-delta nesting weights. Replaces `libtetrabz_dbldelta`."""

        local_weights = _double_delta_weights_on_local_mesh(
            self.mesh,
            self.occupied_tetra,
            self.target_tetra,
        )
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def static_polarization_weights(self) -> FloatArray:
        """Evaluate the static polarization weights. Replaces `libtetrabz_polstat`."""

        local_weights = _static_polarization_weights_on_local_mesh(
            self.mesh,
            self.occupied_tetra,
            self.target_tetra,
        )
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def fermi_golden_rule_weights(self, energies: npt.ArrayLike) -> FloatArray:
        """Evaluate real-frequency transition weights. Replaces `libtetrabz_fermigr`."""

        sample_energies = normalize_energy_samples(energies)
        local_weights = _fermi_golden_rule_weights_on_local_mesh(
            self.mesh,
            self.occupied_tetra,
            self.target_tetra,
            sample_energies,
        )
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_energy_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def complex_frequency_polarization_weights(self, energies: npt.ArrayLike) -> ComplexArray:
        """Evaluate the complex-frequency polarization weights. Replaces `libtetrabz_polcmplx`."""

        sample_energies = normalize_complex_energy_samples(energies)
        local_weights = _complex_polarization_weights_on_local_mesh(
            self.mesh,
            self.occupied_tetra,
            self.target_tetra,
            sample_energies,
        )
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_energy_pair_band_last(output_flat, self.mesh.weight_grid_shape)


def prepare_response_evaluator(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> PreparedResponseEvaluator:
    """Prepare reusable response state for repeated source-to-target sweeps.

    `occupied_eigenvalues` and `target_eigenvalues` must both have shape
    `(nx, ny, nz, nbands)`. The prepared evaluator reuses mesh and tetrahedron
    setup across static, real-frequency, and complex-frequency response calls.
    Set `method="linear"` only when reproducing the legacy linear tetrahedron
    scheme.
    """

    occupied_flat, target_flat, energy_grid_shape = _normalize_eigenvalue_pair(
        occupied_eigenvalues,
        target_eigenvalues,
    )
    mesh = cached_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    occupied_tetra = interpolated_tetrahedron_energies(mesh, occupied_flat)
    target_tetra = interpolated_tetrahedron_energies(mesh, target_flat)
    return PreparedResponseEvaluator(
        mesh=mesh, occupied_tetra=occupied_tetra, target_tetra=target_tetra
    )


def phase_space_overlap_weights(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    """Evaluate the double-step phase-space overlap.

    The result has shape `(wx, wy, wz, ntarget, nsource)` with the last axes
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
    source_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    """Evaluate the double-delta nesting function.

    The result has shape `(wx, wy, wz, ntarget, nsource)` with the last axes
    ordered `(target_band, source_band)`. Replaces `libtetrabz_dbldelta`.
    """

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        source_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.nesting_function_weights()


def fermi_golden_rule_weights(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    """Evaluate real-frequency transition weights.

    `energies` must be one-dimensional. The result has shape
    `(nenergy, wx, wy, wz, ntarget, nsource)` with the last axes ordered
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
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> ComplexArray:
    """Evaluate the complex-frequency polarization function.

    `energies` must be one-dimensional. The result has shape
    `(nenergy, wx, wy, wz, ntarget, nsource)` with the last axes ordered
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


def static_polarization_weights(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    """Evaluate the static polarization function.

    The result has shape `(wx, wy, wz, ntarget, nsource)` with the last axes
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
