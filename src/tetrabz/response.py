from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit
from numba import prange

from ._cut_kernels import accumulate_triangle_weight_sums
from ._cut_kernels import small_tetra_volume_and_coefficients
from ._cut_kernels import sort4
from ._cut_kernels import triangle_volume_and_coefficients
from ._grids import FloatArray
from ._grids import normalize_complex_energy_samples
from ._grids import interpolate_local_values
from ._grids import interpolated_tetrahedron_energies
from ._grids import normalize_energy_samples
from ._grids import normalize_eigenvalues
from .formulas import small_tetrahedron_cut
from .formulas import triangle_cut
from .geometry import IntegrationMesh
from .geometry import TetraMethod
from .geometry import cached_integration_mesh


ComplexArray = npt.NDArray[np.complex128]


@dataclass(slots=True)
class PreparedResponseEvaluator:
    """Reusable response setup for repeated evaluations on a fixed source/target band set."""

    mesh: IntegrationMesh
    occupied_tetra: FloatArray
    target_tetra: FloatArray

    def phase_space_overlap_weights(self) -> FloatArray:
        """Evaluate the double-step phase-space overlap. Replaces `libtetrabz_dblstep`."""

        local_weights = _double_step_weights_on_local_mesh(self.mesh, self.occupied_tetra, self.target_tetra)
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def nesting_function_weights(self) -> FloatArray:
        """Evaluate the double-delta nesting weights. Replaces `libtetrabz_dbldelta`."""

        local_weights = _double_delta_weights_on_local_mesh(self.mesh, self.occupied_tetra, self.target_tetra)
        output_flat = interpolate_local_values(self.mesh, local_weights)
        return _unflatten_pair_band_last(output_flat, self.mesh.weight_grid_shape)

    def static_polarization_weights(self) -> FloatArray:
        """Evaluate the static polarization weights. Replaces `libtetrabz_polstat`."""

        local_weights = _static_polarization_weights_on_local_mesh(self.mesh, self.occupied_tetra, self.target_tetra)
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
    """Prepare reusable response state for repeated source/target sweeps."""

    occupied_flat, target_flat, energy_grid_shape = _normalize_eigenvalue_pair(occupied_eigenvalues, target_eigenvalues)
    mesh = cached_integration_mesh(
        reciprocal_vectors,
        energy_grid_shape,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    occupied_tetra = interpolated_tetrahedron_energies(mesh, occupied_flat)
    target_tetra = interpolated_tetrahedron_energies(mesh, target_flat)
    return PreparedResponseEvaluator(mesh=mesh, occupied_tetra=occupied_tetra, target_tetra=target_tetra)


def phase_space_overlap_weights(
    reciprocal_vectors: npt.ArrayLike,
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
    *,
    weight_grid_shape: tuple[int, int, int] | None = None,
    method: int | TetraMethod = "optimized",
) -> FloatArray:
    """Evaluate the double-step phase-space overlap. Replaces `libtetrabz_dblstep`."""

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
    """Evaluate the double-delta nesting function. Replaces `libtetrabz_dbldelta`."""

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
    """Evaluate real-frequency transition weights. Replaces `libtetrabz_fermigr`."""

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
    """Evaluate the complex-frequency polarization function. Replaces `libtetrabz_polcmplx`."""

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
    """Evaluate the static polarization function. Replaces `libtetrabz_polstat`."""

    evaluator = prepare_response_evaluator(
        reciprocal_vectors,
        occupied_eigenvalues,
        target_eigenvalues,
        weight_grid_shape=weight_grid_shape,
        method=method,
    )
    return evaluator.static_polarization_weights()


def _double_step_weights_on_local_mesh(
    mesh: IntegrationMesh,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
) -> FloatArray:
    normalization = 6 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return _double_step_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        occupied_tetra,
        target_tetra,
        mesh.local_point_count,
        normalization,
    )


def _double_delta_weights_on_local_mesh(
    mesh: IntegrationMesh,
    source_tetra: FloatArray,
    target_tetra: FloatArray,
) -> FloatArray:
    normalization = 6 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return _double_delta_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        source_tetra,
        target_tetra,
        mesh.local_point_count,
        normalization,
    )


def _fermi_golden_rule_weights_on_local_mesh(
    mesh: IntegrationMesh,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: FloatArray,
) -> FloatArray:
    normalization = 6 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    pair_count = occupied_tetra.shape[2] * target_tetra.shape[2]
    if pair_count >= 16 and target_tetra.shape[2] >= 4:
        return _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            mesh.tetrahedron_weight_matrix,
            occupied_tetra,
            target_tetra,
            sample_energies,
            mesh.local_point_count,
            normalization,
        )
    return _fermi_golden_rule_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        occupied_tetra,
        target_tetra,
        sample_energies,
        mesh.local_point_count,
        normalization,
    )


def _complex_polarization_weights_on_local_mesh(
    mesh: IntegrationMesh,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: ComplexArray,
) -> ComplexArray:
    normalization = 6 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    pair_count = occupied_tetra.shape[2] * target_tetra.shape[2]
    if pair_count >= 16 and target_tetra.shape[2] >= 4:
        return _complex_polarization_weights_on_local_mesh_pair_parallel_numba(
            mesh.local_point_indices,
            mesh.tetrahedron_weight_matrix,
            occupied_tetra,
            target_tetra,
            sample_energies,
            mesh.local_point_count,
            normalization,
        )
    return _complex_polarization_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        occupied_tetra,
        target_tetra,
        sample_energies,
        mesh.local_point_count,
        normalization,
    )


def _static_polarization_weights_on_local_mesh(
    mesh: IntegrationMesh,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
) -> FloatArray:
    normalization = 6 * int(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    return _static_polarization_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        mesh.tetrahedron_weight_matrix,
        occupied_tetra,
        target_tetra,
        mesh.local_point_count,
        normalization,
    )


def _accumulate_small_tetra_response(
    weights: FloatArray,
    kind: str,
    sorted_order: npt.NDArray[np.int64],
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    inner_kernel,
) -> None:
    cut = small_tetrahedron_cut(kind, sorted_occupied)
    transformed_occupied = cut.coefficients @ sorted_occupied
    transformed_target = cut.coefficients @ sorted_target
    weights[:, sorted_order] += cut.volume_factor * (inner_kernel(transformed_occupied, transformed_target) @ cut.coefficients)


def _accumulate_triangle_response(
    weights: FloatArray,
    kind: str,
    sorted_order: npt.NDArray[np.int64],
    sorted_source: FloatArray,
    sorted_target: FloatArray,
) -> None:
    cut = triangle_cut(kind, sorted_source)
    transformed_target = cut.coefficients @ sorted_target
    weights[:, sorted_order] += cut.volume_factor * (_double_delta_secondary_weights(transformed_target) @ cut.coefficients)


def _accumulate_small_tetra_polstat_outer(
    weights: FloatArray,
    kind: str,
    sorted_order: npt.NDArray[np.int64],
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
) -> None:
    cut = small_tetrahedron_cut(kind, sorted_occupied)
    if cut.volume_factor <= 1.0e-10:
        return

    transformed_occupied = cut.coefficients @ sorted_occupied
    transformed_target = cut.coefficients @ sorted_target
    weights[:, sorted_order] += cut.volume_factor * (_polstat_secondary_weights(transformed_occupied, transformed_target) @ cut.coefficients)


def _double_step_secondary_weights(occupied_vertices: FloatArray, target_vertices: FloatArray) -> FloatArray:
    target_band_count = target_vertices.shape[1]
    weights = np.zeros((target_band_count, 4), dtype=np.float64)

    for target_band_index in range(target_band_count):
        energy_difference = -occupied_vertices + target_vertices[:, target_band_index]
        sorted_order = np.argsort(energy_difference)
        sorted_difference = energy_difference[sorted_order]

        if abs(sorted_difference[0]) < 1.0e-8 and abs(sorted_difference[3]) < 1.0e-8:
            weights[target_band_index] = 0.125
            continue

        sorted_weights = np.zeros(4, dtype=np.float64)
        if (sorted_difference[0] <= 0.0 < sorted_difference[1]) or (sorted_difference[0] < 0.0 <= sorted_difference[1]):
            _accumulate_small_tetra_step(sorted_weights, "a1", sorted_difference)
        elif (sorted_difference[1] <= 0.0 < sorted_difference[2]) or (sorted_difference[1] < 0.0 <= sorted_difference[2]):
            for kind in ("b1", "b2", "b3"):
                _accumulate_small_tetra_step(sorted_weights, kind, sorted_difference)
        elif (sorted_difference[2] <= 0.0 < sorted_difference[3]) or (sorted_difference[2] < 0.0 <= sorted_difference[3]):
            for kind in ("c1", "c2", "c3"):
                _accumulate_small_tetra_step(sorted_weights, kind, sorted_difference)
        elif sorted_difference[3] <= 0.0:
            sorted_weights[:] = 0.25

        weights[target_band_index, sorted_order] = sorted_weights

    return weights


def _double_delta_secondary_weights(triangle_vertices: FloatArray) -> FloatArray:
    target_band_count = triangle_vertices.shape[1]
    weights = np.zeros((target_band_count, 3), dtype=np.float64)

    for target_band_index in range(target_band_count):
        energies = triangle_vertices[:, target_band_index]
        if float(np.max(np.abs(energies))) < 1.0e-10:
            raise RuntimeError("encountered nesting condition in nesting_function_weights")

        sorted_order = np.argsort(energies)
        sorted_energies = _normalize_sorted_triangle_energies(energies[sorted_order])
        affine = _triangle_affine_coefficients(sorted_energies)
        sorted_weights = np.zeros(3, dtype=np.float64)

        if (sorted_energies[0] < 0.0 <= sorted_energies[1]) or (sorted_energies[0] <= 0.0 < sorted_energies[1]):
            volume_factor = affine[1, 0] / (sorted_energies[2] - sorted_energies[0])
            sorted_weights[0] = volume_factor * (affine[0, 1] + affine[0, 2])
            sorted_weights[1] = volume_factor * affine[1, 0]
            sorted_weights[2] = volume_factor * affine[2, 0]
        elif (sorted_energies[1] <= 0.0 < sorted_energies[2]) or (sorted_energies[1] < 0.0 <= sorted_energies[2]):
            volume_factor = affine[1, 2] / (sorted_energies[2] - sorted_energies[0])
            sorted_weights[0] = volume_factor * affine[0, 2]
            sorted_weights[1] = volume_factor * affine[1, 2]
            sorted_weights[2] = volume_factor * (affine[2, 0] + affine[2, 1])

        weights[target_band_index, sorted_order] = sorted_weights

    return weights


def _polstat_secondary_weights(occupied_vertices: FloatArray, target_vertices: FloatArray) -> FloatArray:
    target_band_count = target_vertices.shape[1]
    weights = np.zeros((target_band_count, 4), dtype=np.float64)

    for target_band_index in range(target_band_count):
        sorted_order = np.argsort(-target_vertices[:, target_band_index])
        sorted_step_energies = -target_vertices[sorted_order, target_band_index]
        sorted_target = target_vertices[sorted_order, target_band_index]
        sorted_occupied = occupied_vertices[sorted_order]
        sorted_weights = np.zeros(4, dtype=np.float64)

        if ((sorted_step_energies[0] <= 0.0 < sorted_step_energies[1]) or
                (sorted_step_energies[0] < 0.0 <= sorted_step_energies[1])):
            _accumulate_small_tetra_polstat_inner(sorted_weights, "a1", sorted_step_energies, sorted_occupied, sorted_target)
        elif ((sorted_step_energies[1] <= 0.0 < sorted_step_energies[2]) or
                (sorted_step_energies[1] < 0.0 <= sorted_step_energies[2])):
            for kind in ("b1", "b2", "b3"):
                _accumulate_small_tetra_polstat_inner(
                    sorted_weights,
                    kind,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                )
        elif ((sorted_step_energies[2] <= 0.0 < sorted_step_energies[3]) or
                (sorted_step_energies[2] < 0.0 <= sorted_step_energies[3])):
            for kind in ("c1", "c2", "c3"):
                _accumulate_small_tetra_polstat_inner(
                    sorted_weights,
                    kind,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                )
        elif sorted_step_energies[3] <= 0.0:
            sorted_weights += _polstat_logarithmic_weights(sorted_target - sorted_occupied)

        weights[target_band_index, sorted_order] = sorted_weights

    return weights


def _accumulate_small_tetra_polstat_inner(
    weights: FloatArray,
    kind: str,
    sorted_step_energies: FloatArray,
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
) -> None:
    cut = small_tetrahedron_cut(kind, sorted_step_energies)
    if cut.volume_factor <= 1.0e-8:
        return

    energy_differences = cut.coefficients @ (sorted_target - sorted_occupied)
    weights += cut.volume_factor * (_polstat_logarithmic_weights(energy_differences) @ cut.coefficients)


def _polstat_logarithmic_weights(energy_differences: FloatArray) -> FloatArray:
    sorted_order = np.argsort(energy_differences)
    sorted_differences = np.asarray(energy_differences, dtype=np.float64)[sorted_order].copy()
    logarithms = np.empty(4, dtype=np.float64)
    threshold = float(np.max(sorted_differences)) * 1.0e-3
    absolute_floor = 1.0e-8

    for index in range(4):
        if sorted_differences[index] < absolute_floor:
            if index == 2:
                raise RuntimeError("encountered nesting condition in static_polarization_weights")
            logarithms[index] = 0.0
            sorted_differences[index] = 0.0
        else:
            logarithms[index] = float(np.log(sorted_differences[index]))

    sorted_weights = np.zeros(4, dtype=np.float64)

    if abs(sorted_differences[3] - sorted_differences[2]) < threshold:
        if abs(sorted_differences[3] - sorted_differences[1]) < threshold:
            if abs(sorted_differences[3] - sorted_differences[0]) < threshold:
                sorted_weights[:] = 0.25 / sorted_differences[3]
            else:
                sorted_weights[3] = _polstat_1211(sorted_differences[3], sorted_differences[0], logarithms[3], logarithms[0])
                sorted_weights[2] = sorted_weights[3]
                sorted_weights[1] = sorted_weights[3]
                sorted_weights[0] = _polstat_1222(sorted_differences[0], sorted_differences[3], logarithms[0], logarithms[3])
                _check_polstat_weights(sorted_weights, "4=3=2")
        elif abs(sorted_differences[1] - sorted_differences[0]) < threshold:
            sorted_weights[3] = _polstat_1221(sorted_differences[3], sorted_differences[1], logarithms[3], logarithms[1])
            sorted_weights[2] = sorted_weights[3]
            sorted_weights[1] = _polstat_1221(sorted_differences[1], sorted_differences[3], logarithms[1], logarithms[3])
            sorted_weights[0] = sorted_weights[1]
            _check_polstat_weights(sorted_weights, "4=3 2=1")
        else:
            sorted_weights[3] = _polstat_1231(
                sorted_differences[3],
                sorted_differences[0],
                sorted_differences[1],
                logarithms[3],
                logarithms[0],
                logarithms[1],
            )
            sorted_weights[2] = sorted_weights[3]
            sorted_weights[1] = _polstat_1233(
                sorted_differences[1],
                sorted_differences[0],
                sorted_differences[3],
                logarithms[1],
                logarithms[0],
                logarithms[3],
            )
            sorted_weights[0] = _polstat_1233(
                sorted_differences[0],
                sorted_differences[1],
                sorted_differences[3],
                logarithms[0],
                logarithms[1],
                logarithms[3],
            )
            _check_polstat_weights(sorted_weights, "4=3")
    elif abs(sorted_differences[2] - sorted_differences[1]) < threshold:
        if abs(sorted_differences[2] - sorted_differences[0]) < threshold:
            sorted_weights[3] = _polstat_1222(sorted_differences[3], sorted_differences[2], logarithms[3], logarithms[2])
            sorted_weights[2] = _polstat_1211(sorted_differences[2], sorted_differences[3], logarithms[2], logarithms[3])
            sorted_weights[1] = sorted_weights[2]
            sorted_weights[0] = sorted_weights[2]
            _check_polstat_weights(sorted_weights, "3=2=1")
        else:
            sorted_weights[3] = _polstat_1233(
                sorted_differences[3],
                sorted_differences[0],
                sorted_differences[2],
                logarithms[3],
                logarithms[0],
                logarithms[2],
            )
            sorted_weights[2] = _polstat_1231(
                sorted_differences[2],
                sorted_differences[0],
                sorted_differences[3],
                logarithms[2],
                logarithms[0],
                logarithms[3],
            )
            sorted_weights[1] = sorted_weights[2]
            sorted_weights[0] = _polstat_1233(
                sorted_differences[0],
                sorted_differences[3],
                sorted_differences[2],
                logarithms[0],
                logarithms[3],
                logarithms[2],
            )
            _check_polstat_weights(sorted_weights, "3=2")
    elif abs(sorted_differences[1] - sorted_differences[0]) < threshold:
        sorted_weights[3] = _polstat_1233(
            sorted_differences[3],
            sorted_differences[2],
            sorted_differences[1],
            logarithms[3],
            logarithms[2],
            logarithms[1],
        )
        sorted_weights[2] = _polstat_1233(
            sorted_differences[2],
            sorted_differences[3],
            sorted_differences[1],
            logarithms[2],
            logarithms[3],
            logarithms[1],
        )
        sorted_weights[1] = _polstat_1231(
            sorted_differences[1],
            sorted_differences[2],
            sorted_differences[3],
            logarithms[1],
            logarithms[2],
            logarithms[3],
        )
        sorted_weights[0] = sorted_weights[1]
        _check_polstat_weights(sorted_weights, "2=1")
    else:
        sorted_weights[3] = _polstat_1234(
            sorted_differences[3],
            sorted_differences[0],
            sorted_differences[1],
            sorted_differences[2],
            logarithms[3],
            logarithms[0],
            logarithms[1],
            logarithms[2],
        )
        sorted_weights[2] = _polstat_1234(
            sorted_differences[2],
            sorted_differences[0],
            sorted_differences[1],
            sorted_differences[3],
            logarithms[2],
            logarithms[0],
            logarithms[1],
            logarithms[3],
        )
        sorted_weights[1] = _polstat_1234(
            sorted_differences[1],
            sorted_differences[0],
            sorted_differences[2],
            sorted_differences[3],
            logarithms[1],
            logarithms[0],
            logarithms[2],
            logarithms[3],
        )
        sorted_weights[0] = _polstat_1234(
            sorted_differences[0],
            sorted_differences[1],
            sorted_differences[2],
            sorted_differences[3],
            logarithms[0],
            logarithms[1],
            logarithms[2],
            logarithms[3],
        )
        _check_polstat_weights(sorted_weights, "general")

    weights = np.zeros(4, dtype=np.float64)
    weights[sorted_order] = sorted_weights
    return weights


def _polstat_1234(g1: float, g2: float, g3: float, g4: float, log1: float, log2: float, log3: float, log4: float) -> float:
    weight_2 = (((log2 - log1) / (g2 - g1) * g2) - 1.0) * g2 / (g2 - g1)
    weight_3 = (((log3 - log1) / (g3 - g1) * g3) - 1.0) * g3 / (g3 - g1)
    weight_4 = (((log4 - log1) / (g4 - g1) * g4) - 1.0) * g4 / (g4 - g1)
    weight_2 = ((weight_2 - weight_3) * g2) / (g2 - g3)
    weight_4 = ((weight_4 - weight_3) * g4) / (g4 - g3)
    return (weight_4 - weight_2) / (g4 - g2)


def _polstat_1231(g1: float, g2: float, g3: float, log1: float, log2: float, log3: float) -> float:
    weight_2 = ((((log2 - log1) / (g2 - g1) * g2) - 1.0) * g2 ** 2 / (g2 - g1) - g1 / 2.0) / (g2 - g1)
    weight_3 = ((((log3 - log1) / (g3 - g1) * g3) - 1.0) * g3 ** 2 / (g3 - g1) - g1 / 2.0) / (g3 - g1)
    return (weight_3 - weight_2) / (g3 - g2)


def _polstat_1233(g1: float, g2: float, g3: float, log1: float, log2: float, log3: float) -> float:
    weight_2 = (log2 - log1) / (g2 - g1) * g2 - 1.0
    weight_2 = (g2 * weight_2) / (g2 - g1)
    weight_3 = (log3 - log1) / (g3 - g1) * g3 - 1.0
    weight_3 = (g3 * weight_3) / (g3 - g1)
    weight_2 = (weight_3 - weight_2) / (g3 - g2)
    weight_3 = (log3 - log1) / (g3 - g1) * g3 - 1.0
    weight_3 = 1.0 - (2.0 * weight_3 * g1) / (g3 - g1)
    weight_3 = weight_3 / (g3 - g1)
    return (g3 * weight_3 - g2 * weight_2) / (g3 - g2)


def _polstat_1221(g1: float, g2: float, log1: float, log2: float) -> float:
    weight = 1.0 - (log2 - log1) / (g2 - g1) * g1
    weight = -1.0 + (2.0 * g2 * weight) / (g2 - g1)
    weight = -1.0 + (3.0 * g2 * weight) / (g2 - g1)
    return weight / (2.0 * (g2 - g1))


def _polstat_1222(g1: float, g2: float, log1: float, log2: float) -> float:
    weight = (log2 - log1) / (g2 - g1) * g2 - 1.0
    weight = (2.0 * g1 * weight) / (g2 - g1) - 1.0
    weight = (3.0 * g1 * weight) / (g2 - g1) + 1.0
    return weight / (2.0 * (g2 - g1))


def _polstat_1211(g1: float, g2: float, log1: float, log2: float) -> float:
    weight = -1.0 + (log2 - log1) / (g2 - g1) * g2
    weight = -1.0 + (2.0 * g2 * weight) / (g2 - g1)
    weight = -1.0 + (3.0 * g2 * weight) / (2.0 * (g2 - g1))
    return weight / (3.0 * (g2 - g1))


def _check_polstat_weights(weights: FloatArray, label: str) -> None:
    if np.any(weights < 0.0):
        raise RuntimeError(f"negative static_polarization_weights values encountered in {label}")


@njit(cache=True)
def _double_step_weights_on_local_mesh_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    local_point_count: int,
    normalization: int,
) -> FloatArray:
    tetrahedron_count = occupied_tetra.shape[0]
    occupied_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    local_weights = np.zeros((local_point_count, target_band_count, occupied_band_count), dtype=np.float64)

    sorted_order = np.empty(4, dtype=np.int64)
    identity_order = np.empty(4, dtype=np.int64)
    for index in range(4):
        identity_order[index] = index

    sorted_occupied = np.empty(4, dtype=np.float64)
    sorted_target = np.empty((4, target_band_count), dtype=np.float64)
    outer_weights = np.empty((target_band_count, 4), dtype=np.float64)
    point_weights = np.empty(20, dtype=np.float64)

    outer_strict = np.empty(4, dtype=np.float64)
    outer_affine = np.empty((4, 4), dtype=np.float64)
    outer_coefficients = np.empty((4, 4), dtype=np.float64)
    transformed_occupied = np.empty(4, dtype=np.float64)
    transformed_target = np.empty((4, target_band_count), dtype=np.float64)

    secondary_weights = np.empty((target_band_count, 4), dtype=np.float64)
    difference = np.empty(4, dtype=np.float64)
    secondary_sorted_order = np.empty(4, dtype=np.int64)
    secondary_sorted_difference = np.empty(4, dtype=np.float64)
    secondary_sorted_weights = np.empty(4, dtype=np.float64)
    secondary_strict = np.empty(4, dtype=np.float64)
    secondary_affine = np.empty((4, 4), dtype=np.float64)
    secondary_coefficients = np.empty((4, 4), dtype=np.float64)

    for tetrahedron_index in range(tetrahedron_count):
        for occupied_band_index in range(occupied_band_count):
            sort4(
                occupied_tetra[tetrahedron_index, :, occupied_band_index],
                sorted_order,
                sorted_occupied,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                for target_band_index in range(target_band_count):
                    sorted_target[vertex_index, target_band_index] = target_tetra[
                        tetrahedron_index,
                        source_vertex,
                        target_band_index,
                    ]

            outer_weights[:, :] = 0.0
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_dblstep_outer(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    target_band_count,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_occupied,
                    transformed_target,
                    secondary_weights,
                    difference,
                    secondary_sorted_order,
                    secondary_sorted_difference,
                    secondary_sorted_weights,
                    secondary_strict,
                    secondary_affine,
                    secondary_coefficients,
                    identity_order,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for case_id in (1, 2, 3):
                    _accumulate_small_tetra_dblstep_outer(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        difference,
                        secondary_sorted_order,
                        secondary_sorted_difference,
                        secondary_sorted_weights,
                        secondary_strict,
                        secondary_affine,
                        secondary_coefficients,
                        identity_order,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for case_id in (4, 5, 6):
                    _accumulate_small_tetra_dblstep_outer(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        difference,
                        secondary_sorted_order,
                        secondary_sorted_difference,
                        secondary_sorted_weights,
                        secondary_strict,
                        secondary_affine,
                        secondary_coefficients,
                        identity_order,
                    )
            elif sorted_occupied[3] <= 0.0:
                _double_step_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, occupied_band_index],
                    target_tetra[tetrahedron_index],
                    secondary_weights,
                    difference,
                    secondary_sorted_order,
                    secondary_sorted_difference,
                    secondary_sorted_weights,
                    secondary_strict,
                    secondary_affine,
                    secondary_coefficients,
                    identity_order,
                )
                for target_band_index in range(target_band_count):
                    for vertex_index in range(4):
                        outer_weights[target_band_index, vertex_index] += secondary_weights[target_band_index, vertex_index]

            for target_band_index in range(target_band_count):
                for point_index in range(20):
                    total = 0.0
                    for vertex_index in range(4):
                        total += outer_weights[target_band_index, vertex_index] * tetrahedron_weight_matrix[vertex_index, point_index]
                    point_weights[point_index] = total
                for point_index in range(20):
                    local_weights[
                        local_point_indices[tetrahedron_index, point_index],
                        target_band_index,
                        occupied_band_index,
                    ] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True)
def _accumulate_small_tetra_dblstep_outer(
    outer_weights: FloatArray,
    case_id: int,
    sorted_order: npt.NDArray[np.int64],
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    target_band_count: int,
    outer_strict: FloatArray,
    outer_affine: FloatArray,
    outer_coefficients: FloatArray,
    transformed_occupied: FloatArray,
    transformed_target: FloatArray,
    secondary_weights: FloatArray,
    difference: FloatArray,
    secondary_sorted_order: npt.NDArray[np.int64],
    secondary_sorted_difference: FloatArray,
    secondary_sorted_weights: FloatArray,
    secondary_strict: FloatArray,
    secondary_affine: FloatArray,
    secondary_coefficients: FloatArray,
    identity_order: npt.NDArray[np.int64],
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_occupied,
        outer_strict,
        outer_affine,
        outer_coefficients,
    )

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += outer_coefficients[row_index, column_index] * sorted_occupied[column_index]
        transformed_occupied[row_index] = total

    for row_index in range(4):
        for target_band_index in range(target_band_count):
            total = 0.0
            for column_index in range(4):
                total += outer_coefficients[row_index, column_index] * sorted_target[column_index, target_band_index]
            transformed_target[row_index, target_band_index] = total

    _double_step_secondary_weights_numba(
        transformed_occupied,
        transformed_target,
        secondary_weights,
        difference,
        secondary_sorted_order,
        secondary_sorted_difference,
        secondary_sorted_weights,
        secondary_strict,
        secondary_affine,
        secondary_coefficients,
        identity_order,
    )

    for target_band_index in range(target_band_count):
        for column_index in range(4):
            total = 0.0
            for row_index in range(4):
                total += secondary_weights[target_band_index, row_index] * outer_coefficients[row_index, column_index]
            outer_weights[target_band_index, sorted_order[column_index]] += volume_factor * total


@njit(cache=True)
def _double_step_secondary_weights_numba(
    occupied_vertices: FloatArray,
    target_vertices: FloatArray,
    weights: FloatArray,
    difference: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_difference: FloatArray,
    sorted_weights: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    identity_order: npt.NDArray[np.int64],
) -> None:
    target_band_count = target_vertices.shape[1]
    weights[:, :] = 0.0

    for target_band_index in range(target_band_count):
        for vertex_index in range(4):
            difference[vertex_index] = -occupied_vertices[vertex_index] + target_vertices[vertex_index, target_band_index]
        sort4(difference, sorted_order, sorted_difference)

        if abs(sorted_difference[0]) < 1.0e-8 and abs(sorted_difference[3]) < 1.0e-8:
            for vertex_index in range(4):
                weights[target_band_index, vertex_index] = 0.125
            continue

        sorted_weights[:] = 0.0
        if ((sorted_difference[0] <= 0.0 < sorted_difference[1]) or
                (sorted_difference[0] < 0.0 <= sorted_difference[1])):
            _accumulate_small_tetra_step_numba(
                sorted_weights,
                0,
                sorted_difference,
                strict_energies,
                affine,
                coefficients,
                identity_order,
            )
        elif ((sorted_difference[1] <= 0.0 < sorted_difference[2]) or
                (sorted_difference[1] < 0.0 <= sorted_difference[2])):
            for case_id in (1, 2, 3):
                _accumulate_small_tetra_step_numba(
                    sorted_weights,
                    case_id,
                    sorted_difference,
                    strict_energies,
                    affine,
                    coefficients,
                    identity_order,
                )
        elif ((sorted_difference[2] <= 0.0 < sorted_difference[3]) or
                (sorted_difference[2] < 0.0 <= sorted_difference[3])):
            for case_id in (4, 5, 6):
                _accumulate_small_tetra_step_numba(
                    sorted_weights,
                    case_id,
                    sorted_difference,
                    strict_energies,
                    affine,
                    coefficients,
                    identity_order,
                )
        elif sorted_difference[3] <= 0.0:
            sorted_weights[:] = 0.25

        for vertex_index in range(4):
            weights[target_band_index, sorted_order[vertex_index]] = sorted_weights[vertex_index]


@njit(cache=True)
def _accumulate_small_tetra_step_numba(
    weights: FloatArray,
    case_id: int,
    sorted_energies: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    identity_order: npt.NDArray[np.int64],
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_energies,
        strict_energies,
        affine,
        coefficients,
    )
    for column_index in range(4):
        column_sum = 0.0
        for row_index in range(4):
            column_sum += coefficients[row_index, column_index]
        weights[identity_order[column_index]] += volume_factor * column_sum * 0.25


@njit(cache=True)
def _double_delta_weights_on_local_mesh_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    source_tetra: FloatArray,
    target_tetra: FloatArray,
    local_point_count: int,
    normalization: int,
) -> FloatArray:
    tetrahedron_count = source_tetra.shape[0]
    source_band_count = source_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    local_weights = np.zeros((local_point_count, target_band_count, source_band_count), dtype=np.float64)

    sorted_order = np.empty(4, dtype=np.int64)
    sorted_source = np.empty(4, dtype=np.float64)
    sorted_target = np.empty((4, target_band_count), dtype=np.float64)
    outer_weights = np.empty((target_band_count, 4), dtype=np.float64)
    point_weights = np.empty(20, dtype=np.float64)

    outer_strict = np.empty(4, dtype=np.float64)
    outer_affine = np.empty((4, 4), dtype=np.float64)
    outer_coefficients = np.empty((3, 4), dtype=np.float64)
    transformed_target = np.empty((3, target_band_count), dtype=np.float64)

    secondary_weights = np.empty((target_band_count, 3), dtype=np.float64)
    secondary_sorted_order = np.empty(3, dtype=np.int64)
    secondary_sorted_energies = np.empty(3, dtype=np.float64)
    secondary_sorted_weights = np.empty(3, dtype=np.float64)
    secondary_strict = np.empty(3, dtype=np.float64)
    secondary_affine = np.empty((3, 3), dtype=np.float64)

    for tetrahedron_index in range(tetrahedron_count):
        for source_band_index in range(source_band_count):
            sort4(
                source_tetra[tetrahedron_index, :, source_band_index],
                sorted_order,
                sorted_source,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                for target_band_index in range(target_band_count):
                    sorted_target[vertex_index, target_band_index] = target_tetra[
                        tetrahedron_index,
                        source_vertex,
                        target_band_index,
                    ]

            outer_weights[:, :] = 0.0
            if sorted_source[0] < 0.0 <= sorted_source[1]:
                _accumulate_triangle_dbldelta_outer(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_source,
                    sorted_target,
                    target_band_count,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_target,
                    secondary_weights,
                    secondary_sorted_order,
                    secondary_sorted_energies,
                    secondary_sorted_weights,
                    secondary_strict,
                    secondary_affine,
                )
            elif sorted_source[1] < 0.0 <= sorted_source[2]:
                for case_id in (1, 2):
                    _accumulate_triangle_dbldelta_outer(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_source,
                        sorted_target,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_target,
                        secondary_weights,
                        secondary_sorted_order,
                        secondary_sorted_energies,
                        secondary_sorted_weights,
                        secondary_strict,
                        secondary_affine,
                    )
            elif sorted_source[2] < 0.0 < sorted_source[3]:
                _accumulate_triangle_dbldelta_outer(
                    outer_weights,
                    3,
                    sorted_order,
                    sorted_source,
                    sorted_target,
                    target_band_count,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_target,
                    secondary_weights,
                    secondary_sorted_order,
                    secondary_sorted_energies,
                    secondary_sorted_weights,
                    secondary_strict,
                    secondary_affine,
                )

            for target_band_index in range(target_band_count):
                for point_index in range(20):
                    total = 0.0
                    for vertex_index in range(4):
                        total += outer_weights[target_band_index, vertex_index] * tetrahedron_weight_matrix[vertex_index, point_index]
                    point_weights[point_index] = total
                for point_index in range(20):
                    local_weights[
                        local_point_indices[tetrahedron_index, point_index],
                        target_band_index,
                        source_band_index,
                    ] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True)
def _accumulate_triangle_dbldelta_outer(
    outer_weights: FloatArray,
    case_id: int,
    sorted_order: npt.NDArray[np.int64],
    sorted_source: FloatArray,
    sorted_target: FloatArray,
    target_band_count: int,
    outer_strict: FloatArray,
    outer_affine: FloatArray,
    outer_coefficients: FloatArray,
    transformed_target: FloatArray,
    secondary_weights: FloatArray,
    secondary_sorted_order: npt.NDArray[np.int64],
    secondary_sorted_energies: FloatArray,
    secondary_sorted_weights: FloatArray,
    secondary_strict: FloatArray,
    secondary_affine: FloatArray,
) -> None:
    volume_factor = triangle_volume_and_coefficients(
        case_id,
        sorted_source,
        outer_strict,
        outer_affine,
        outer_coefficients,
    )

    for row_index in range(3):
        for target_band_index in range(target_band_count):
            total = 0.0
            for column_index in range(4):
                total += outer_coefficients[row_index, column_index] * sorted_target[column_index, target_band_index]
            transformed_target[row_index, target_band_index] = total

    _double_delta_secondary_weights_numba(
        transformed_target,
        secondary_weights,
        secondary_sorted_order,
        secondary_sorted_energies,
        secondary_sorted_weights,
        secondary_strict,
        secondary_affine,
    )

    for target_band_index in range(target_band_count):
        for column_index in range(4):
            total = 0.0
            for row_index in range(3):
                total += secondary_weights[target_band_index, row_index] * outer_coefficients[row_index, column_index]
            outer_weights[target_band_index, sorted_order[column_index]] += volume_factor * total


@njit(cache=True)
def _double_delta_secondary_weights_numba(
    triangle_vertices: FloatArray,
    weights: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_energies: FloatArray,
    sorted_weights: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
) -> None:
    target_band_count = triangle_vertices.shape[1]
    weights[:, :] = 0.0

    for target_band_index in range(target_band_count):
        energies = triangle_vertices[:, target_band_index]
        max_abs_energy = 0.0
        for vertex_index in range(3):
            value = abs(energies[vertex_index])
            if value > max_abs_energy:
                max_abs_energy = value
        if max_abs_energy < 1.0e-10:
            raise RuntimeError("encountered nesting condition in nesting_function_weights")

        _sort3(energies, sorted_order, sorted_energies)
        _strict_sorted_energies3(sorted_energies, strict_energies)
        _fill_triangle_affine3(strict_energies, affine)
        sorted_weights[:] = 0.0

        if ((strict_energies[0] < 0.0 <= strict_energies[1]) or
                (strict_energies[0] <= 0.0 < strict_energies[1])):
            volume_factor = affine[1, 0] / (strict_energies[2] - strict_energies[0])
            sorted_weights[0] = volume_factor * (affine[0, 1] + affine[0, 2])
            sorted_weights[1] = volume_factor * affine[1, 0]
            sorted_weights[2] = volume_factor * affine[2, 0]
        elif ((strict_energies[1] <= 0.0 < strict_energies[2]) or
                (strict_energies[1] < 0.0 <= strict_energies[2])):
            volume_factor = affine[1, 2] / (strict_energies[2] - strict_energies[0])
            sorted_weights[0] = volume_factor * affine[0, 2]
            sorted_weights[1] = volume_factor * affine[1, 2]
            sorted_weights[2] = volume_factor * (affine[2, 0] + affine[2, 1])

        for vertex_index in range(3):
            weights[target_band_index, sorted_order[vertex_index]] = sorted_weights[vertex_index]


@njit(cache=True)
def _sort3(
    values: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_values: FloatArray,
) -> None:
    for index in range(3):
        sorted_order[index] = index
        sorted_values[index] = values[index]

    for index in range(1, 3):
        key_value = sorted_values[index]
        key_order = sorted_order[index]
        scan = index - 1
        while scan >= 0 and sorted_values[scan] > key_value:
            sorted_values[scan + 1] = sorted_values[scan]
            sorted_order[scan + 1] = sorted_order[scan]
            scan -= 1
        sorted_values[scan + 1] = key_value
        sorted_order[scan + 1] = key_order


@njit(cache=True)
def _strict_sorted_energies3(sorted_energies: FloatArray, adjusted_energies: FloatArray) -> None:
    for index in range(3):
        adjusted_energies[index] = sorted_energies[index]

    for index in range(1, 3):
        if adjusted_energies[index] <= adjusted_energies[index - 1]:
            adjusted_energies[index] = np.nextafter(adjusted_energies[index - 1], np.inf)


@njit(cache=True)
def _fill_triangle_affine3(energies: FloatArray, affine: FloatArray) -> None:
    affine[:, :] = 0.0
    for column in range(3):
        energy = energies[column]
        for row in range(3):
            if row != column:
                affine[row, column] = -energy / (energies[row] - energy)


@njit(cache=True, parallel=True)
def _fermi_golden_rule_weights_on_local_mesh_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: FloatArray,
    local_point_count: int,
    normalization: int,
) -> FloatArray:
    tetrahedron_count = occupied_tetra.shape[0]
    source_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    energy_count = sample_energies.shape[0]
    local_weights = np.zeros((local_point_count, energy_count, target_band_count, source_band_count), dtype=np.float64)

    for source_band_index in prange(source_band_count):
        sorted_order = np.empty(4, dtype=np.int64)
        sorted_occupied = np.empty(4, dtype=np.float64)
        sorted_target = np.empty((4, target_band_count), dtype=np.float64)
        outer_weights = np.empty((energy_count, target_band_count, 4), dtype=np.float64)
        point_weights = np.empty(20, dtype=np.float64)

        outer_strict = np.empty(4, dtype=np.float64)
        outer_affine = np.empty((4, 4), dtype=np.float64)
        outer_coefficients = np.empty((4, 4), dtype=np.float64)
        transformed_occupied = np.empty(4, dtype=np.float64)
        transformed_target = np.empty((4, target_band_count), dtype=np.float64)

        secondary_weights = np.empty((energy_count, target_band_count, 4), dtype=np.float64)
        step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_order = np.empty(4, dtype=np.int64)
        secondary_sorted_step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_target = np.empty(4, dtype=np.float64)
        secondary_sorted_occupied = np.empty(4, dtype=np.float64)
        secondary_sorted_weights = np.empty((energy_count, 4), dtype=np.float64)
        inner_strict = np.empty(4, dtype=np.float64)
        inner_affine = np.empty((4, 4), dtype=np.float64)
        inner_coefficients = np.empty((4, 4), dtype=np.float64)
        energy_differences = np.empty(4, dtype=np.float64)
        delta_weights = np.empty((energy_count, 4), dtype=np.float64)
        delta_sorted_order = np.empty(4, dtype=np.int64)
        delta_sorted_energies = np.empty(4, dtype=np.float64)
        delta_shifted_energies = np.empty(4, dtype=np.float64)
        triangle_strict = np.empty(4, dtype=np.float64)
        triangle_affine = np.empty((4, 4), dtype=np.float64)
        triangle_coefficients = np.empty((3, 4), dtype=np.float64)

        for tetrahedron_index in range(tetrahedron_count):
            sort4(
                occupied_tetra[tetrahedron_index, :, source_band_index],
                sorted_order,
                sorted_occupied,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                for target_band_index in range(target_band_count):
                    sorted_target[vertex_index, target_band_index] = target_tetra[
                        tetrahedron_index,
                        source_vertex,
                        target_band_index,
                    ]

            outer_weights[:, :, :] = 0.0
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_fermigr_outer_numba(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    target_band_count,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_occupied,
                    transformed_target,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for case_id in (1, 2, 3):
                    _accumulate_small_tetra_fermigr_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        delta_weights,
                        delta_sorted_order,
                        delta_sorted_energies,
                        delta_shifted_energies,
                        triangle_strict,
                        triangle_affine,
                        triangle_coefficients,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for case_id in (4, 5, 6):
                    _accumulate_small_tetra_fermigr_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        delta_weights,
                        delta_sorted_order,
                        delta_sorted_energies,
                        delta_shifted_energies,
                        triangle_strict,
                        triangle_affine,
                        triangle_coefficients,
                    )
            elif sorted_occupied[3] <= 0.0:
                _fermigr_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    target_tetra[tetrahedron_index],
                    sample_energies,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                )
                for energy_index in range(energy_count):
                    for target_band_index in range(target_band_count):
                        for vertex_index in range(4):
                            outer_weights[energy_index, target_band_index, vertex_index] += secondary_weights[
                                energy_index,
                                target_band_index,
                                vertex_index,
                            ]

            for energy_index in range(energy_count):
                for target_band_index in range(target_band_count):
                    for point_index in range(20):
                        total = 0.0
                        for vertex_index in range(4):
                            total += outer_weights[energy_index, target_band_index, vertex_index] * tetrahedron_weight_matrix[
                                vertex_index,
                                point_index,
                            ]
                        point_weights[point_index] = total
                    for point_index in range(20):
                        local_weights[
                            local_point_indices[tetrahedron_index, point_index],
                            energy_index,
                            target_band_index,
                            source_band_index,
                        ] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True, parallel=True)
def _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: FloatArray,
    local_point_count: int,
    normalization: int,
) -> FloatArray:
    tetrahedron_count = occupied_tetra.shape[0]
    source_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    energy_count = sample_energies.shape[0]
    pair_count = source_band_count * target_band_count
    local_weights = np.zeros((local_point_count, energy_count, target_band_count, source_band_count), dtype=np.float64)

    for pair_index in prange(pair_count):
        source_band_index = pair_index // target_band_count
        target_band_index = pair_index % target_band_count

        sorted_order = np.empty(4, dtype=np.int64)
        sorted_occupied = np.empty(4, dtype=np.float64)
        sorted_target = np.empty((4, 1), dtype=np.float64)
        full_target = np.empty((4, 1), dtype=np.float64)
        outer_weights = np.empty((energy_count, 1, 4), dtype=np.float64)
        point_weights = np.empty(20, dtype=np.float64)

        outer_strict = np.empty(4, dtype=np.float64)
        outer_affine = np.empty((4, 4), dtype=np.float64)
        outer_coefficients = np.empty((4, 4), dtype=np.float64)
        transformed_occupied = np.empty(4, dtype=np.float64)
        transformed_target = np.empty((4, 1), dtype=np.float64)

        secondary_weights = np.empty((energy_count, 1, 4), dtype=np.float64)
        step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_order = np.empty(4, dtype=np.int64)
        secondary_sorted_step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_target = np.empty(4, dtype=np.float64)
        secondary_sorted_occupied = np.empty(4, dtype=np.float64)
        secondary_sorted_weights = np.empty((energy_count, 4), dtype=np.float64)
        inner_strict = np.empty(4, dtype=np.float64)
        inner_affine = np.empty((4, 4), dtype=np.float64)
        inner_coefficients = np.empty((4, 4), dtype=np.float64)
        energy_differences = np.empty(4, dtype=np.float64)
        delta_weights = np.empty((energy_count, 4), dtype=np.float64)
        delta_sorted_order = np.empty(4, dtype=np.int64)
        delta_sorted_energies = np.empty(4, dtype=np.float64)
        delta_shifted_energies = np.empty(4, dtype=np.float64)
        triangle_strict = np.empty(4, dtype=np.float64)
        triangle_affine = np.empty((4, 4), dtype=np.float64)
        triangle_coefficients = np.empty((3, 4), dtype=np.float64)

        for tetrahedron_index in range(tetrahedron_count):
            sort4(
                occupied_tetra[tetrahedron_index, :, source_band_index],
                sorted_order,
                sorted_occupied,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                sorted_target[vertex_index, 0] = target_tetra[
                    tetrahedron_index,
                    source_vertex,
                    target_band_index,
                ]
                full_target[vertex_index, 0] = target_tetra[
                    tetrahedron_index,
                    vertex_index,
                    target_band_index,
                ]

            outer_weights[:, :, :] = 0.0
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_fermigr_outer_numba(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    1,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_occupied,
                    transformed_target,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for case_id in (1, 2, 3):
                    _accumulate_small_tetra_fermigr_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        1,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        delta_weights,
                        delta_sorted_order,
                        delta_sorted_energies,
                        delta_shifted_energies,
                        triangle_strict,
                        triangle_affine,
                        triangle_coefficients,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for case_id in (4, 5, 6):
                    _accumulate_small_tetra_fermigr_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        1,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        delta_weights,
                        delta_sorted_order,
                        delta_sorted_energies,
                        delta_shifted_energies,
                        triangle_strict,
                        triangle_affine,
                        triangle_coefficients,
                    )
            elif sorted_occupied[3] <= 0.0:
                _fermigr_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    full_target,
                    sample_energies,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                )
                for energy_index in range(energy_count):
                    for vertex_index in range(4):
                        outer_weights[energy_index, 0, vertex_index] += secondary_weights[
                            energy_index,
                            0,
                            vertex_index,
                        ]

            for energy_index in range(energy_count):
                for point_index in range(20):
                    total = 0.0
                    for vertex_index in range(4):
                        total += outer_weights[energy_index, 0, vertex_index] * tetrahedron_weight_matrix[
                            vertex_index,
                            point_index,
                        ]
                    point_weights[point_index] = total
                for point_index in range(20):
                    local_weights[
                        local_point_indices[tetrahedron_index, point_index],
                        energy_index,
                        target_band_index,
                        source_band_index,
                    ] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True)
def _accumulate_small_tetra_fermigr_outer_numba(
    outer_weights: FloatArray,
    case_id: int,
    sorted_order: npt.NDArray[np.int64],
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energies: FloatArray,
    target_band_count: int,
    outer_strict: FloatArray,
    outer_affine: FloatArray,
    outer_coefficients: FloatArray,
    transformed_occupied: FloatArray,
    transformed_target: FloatArray,
    secondary_weights: FloatArray,
    step_energies: FloatArray,
    secondary_sorted_order: npt.NDArray[np.int64],
    secondary_sorted_step_energies: FloatArray,
    secondary_sorted_target: FloatArray,
    secondary_sorted_occupied: FloatArray,
    secondary_sorted_weights: FloatArray,
    inner_strict: FloatArray,
    inner_affine: FloatArray,
    inner_coefficients: FloatArray,
    energy_differences: FloatArray,
    delta_weights: FloatArray,
    delta_sorted_order: npt.NDArray[np.int64],
    delta_sorted_energies: FloatArray,
    delta_shifted_energies: FloatArray,
    triangle_strict: FloatArray,
    triangle_affine: FloatArray,
    triangle_coefficients: FloatArray,
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_occupied,
        outer_strict,
        outer_affine,
        outer_coefficients,
    )
    if volume_factor <= 1.0e-10:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += outer_coefficients[row_index, column_index] * sorted_occupied[column_index]
        transformed_occupied[row_index] = total

    for row_index in range(4):
        for target_band_index in range(target_band_count):
            total = 0.0
            for column_index in range(4):
                total += outer_coefficients[row_index, column_index] * sorted_target[column_index, target_band_index]
            transformed_target[row_index, target_band_index] = total

    _fermigr_secondary_weights_numba(
        transformed_occupied,
        transformed_target,
        sample_energies,
        secondary_weights,
        step_energies,
        secondary_sorted_order,
        secondary_sorted_step_energies,
        secondary_sorted_target,
        secondary_sorted_occupied,
        secondary_sorted_weights,
        inner_strict,
        inner_affine,
        inner_coefficients,
        energy_differences,
        delta_weights,
        delta_sorted_order,
        delta_sorted_energies,
        delta_shifted_energies,
        triangle_strict,
        triangle_affine,
        triangle_coefficients,
    )

    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        for target_band_index in range(target_band_count):
            for column_index in range(4):
                total = 0.0
                for row_index in range(4):
                    total += secondary_weights[energy_index, target_band_index, row_index] * outer_coefficients[
                        row_index,
                        column_index,
                    ]
                outer_weights[energy_index, target_band_index, sorted_order[column_index]] += volume_factor * total


@njit(cache=True)
def _fermigr_secondary_weights_numba(
    occupied_vertices: FloatArray,
    target_vertices: FloatArray,
    sample_energies: FloatArray,
    weights: FloatArray,
    step_energies: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_step_energies: FloatArray,
    sorted_target: FloatArray,
    sorted_occupied: FloatArray,
    sorted_weights: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    delta_weights: FloatArray,
    delta_sorted_order: npt.NDArray[np.int64],
    delta_sorted_energies: FloatArray,
    delta_shifted_energies: FloatArray,
    triangle_strict: FloatArray,
    triangle_affine: FloatArray,
    triangle_coefficients: FloatArray,
) -> None:
    target_band_count = target_vertices.shape[1]
    energy_count = sample_energies.shape[0]
    weights[:, :, :] = 0.0

    for target_band_index in range(target_band_count):
        for vertex_index in range(4):
            step_energies[vertex_index] = -target_vertices[vertex_index, target_band_index]
        sort4(step_energies, sorted_order, sorted_step_energies)
        for vertex_index in range(4):
            source_vertex = sorted_order[vertex_index]
            sorted_target[vertex_index] = target_vertices[source_vertex, target_band_index]
            sorted_occupied[vertex_index] = occupied_vertices[source_vertex]
        sorted_weights[:, :] = 0.0

        if ((sorted_step_energies[0] <= 0.0 < sorted_step_energies[1]) or
                (sorted_step_energies[0] < 0.0 <= sorted_step_energies[1])):
            _accumulate_small_tetra_fermigr_inner_numba(
                sorted_weights,
                0,
                sorted_step_energies,
                sorted_occupied,
                sorted_target,
                sample_energies,
                strict_energies,
                affine,
                coefficients,
                energy_differences,
                delta_weights,
                delta_sorted_order,
                delta_sorted_energies,
                delta_shifted_energies,
                triangle_strict,
                triangle_affine,
                triangle_coefficients,
            )
        elif ((sorted_step_energies[1] <= 0.0 < sorted_step_energies[2]) or
                (sorted_step_energies[1] < 0.0 <= sorted_step_energies[2])):
            for case_id in (1, 2, 3):
                _accumulate_small_tetra_fermigr_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                )
        elif ((sorted_step_energies[2] <= 0.0 < sorted_step_energies[3]) or
                (sorted_step_energies[2] < 0.0 <= sorted_step_energies[3])):
            for case_id in (4, 5, 6):
                _accumulate_small_tetra_fermigr_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                    delta_weights,
                    delta_sorted_order,
                    delta_sorted_energies,
                    delta_shifted_energies,
                    triangle_strict,
                    triangle_affine,
                    triangle_coefficients,
                )
        elif sorted_step_energies[3] <= 0.0:
            for vertex_index in range(4):
                energy_differences[vertex_index] = sorted_target[vertex_index] - sorted_occupied[vertex_index]
            _fermigr_delta_weights_numba(
                sample_energies,
                energy_differences,
                delta_weights,
                delta_sorted_order,
                delta_sorted_energies,
                delta_shifted_energies,
                triangle_strict,
                triangle_affine,
                triangle_coefficients,
            )
            for energy_index in range(energy_count):
                for vertex_index in range(4):
                    sorted_weights[energy_index, vertex_index] += delta_weights[energy_index, vertex_index]

        for energy_index in range(energy_count):
            for vertex_index in range(4):
                weights[energy_index, target_band_index, sorted_order[vertex_index]] = sorted_weights[
                    energy_index,
                    vertex_index,
                ]


@njit(cache=True)
def _accumulate_small_tetra_fermigr_inner_numba(
    weights: FloatArray,
    case_id: int,
    sorted_step_energies: FloatArray,
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energies: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    delta_weights: FloatArray,
    delta_sorted_order: npt.NDArray[np.int64],
    delta_sorted_energies: FloatArray,
    delta_shifted_energies: FloatArray,
    triangle_strict: FloatArray,
    triangle_affine: FloatArray,
    triangle_coefficients: FloatArray,
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_step_energies,
        strict_energies,
        affine,
        coefficients,
    )
    if volume_factor <= 1.0e-8:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += coefficients[row_index, column_index] * (sorted_target[column_index] - sorted_occupied[column_index])
        energy_differences[row_index] = total

    _fermigr_delta_weights_numba(
        sample_energies,
        energy_differences,
        delta_weights,
        delta_sorted_order,
        delta_sorted_energies,
        delta_shifted_energies,
        triangle_strict,
        triangle_affine,
        triangle_coefficients,
    )

    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        for column_index in range(4):
            total = 0.0
            for row_index in range(4):
                total += delta_weights[energy_index, row_index] * coefficients[row_index, column_index]
            weights[energy_index, column_index] += volume_factor * total


@njit(cache=True)
def _fermigr_delta_weights_numba(
    sample_energies: FloatArray,
    energy_differences: FloatArray,
    weights: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_energies: FloatArray,
    shifted_energies: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
) -> None:
    sort4(energy_differences, sorted_order, sorted_energies)
    energy_count = sample_energies.shape[0]

    for energy_index in range(energy_count):
        energy = sample_energies[energy_index]
        for vertex_index in range(4):
            weights[energy_index, vertex_index] = 0.0
            shifted_energies[vertex_index] = sorted_energies[vertex_index] - energy

        if sorted_energies[0] < energy <= sorted_energies[1]:
            accumulate_triangle_weight_sums(
                weights[energy_index],
                sorted_order,
                0,
                shifted_energies,
                strict_energies,
                1.0 / 3.0,
                affine,
                coefficients,
            )
        elif sorted_energies[1] < energy <= sorted_energies[2]:
            accumulate_triangle_weight_sums(
                weights[energy_index],
                sorted_order,
                1,
                shifted_energies,
                strict_energies,
                1.0 / 3.0,
                affine,
                coefficients,
            )
            accumulate_triangle_weight_sums(
                weights[energy_index],
                sorted_order,
                2,
                shifted_energies,
                strict_energies,
                1.0 / 3.0,
                affine,
                coefficients,
            )
        elif sorted_energies[2] < energy < sorted_energies[3]:
            accumulate_triangle_weight_sums(
                weights[energy_index],
                sorted_order,
                3,
                shifted_energies,
                strict_energies,
                1.0 / 3.0,
                affine,
                coefficients,
            )


@njit(cache=True, parallel=True)
def _complex_polarization_weights_on_local_mesh_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: ComplexArray,
    local_point_count: int,
    normalization: int,
) -> ComplexArray:
    tetrahedron_count = occupied_tetra.shape[0]
    source_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    energy_count = sample_energies.shape[0]
    local_weights = np.zeros((local_point_count, energy_count, target_band_count, source_band_count), dtype=np.complex128)

    for source_band_index in prange(source_band_count):
        sorted_order = np.empty(4, dtype=np.int64)
        sorted_occupied = np.empty(4, dtype=np.float64)
        sorted_target = np.empty((4, target_band_count), dtype=np.float64)
        outer_weights = np.empty((energy_count, target_band_count, 4), dtype=np.complex128)
        point_weights = np.empty(20, dtype=np.complex128)

        outer_strict = np.empty(4, dtype=np.float64)
        outer_affine = np.empty((4, 4), dtype=np.float64)
        outer_coefficients = np.empty((4, 4), dtype=np.float64)
        transformed_occupied = np.empty(4, dtype=np.float64)
        transformed_target = np.empty((4, target_band_count), dtype=np.float64)

        secondary_weights = np.empty((energy_count, target_band_count, 4), dtype=np.complex128)
        step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_order = np.empty(4, dtype=np.int64)
        secondary_sorted_step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_target = np.empty(4, dtype=np.float64)
        secondary_sorted_occupied = np.empty(4, dtype=np.float64)
        secondary_sorted_weights = np.empty((energy_count, 4), dtype=np.complex128)
        inner_strict = np.empty(4, dtype=np.float64)
        inner_affine = np.empty((4, 4), dtype=np.float64)
        inner_coefficients = np.empty((4, 4), dtype=np.float64)
        energy_differences = np.empty(4, dtype=np.float64)
        sample_weights = np.empty((energy_count, 4), dtype=np.complex128)

        for tetrahedron_index in range(tetrahedron_count):
            sort4(
                occupied_tetra[tetrahedron_index, :, source_band_index],
                sorted_order,
                sorted_occupied,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                for target_band_index in range(target_band_count):
                    sorted_target[vertex_index, target_band_index] = target_tetra[
                        tetrahedron_index,
                        source_vertex,
                        target_band_index,
                    ]

            outer_weights[:, :, :] = 0.0j
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_polcmplx_outer_numba(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    target_band_count,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_occupied,
                    transformed_target,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    sample_weights,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for case_id in (1, 2, 3):
                    _accumulate_small_tetra_polcmplx_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        sample_weights,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for case_id in (4, 5, 6):
                    _accumulate_small_tetra_polcmplx_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        sample_weights,
                    )
            elif sorted_occupied[3] <= 0.0:
                _polcmplx_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    target_tetra[tetrahedron_index],
                    sample_energies,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    sample_weights,
                )
                for energy_index in range(energy_count):
                    for target_band_index in range(target_band_count):
                        for vertex_index in range(4):
                            outer_weights[energy_index, target_band_index, vertex_index] += secondary_weights[
                                energy_index,
                                target_band_index,
                                vertex_index,
                            ]

            for energy_index in range(energy_count):
                for target_band_index in range(target_band_count):
                    for point_index in range(20):
                        total = 0.0j
                        for vertex_index in range(4):
                            total += outer_weights[energy_index, target_band_index, vertex_index] * tetrahedron_weight_matrix[
                                vertex_index,
                                point_index,
                            ]
                        point_weights[point_index] = total
                    for point_index in range(20):
                        local_weights[
                            local_point_indices[tetrahedron_index, point_index],
                            energy_index,
                            target_band_index,
                            source_band_index,
                        ] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True, parallel=True)
def _complex_polarization_weights_on_local_mesh_pair_parallel_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    sample_energies: ComplexArray,
    local_point_count: int,
    normalization: int,
) -> ComplexArray:
    tetrahedron_count = occupied_tetra.shape[0]
    source_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    energy_count = sample_energies.shape[0]
    pair_count = source_band_count * target_band_count
    local_weights = np.zeros((local_point_count, energy_count, target_band_count, source_band_count), dtype=np.complex128)

    for pair_index in prange(pair_count):
        source_band_index = pair_index // target_band_count
        target_band_index = pair_index % target_band_count

        sorted_order = np.empty(4, dtype=np.int64)
        sorted_occupied = np.empty(4, dtype=np.float64)
        sorted_target = np.empty((4, 1), dtype=np.float64)
        full_target = np.empty((4, 1), dtype=np.float64)
        outer_weights = np.empty((energy_count, 1, 4), dtype=np.complex128)
        point_weights = np.empty(20, dtype=np.complex128)

        outer_strict = np.empty(4, dtype=np.float64)
        outer_affine = np.empty((4, 4), dtype=np.float64)
        outer_coefficients = np.empty((4, 4), dtype=np.float64)
        transformed_occupied = np.empty(4, dtype=np.float64)
        transformed_target = np.empty((4, 1), dtype=np.float64)

        secondary_weights = np.empty((energy_count, 1, 4), dtype=np.complex128)
        step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_order = np.empty(4, dtype=np.int64)
        secondary_sorted_step_energies = np.empty(4, dtype=np.float64)
        secondary_sorted_target = np.empty(4, dtype=np.float64)
        secondary_sorted_occupied = np.empty(4, dtype=np.float64)
        secondary_sorted_weights = np.empty((energy_count, 4), dtype=np.complex128)
        inner_strict = np.empty(4, dtype=np.float64)
        inner_affine = np.empty((4, 4), dtype=np.float64)
        inner_coefficients = np.empty((4, 4), dtype=np.float64)
        energy_differences = np.empty(4, dtype=np.float64)
        sample_weights = np.empty((energy_count, 4), dtype=np.complex128)

        for tetrahedron_index in range(tetrahedron_count):
            sort4(
                occupied_tetra[tetrahedron_index, :, source_band_index],
                sorted_order,
                sorted_occupied,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                sorted_target[vertex_index, 0] = target_tetra[
                    tetrahedron_index,
                    source_vertex,
                    target_band_index,
                ]
                full_target[vertex_index, 0] = target_tetra[
                    tetrahedron_index,
                    vertex_index,
                    target_band_index,
                ]

            outer_weights[:, :, :] = 0.0j
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_polcmplx_outer_numba(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    1,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_occupied,
                    transformed_target,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    sample_weights,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for case_id in (1, 2, 3):
                    _accumulate_small_tetra_polcmplx_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        1,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        sample_weights,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for case_id in (4, 5, 6):
                    _accumulate_small_tetra_polcmplx_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        sample_energies,
                        1,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        sample_weights,
                    )
            elif sorted_occupied[3] <= 0.0:
                _polcmplx_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, source_band_index],
                    full_target,
                    sample_energies,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    sample_weights,
                )
                for energy_index in range(energy_count):
                    for vertex_index in range(4):
                        outer_weights[energy_index, 0, vertex_index] += secondary_weights[
                            energy_index,
                            0,
                            vertex_index,
                        ]

            for energy_index in range(energy_count):
                for point_index in range(20):
                    total = 0.0j
                    for vertex_index in range(4):
                        total += outer_weights[energy_index, 0, vertex_index] * tetrahedron_weight_matrix[
                            vertex_index,
                            point_index,
                        ]
                    point_weights[point_index] = total
                for point_index in range(20):
                    local_weights[
                        local_point_indices[tetrahedron_index, point_index],
                        energy_index,
                        target_band_index,
                        source_band_index,
                    ] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True)
def _accumulate_small_tetra_polcmplx_outer_numba(
    outer_weights: ComplexArray,
    case_id: int,
    sorted_order: npt.NDArray[np.int64],
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energies: ComplexArray,
    target_band_count: int,
    outer_strict: FloatArray,
    outer_affine: FloatArray,
    outer_coefficients: FloatArray,
    transformed_occupied: FloatArray,
    transformed_target: FloatArray,
    secondary_weights: ComplexArray,
    step_energies: FloatArray,
    secondary_sorted_order: npt.NDArray[np.int64],
    secondary_sorted_step_energies: FloatArray,
    secondary_sorted_target: FloatArray,
    secondary_sorted_occupied: FloatArray,
    secondary_sorted_weights: ComplexArray,
    inner_strict: FloatArray,
    inner_affine: FloatArray,
    inner_coefficients: FloatArray,
    energy_differences: FloatArray,
    sample_weights: ComplexArray,
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_occupied,
        outer_strict,
        outer_affine,
        outer_coefficients,
    )
    if volume_factor <= 1.0e-8:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += outer_coefficients[row_index, column_index] * sorted_occupied[column_index]
        transformed_occupied[row_index] = total

    for row_index in range(4):
        for target_band_index in range(target_band_count):
            total = 0.0
            for column_index in range(4):
                total += outer_coefficients[row_index, column_index] * sorted_target[column_index, target_band_index]
            transformed_target[row_index, target_band_index] = total

    _polcmplx_secondary_weights_numba(
        transformed_occupied,
        transformed_target,
        sample_energies,
        secondary_weights,
        step_energies,
        secondary_sorted_order,
        secondary_sorted_step_energies,
        secondary_sorted_target,
        secondary_sorted_occupied,
        secondary_sorted_weights,
        inner_strict,
        inner_affine,
        inner_coefficients,
        energy_differences,
        sample_weights,
    )

    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        for target_band_index in range(target_band_count):
            for column_index in range(4):
                total = 0.0j
                for row_index in range(4):
                    total += secondary_weights[energy_index, target_band_index, row_index] * outer_coefficients[
                        row_index,
                        column_index,
                    ]
                outer_weights[energy_index, target_band_index, sorted_order[column_index]] += volume_factor * total


@njit(cache=True)
def _polcmplx_secondary_weights_numba(
    occupied_vertices: FloatArray,
    target_vertices: FloatArray,
    sample_energies: ComplexArray,
    weights: ComplexArray,
    step_energies: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_step_energies: FloatArray,
    sorted_target: FloatArray,
    sorted_occupied: FloatArray,
    sorted_weights: ComplexArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    sample_weights: ComplexArray,
) -> None:
    target_band_count = target_vertices.shape[1]
    energy_count = sample_energies.shape[0]
    weights[:, :, :] = 0.0j

    for target_band_index in range(target_band_count):
        for vertex_index in range(4):
            step_energies[vertex_index] = -target_vertices[vertex_index, target_band_index]
        sort4(step_energies, sorted_order, sorted_step_energies)
        for vertex_index in range(4):
            source_vertex = sorted_order[vertex_index]
            sorted_target[vertex_index] = target_vertices[source_vertex, target_band_index]
            sorted_occupied[vertex_index] = occupied_vertices[source_vertex]
        sorted_weights[:, :] = 0.0j

        if ((sorted_step_energies[0] <= 0.0 < sorted_step_energies[1]) or
                (sorted_step_energies[0] < 0.0 <= sorted_step_energies[1])):
            _accumulate_small_tetra_polcmplx_inner_numba(
                sorted_weights,
                0,
                sorted_step_energies,
                sorted_occupied,
                sorted_target,
                sample_energies,
                strict_energies,
                affine,
                coefficients,
                energy_differences,
                sample_weights,
            )
        elif ((sorted_step_energies[1] <= 0.0 < sorted_step_energies[2]) or
                (sorted_step_energies[1] < 0.0 <= sorted_step_energies[2])):
            for case_id in (1, 2, 3):
                _accumulate_small_tetra_polcmplx_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                    sample_weights,
                )
        elif ((sorted_step_energies[2] <= 0.0 < sorted_step_energies[3]) or
                (sorted_step_energies[2] < 0.0 <= sorted_step_energies[3])):
            for case_id in (4, 5, 6):
                _accumulate_small_tetra_polcmplx_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    sample_energies,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                    sample_weights,
                )
        elif sorted_step_energies[3] <= 0.0:
            for vertex_index in range(4):
                energy_differences[vertex_index] = sorted_target[vertex_index] - sorted_occupied[vertex_index]
            _polcmplx_sample_weights_numba(sample_energies, energy_differences, sample_weights)
            for energy_index in range(energy_count):
                for vertex_index in range(4):
                    sorted_weights[energy_index, vertex_index] += sample_weights[energy_index, vertex_index]

        for energy_index in range(energy_count):
            for vertex_index in range(4):
                weights[energy_index, target_band_index, sorted_order[vertex_index]] = sorted_weights[
                    energy_index,
                    vertex_index,
                ]


@njit(cache=True)
def _accumulate_small_tetra_polcmplx_inner_numba(
    weights: ComplexArray,
    case_id: int,
    sorted_step_energies: FloatArray,
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    sample_energies: ComplexArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    sample_weights: ComplexArray,
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_step_energies,
        strict_energies,
        affine,
        coefficients,
    )
    if volume_factor <= 1.0e-8:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += coefficients[row_index, column_index] * (sorted_target[column_index] - sorted_occupied[column_index])
        energy_differences[row_index] = total

    _polcmplx_sample_weights_numba(sample_energies, energy_differences, sample_weights)

    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        for column_index in range(4):
            total = 0.0j
            for row_index in range(4):
                total += sample_weights[energy_index, row_index] * coefficients[row_index, column_index]
            weights[energy_index, column_index] += volume_factor * total


@njit(cache=True)
def _polcmplx_sample_weights_numba(
    sample_energies: ComplexArray,
    energy_differences: FloatArray,
    weights: ComplexArray,
) -> None:
    # The live legacy polcmplx3 path is just the direct vertex average 0.25 / (de + z).
    energy_count = sample_energies.shape[0]
    for energy_index in range(energy_count):
        for vertex_index in range(4):
            weights[energy_index, vertex_index] = 0.25 / (energy_differences[vertex_index] + sample_energies[energy_index])


@njit(cache=True)
def _static_polarization_weights_on_local_mesh_numba(
    local_point_indices: npt.NDArray[np.int64],
    tetrahedron_weight_matrix: FloatArray,
    occupied_tetra: FloatArray,
    target_tetra: FloatArray,
    local_point_count: int,
    normalization: int,
) -> FloatArray:
    tetrahedron_count = occupied_tetra.shape[0]
    occupied_band_count = occupied_tetra.shape[2]
    target_band_count = target_tetra.shape[2]
    local_weights = np.zeros((local_point_count, target_band_count, occupied_band_count), dtype=np.float64)

    sorted_order = np.empty(4, dtype=np.int64)
    sorted_occupied = np.empty(4, dtype=np.float64)
    sorted_target = np.empty((4, target_band_count), dtype=np.float64)
    outer_weights = np.empty((target_band_count, 4), dtype=np.float64)
    point_weights = np.empty(20, dtype=np.float64)

    outer_strict = np.empty(4, dtype=np.float64)
    outer_affine = np.empty((4, 4), dtype=np.float64)
    outer_coefficients = np.empty((4, 4), dtype=np.float64)
    transformed_occupied = np.empty(4, dtype=np.float64)
    transformed_target = np.empty((4, target_band_count), dtype=np.float64)

    secondary_weights = np.empty((target_band_count, 4), dtype=np.float64)
    step_energies = np.empty(4, dtype=np.float64)
    secondary_sorted_order = np.empty(4, dtype=np.int64)
    secondary_sorted_step_energies = np.empty(4, dtype=np.float64)
    secondary_sorted_target = np.empty(4, dtype=np.float64)
    secondary_sorted_occupied = np.empty(4, dtype=np.float64)
    secondary_sorted_weights = np.empty(4, dtype=np.float64)
    inner_strict = np.empty(4, dtype=np.float64)
    inner_affine = np.empty((4, 4), dtype=np.float64)
    inner_coefficients = np.empty((4, 4), dtype=np.float64)
    energy_differences = np.empty(4, dtype=np.float64)
    log_weights = np.empty(4, dtype=np.float64)
    log_sorted_order = np.empty(4, dtype=np.int64)
    log_sorted_differences = np.empty(4, dtype=np.float64)
    logarithms = np.empty(4, dtype=np.float64)
    log_sorted_weights = np.empty(4, dtype=np.float64)

    for tetrahedron_index in range(tetrahedron_count):
        for occupied_band_index in range(occupied_band_count):
            sort4(
                occupied_tetra[tetrahedron_index, :, occupied_band_index],
                sorted_order,
                sorted_occupied,
            )
            for vertex_index in range(4):
                source_vertex = sorted_order[vertex_index]
                for target_band_index in range(target_band_count):
                    sorted_target[vertex_index, target_band_index] = target_tetra[
                        tetrahedron_index,
                        source_vertex,
                        target_band_index,
                    ]

            outer_weights[:, :] = 0.0
            if sorted_occupied[0] <= 0.0 < sorted_occupied[1]:
                _accumulate_small_tetra_polstat_outer_numba(
                    outer_weights,
                    0,
                    sorted_order,
                    sorted_occupied,
                    sorted_target,
                    target_band_count,
                    outer_strict,
                    outer_affine,
                    outer_coefficients,
                    transformed_occupied,
                    transformed_target,
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    log_weights,
                    log_sorted_order,
                    log_sorted_differences,
                    logarithms,
                    log_sorted_weights,
                )
            elif sorted_occupied[1] <= 0.0 < sorted_occupied[2]:
                for case_id in (1, 2, 3):
                    _accumulate_small_tetra_polstat_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        log_weights,
                        log_sorted_order,
                        log_sorted_differences,
                        logarithms,
                        log_sorted_weights,
                    )
            elif sorted_occupied[2] <= 0.0 < sorted_occupied[3]:
                for case_id in (4, 5, 6):
                    _accumulate_small_tetra_polstat_outer_numba(
                        outer_weights,
                        case_id,
                        sorted_order,
                        sorted_occupied,
                        sorted_target,
                        target_band_count,
                        outer_strict,
                        outer_affine,
                        outer_coefficients,
                        transformed_occupied,
                        transformed_target,
                        secondary_weights,
                        step_energies,
                        secondary_sorted_order,
                        secondary_sorted_step_energies,
                        secondary_sorted_target,
                        secondary_sorted_occupied,
                        secondary_sorted_weights,
                        inner_strict,
                        inner_affine,
                        inner_coefficients,
                        energy_differences,
                        log_weights,
                        log_sorted_order,
                        log_sorted_differences,
                        logarithms,
                        log_sorted_weights,
                    )
            elif sorted_occupied[3] <= 0.0:
                _polstat_secondary_weights_numba(
                    occupied_tetra[tetrahedron_index, :, occupied_band_index],
                    target_tetra[tetrahedron_index],
                    secondary_weights,
                    step_energies,
                    secondary_sorted_order,
                    secondary_sorted_step_energies,
                    secondary_sorted_target,
                    secondary_sorted_occupied,
                    secondary_sorted_weights,
                    inner_strict,
                    inner_affine,
                    inner_coefficients,
                    energy_differences,
                    log_weights,
                    log_sorted_order,
                    log_sorted_differences,
                    logarithms,
                    log_sorted_weights,
                )
                for target_band_index in range(target_band_count):
                    for vertex_index in range(4):
                        outer_weights[target_band_index, vertex_index] += secondary_weights[target_band_index, vertex_index]

            for target_band_index in range(target_band_count):
                for point_index in range(20):
                    total = 0.0
                    for vertex_index in range(4):
                        total += outer_weights[target_band_index, vertex_index] * tetrahedron_weight_matrix[vertex_index, point_index]
                    point_weights[point_index] = total
                for point_index in range(20):
                    local_weights[
                        local_point_indices[tetrahedron_index, point_index],
                        target_band_index,
                        occupied_band_index,
                    ] += point_weights[point_index]

    local_weights /= float(normalization)
    return local_weights


@njit(cache=True)
def _accumulate_small_tetra_polstat_outer_numba(
    outer_weights: FloatArray,
    case_id: int,
    sorted_order: npt.NDArray[np.int64],
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    target_band_count: int,
    outer_strict: FloatArray,
    outer_affine: FloatArray,
    outer_coefficients: FloatArray,
    transformed_occupied: FloatArray,
    transformed_target: FloatArray,
    secondary_weights: FloatArray,
    step_energies: FloatArray,
    secondary_sorted_order: npt.NDArray[np.int64],
    secondary_sorted_step_energies: FloatArray,
    secondary_sorted_target: FloatArray,
    secondary_sorted_occupied: FloatArray,
    secondary_sorted_weights: FloatArray,
    inner_strict: FloatArray,
    inner_affine: FloatArray,
    inner_coefficients: FloatArray,
    energy_differences: FloatArray,
    log_weights: FloatArray,
    log_sorted_order: npt.NDArray[np.int64],
    log_sorted_differences: FloatArray,
    logarithms: FloatArray,
    log_sorted_weights: FloatArray,
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_occupied,
        outer_strict,
        outer_affine,
        outer_coefficients,
    )
    if volume_factor <= 1.0e-10:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += outer_coefficients[row_index, column_index] * sorted_occupied[column_index]
        transformed_occupied[row_index] = total

    for row_index in range(4):
        for target_band_index in range(target_band_count):
            total = 0.0
            for column_index in range(4):
                total += outer_coefficients[row_index, column_index] * sorted_target[column_index, target_band_index]
            transformed_target[row_index, target_band_index] = total

    _polstat_secondary_weights_numba(
        transformed_occupied,
        transformed_target,
        secondary_weights,
        step_energies,
        secondary_sorted_order,
        secondary_sorted_step_energies,
        secondary_sorted_target,
        secondary_sorted_occupied,
        secondary_sorted_weights,
        inner_strict,
        inner_affine,
        inner_coefficients,
        energy_differences,
        log_weights,
        log_sorted_order,
        log_sorted_differences,
        logarithms,
        log_sorted_weights,
    )

    for target_band_index in range(target_band_count):
        for column_index in range(4):
            total = 0.0
            for row_index in range(4):
                total += secondary_weights[target_band_index, row_index] * outer_coefficients[row_index, column_index]
            outer_weights[target_band_index, sorted_order[column_index]] += volume_factor * total


@njit(cache=True)
def _polstat_secondary_weights_numba(
    occupied_vertices: FloatArray,
    target_vertices: FloatArray,
    weights: FloatArray,
    step_energies: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_step_energies: FloatArray,
    sorted_target: FloatArray,
    sorted_occupied: FloatArray,
    sorted_weights: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    log_weights: FloatArray,
    log_sorted_order: npt.NDArray[np.int64],
    log_sorted_differences: FloatArray,
    logarithms: FloatArray,
    log_sorted_weights: FloatArray,
) -> None:
    target_band_count = target_vertices.shape[1]
    weights[:, :] = 0.0

    for target_band_index in range(target_band_count):
        for vertex_index in range(4):
            step_energies[vertex_index] = -target_vertices[vertex_index, target_band_index]
        sort4(step_energies, sorted_order, sorted_step_energies)
        for vertex_index in range(4):
            source_vertex = sorted_order[vertex_index]
            sorted_target[vertex_index] = target_vertices[source_vertex, target_band_index]
            sorted_occupied[vertex_index] = occupied_vertices[source_vertex]
        sorted_weights[:] = 0.0

        if ((sorted_step_energies[0] <= 0.0 < sorted_step_energies[1]) or
                (sorted_step_energies[0] < 0.0 <= sorted_step_energies[1])):
            _accumulate_small_tetra_polstat_inner_numba(
                sorted_weights,
                0,
                sorted_step_energies,
                sorted_occupied,
                sorted_target,
                strict_energies,
                affine,
                coefficients,
                energy_differences,
                log_weights,
                log_sorted_order,
                log_sorted_differences,
                logarithms,
                log_sorted_weights,
            )
        elif ((sorted_step_energies[1] <= 0.0 < sorted_step_energies[2]) or
                (sorted_step_energies[1] < 0.0 <= sorted_step_energies[2])):
            for case_id in (1, 2, 3):
                _accumulate_small_tetra_polstat_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                    log_weights,
                    log_sorted_order,
                    log_sorted_differences,
                    logarithms,
                    log_sorted_weights,
                )
        elif ((sorted_step_energies[2] <= 0.0 < sorted_step_energies[3]) or
                (sorted_step_energies[2] < 0.0 <= sorted_step_energies[3])):
            for case_id in (4, 5, 6):
                _accumulate_small_tetra_polstat_inner_numba(
                    sorted_weights,
                    case_id,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                    strict_energies,
                    affine,
                    coefficients,
                    energy_differences,
                    log_weights,
                    log_sorted_order,
                    log_sorted_differences,
                    logarithms,
                    log_sorted_weights,
                )
        elif sorted_step_energies[3] <= 0.0:
            for vertex_index in range(4):
                energy_differences[vertex_index] = sorted_target[vertex_index] - sorted_occupied[vertex_index]
            _polstat_logarithmic_weights_numba(
                energy_differences,
                log_weights,
                log_sorted_order,
                log_sorted_differences,
                logarithms,
                log_sorted_weights,
            )
            for vertex_index in range(4):
                sorted_weights[vertex_index] += log_weights[vertex_index]

        for vertex_index in range(4):
            weights[target_band_index, sorted_order[vertex_index]] = sorted_weights[vertex_index]


@njit(cache=True)
def _accumulate_small_tetra_polstat_inner_numba(
    weights: FloatArray,
    case_id: int,
    sorted_step_energies: FloatArray,
    sorted_occupied: FloatArray,
    sorted_target: FloatArray,
    strict_energies: FloatArray,
    affine: FloatArray,
    coefficients: FloatArray,
    energy_differences: FloatArray,
    log_weights: FloatArray,
    log_sorted_order: npt.NDArray[np.int64],
    log_sorted_differences: FloatArray,
    logarithms: FloatArray,
    log_sorted_weights: FloatArray,
) -> None:
    volume_factor = small_tetra_volume_and_coefficients(
        case_id,
        sorted_step_energies,
        strict_energies,
        affine,
        coefficients,
    )
    if volume_factor <= 1.0e-8:
        return

    for row_index in range(4):
        total = 0.0
        for column_index in range(4):
            total += coefficients[row_index, column_index] * (sorted_target[column_index] - sorted_occupied[column_index])
        energy_differences[row_index] = total

    _polstat_logarithmic_weights_numba(
        energy_differences,
        log_weights,
        log_sorted_order,
        log_sorted_differences,
        logarithms,
        log_sorted_weights,
    )

    for column_index in range(4):
        total = 0.0
        for row_index in range(4):
            total += log_weights[row_index] * coefficients[row_index, column_index]
        weights[column_index] += volume_factor * total


@njit(cache=True)
def _polstat_logarithmic_weights_numba(
    energy_differences: FloatArray,
    weights: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_differences: FloatArray,
    logarithms: FloatArray,
    sorted_weights: FloatArray,
) -> None:
    sort4(energy_differences, sorted_order, sorted_differences)
    threshold = sorted_differences[3] * 1.0e-3
    absolute_floor = 1.0e-8

    for index in range(4):
        if sorted_differences[index] < absolute_floor:
            if index == 2:
                raise RuntimeError("encountered nesting condition in static_polarization_weights")
            logarithms[index] = 0.0
            sorted_differences[index] = 0.0
        else:
            logarithms[index] = np.log(sorted_differences[index])

    sorted_weights[:] = 0.0

    if abs(sorted_differences[3] - sorted_differences[2]) < threshold:
        if abs(sorted_differences[3] - sorted_differences[1]) < threshold:
            if abs(sorted_differences[3] - sorted_differences[0]) < threshold:
                sorted_weights[:] = 0.25 / sorted_differences[3]
            else:
                sorted_weights[3] = _polstat_1211_numba(sorted_differences[3], sorted_differences[0], logarithms[3], logarithms[0])
                sorted_weights[2] = sorted_weights[3]
                sorted_weights[1] = sorted_weights[3]
                sorted_weights[0] = _polstat_1222_numba(sorted_differences[0], sorted_differences[3], logarithms[0], logarithms[3])
                _check_polstat_weights_numba(sorted_weights)
        elif abs(sorted_differences[1] - sorted_differences[0]) < threshold:
            sorted_weights[3] = _polstat_1221_numba(sorted_differences[3], sorted_differences[1], logarithms[3], logarithms[1])
            sorted_weights[2] = sorted_weights[3]
            sorted_weights[1] = _polstat_1221_numba(sorted_differences[1], sorted_differences[3], logarithms[1], logarithms[3])
            sorted_weights[0] = sorted_weights[1]
            _check_polstat_weights_numba(sorted_weights)
        else:
            sorted_weights[3] = _polstat_1231_numba(
                sorted_differences[3],
                sorted_differences[0],
                sorted_differences[1],
                logarithms[3],
                logarithms[0],
                logarithms[1],
            )
            sorted_weights[2] = sorted_weights[3]
            sorted_weights[1] = _polstat_1233_numba(
                sorted_differences[1],
                sorted_differences[0],
                sorted_differences[3],
                logarithms[1],
                logarithms[0],
                logarithms[3],
            )
            sorted_weights[0] = _polstat_1233_numba(
                sorted_differences[0],
                sorted_differences[1],
                sorted_differences[3],
                logarithms[0],
                logarithms[1],
                logarithms[3],
            )
            _check_polstat_weights_numba(sorted_weights)
    elif abs(sorted_differences[2] - sorted_differences[1]) < threshold:
        if abs(sorted_differences[2] - sorted_differences[0]) < threshold:
            sorted_weights[3] = _polstat_1222_numba(sorted_differences[3], sorted_differences[2], logarithms[3], logarithms[2])
            sorted_weights[2] = _polstat_1211_numba(sorted_differences[2], sorted_differences[3], logarithms[2], logarithms[3])
            sorted_weights[1] = sorted_weights[2]
            sorted_weights[0] = sorted_weights[2]
            _check_polstat_weights_numba(sorted_weights)
        else:
            sorted_weights[3] = _polstat_1233_numba(
                sorted_differences[3],
                sorted_differences[0],
                sorted_differences[2],
                logarithms[3],
                logarithms[0],
                logarithms[2],
            )
            sorted_weights[2] = _polstat_1231_numba(
                sorted_differences[2],
                sorted_differences[0],
                sorted_differences[3],
                logarithms[2],
                logarithms[0],
                logarithms[3],
            )
            sorted_weights[1] = sorted_weights[2]
            sorted_weights[0] = _polstat_1233_numba(
                sorted_differences[0],
                sorted_differences[3],
                sorted_differences[2],
                logarithms[0],
                logarithms[3],
                logarithms[2],
            )
            _check_polstat_weights_numba(sorted_weights)
    elif abs(sorted_differences[1] - sorted_differences[0]) < threshold:
        sorted_weights[3] = _polstat_1233_numba(
            sorted_differences[3],
            sorted_differences[2],
            sorted_differences[1],
            logarithms[3],
            logarithms[2],
            logarithms[1],
        )
        sorted_weights[2] = _polstat_1233_numba(
            sorted_differences[2],
            sorted_differences[3],
            sorted_differences[1],
            logarithms[2],
            logarithms[3],
            logarithms[1],
        )
        sorted_weights[1] = _polstat_1231_numba(
            sorted_differences[1],
            sorted_differences[2],
            sorted_differences[3],
            logarithms[1],
            logarithms[2],
            logarithms[3],
        )
        sorted_weights[0] = sorted_weights[1]
        _check_polstat_weights_numba(sorted_weights)
    else:
        sorted_weights[3] = _polstat_1234_numba(
            sorted_differences[3],
            sorted_differences[0],
            sorted_differences[1],
            sorted_differences[2],
            logarithms[3],
            logarithms[0],
            logarithms[1],
            logarithms[2],
        )
        sorted_weights[2] = _polstat_1234_numba(
            sorted_differences[2],
            sorted_differences[0],
            sorted_differences[1],
            sorted_differences[3],
            logarithms[2],
            logarithms[0],
            logarithms[1],
            logarithms[3],
        )
        sorted_weights[1] = _polstat_1234_numba(
            sorted_differences[1],
            sorted_differences[0],
            sorted_differences[2],
            sorted_differences[3],
            logarithms[1],
            logarithms[0],
            logarithms[2],
            logarithms[3],
        )
        sorted_weights[0] = _polstat_1234_numba(
            sorted_differences[0],
            sorted_differences[1],
            sorted_differences[2],
            sorted_differences[3],
            logarithms[0],
            logarithms[1],
            logarithms[2],
            logarithms[3],
        )
        _check_polstat_weights_numba(sorted_weights)

    weights[:] = 0.0
    for vertex_index in range(4):
        weights[sorted_order[vertex_index]] = sorted_weights[vertex_index]


@njit(cache=True)
def _check_polstat_weights_numba(weights: FloatArray) -> None:
    for vertex_index in range(4):
        if weights[vertex_index] < 0.0:
            raise RuntimeError("negative static_polarization_weights values encountered")


@njit(cache=True)
def _polstat_1234_numba(g1: float, g2: float, g3: float, g4: float, log1: float, log2: float, log3: float, log4: float) -> float:
    weight_2 = (((log2 - log1) / (g2 - g1) * g2) - 1.0) * g2 / (g2 - g1)
    weight_3 = (((log3 - log1) / (g3 - g1) * g3) - 1.0) * g3 / (g3 - g1)
    weight_4 = (((log4 - log1) / (g4 - g1) * g4) - 1.0) * g4 / (g4 - g1)
    weight_2 = ((weight_2 - weight_3) * g2) / (g2 - g3)
    weight_4 = ((weight_4 - weight_3) * g4) / (g4 - g3)
    return (weight_4 - weight_2) / (g4 - g2)


@njit(cache=True)
def _polstat_1231_numba(g1: float, g2: float, g3: float, log1: float, log2: float, log3: float) -> float:
    weight_2 = ((((log2 - log1) / (g2 - g1) * g2) - 1.0) * g2 ** 2 / (g2 - g1) - g1 / 2.0) / (g2 - g1)
    weight_3 = ((((log3 - log1) / (g3 - g1) * g3) - 1.0) * g3 ** 2 / (g3 - g1) - g1 / 2.0) / (g3 - g1)
    return (weight_3 - weight_2) / (g3 - g2)


@njit(cache=True)
def _polstat_1233_numba(g1: float, g2: float, g3: float, log1: float, log2: float, log3: float) -> float:
    weight_2 = (log2 - log1) / (g2 - g1) * g2 - 1.0
    weight_2 = (g2 * weight_2) / (g2 - g1)
    weight_3 = (log3 - log1) / (g3 - g1) * g3 - 1.0
    weight_3 = (g3 * weight_3) / (g3 - g1)
    weight_2 = (weight_3 - weight_2) / (g3 - g2)
    weight_3 = (log3 - log1) / (g3 - g1) * g3 - 1.0
    weight_3 = 1.0 - (2.0 * weight_3 * g1) / (g3 - g1)
    weight_3 = weight_3 / (g3 - g1)
    return (g3 * weight_3 - g2 * weight_2) / (g3 - g2)


@njit(cache=True)
def _polstat_1221_numba(g1: float, g2: float, log1: float, log2: float) -> float:
    weight = 1.0 - (log2 - log1) / (g2 - g1) * g1
    weight = -1.0 + (2.0 * g2 * weight) / (g2 - g1)
    weight = -1.0 + (3.0 * g2 * weight) / (g2 - g1)
    return weight / (2.0 * (g2 - g1))


@njit(cache=True)
def _polstat_1222_numba(g1: float, g2: float, log1: float, log2: float) -> float:
    weight = (log2 - log1) / (g2 - g1) * g2 - 1.0
    weight = (2.0 * g1 * weight) / (g2 - g1) - 1.0
    weight = (3.0 * g1 * weight) / (g2 - g1) + 1.0
    return weight / (2.0 * (g2 - g1))


@njit(cache=True)
def _polstat_1211_numba(g1: float, g2: float, log1: float, log2: float) -> float:
    weight = -1.0 + (log2 - log1) / (g2 - g1) * g2
    weight = -1.0 + (2.0 * g2 * weight) / (g2 - g1)
    weight = -1.0 + (3.0 * g2 * weight) / (2.0 * (g2 - g1))
    return weight / (3.0 * (g2 - g1))


def _accumulate_small_tetra_step(weights: FloatArray, kind: str, sorted_energies: FloatArray) -> None:
    cut = small_tetrahedron_cut(kind, sorted_energies)
    weights += cut.volume_factor * cut.coefficients.sum(axis=0) * 0.25


def _normalize_eigenvalue_pair(
    occupied_eigenvalues: npt.ArrayLike,
    target_eigenvalues: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, tuple[int, int, int]]:
    occupied_flat, occupied_grid_shape = normalize_eigenvalues(occupied_eigenvalues)
    target_flat, target_grid_shape = normalize_eigenvalues(target_eigenvalues)
    if occupied_grid_shape != target_grid_shape:
        raise ValueError("occupied and target eigenvalue grids must share the same shape")
    return occupied_flat, target_flat, occupied_grid_shape


def _normalize_sorted_triangle_energies(energies: npt.ArrayLike) -> FloatArray:
    values = np.asarray(energies, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"expected three sorted energies, got shape {values.shape!r}")
    if not np.all(np.isfinite(values)):
        raise ValueError("triangle energies must be finite")

    adjusted = values.copy()
    if np.any(np.diff(adjusted) < 0.0):
        raise ValueError("triangle energies must be sorted in nondecreasing order")
    for index in range(1, adjusted.size):
        if adjusted[index] <= adjusted[index - 1]:
            adjusted[index] = np.nextafter(adjusted[index - 1], np.inf)
    return adjusted


def _triangle_affine_coefficients(energies: FloatArray) -> FloatArray:
    coefficients = np.full((3, 3), np.nan, dtype=np.float64)
    for column, energy in enumerate(energies):
        mask = np.arange(3) != column
        coefficients[mask, column] = -energy / (energies[mask] - energy)
    return coefficients


def _unflatten_pair_band_last(values: FloatArray, grid_shape: tuple[int, int, int]) -> FloatArray:
    target_band_count = values.shape[1]
    source_band_count = values.shape[2]
    reshaped = values.reshape((grid_shape[2], grid_shape[1], grid_shape[0], target_band_count, source_band_count))
    return np.transpose(reshaped, (2, 1, 0, 3, 4))


def _unflatten_energy_pair_band_last(values: FloatArray, grid_shape: tuple[int, int, int]) -> FloatArray:
    energy_count = values.shape[1]
    target_band_count = values.shape[2]
    source_band_count = values.shape[3]
    reshaped = values.reshape((grid_shape[2], grid_shape[1], grid_shape[0], energy_count, target_band_count, source_band_count))
    return np.transpose(reshaped, (3, 2, 1, 0, 4, 5))
