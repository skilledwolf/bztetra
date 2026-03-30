from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._grids import FloatArray
from .formulas import small_tetrahedron_cut


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
    weights[:, sorted_order] += cut.volume_factor * (
        _polstat_secondary_weights(transformed_occupied, transformed_target) @ cut.coefficients
    )



def _polstat_secondary_weights(
    occupied_vertices: FloatArray, target_vertices: FloatArray
) -> FloatArray:
    target_band_count = target_vertices.shape[1]
    weights = np.zeros((target_band_count, 4), dtype=np.float64)

    for target_band_index in range(target_band_count):
        sorted_order = np.argsort(-target_vertices[:, target_band_index])
        sorted_step_energies = -target_vertices[sorted_order, target_band_index]
        sorted_target = target_vertices[sorted_order, target_band_index]
        sorted_occupied = occupied_vertices[sorted_order]
        sorted_weights = np.zeros(4, dtype=np.float64)

        if (sorted_step_energies[0] <= 0.0 < sorted_step_energies[1]) or (
            sorted_step_energies[0] < 0.0 <= sorted_step_energies[1]
        ):
            _accumulate_small_tetra_polstat_inner(
                sorted_weights, "a1", sorted_step_energies, sorted_occupied, sorted_target
            )
        elif (sorted_step_energies[1] <= 0.0 < sorted_step_energies[2]) or (
            sorted_step_energies[1] < 0.0 <= sorted_step_energies[2]
        ):
            for kind in ("b1", "b2", "b3"):
                _accumulate_small_tetra_polstat_inner(
                    sorted_weights,
                    kind,
                    sorted_step_energies,
                    sorted_occupied,
                    sorted_target,
                )
        elif (sorted_step_energies[2] <= 0.0 < sorted_step_energies[3]) or (
            sorted_step_energies[2] < 0.0 <= sorted_step_energies[3]
        ):
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
    weights += cut.volume_factor * (
        _polstat_logarithmic_weights(energy_differences) @ cut.coefficients
    )


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
                sorted_weights[3] = _polstat_1211(
                    sorted_differences[3], sorted_differences[0], logarithms[3], logarithms[0]
                )
                sorted_weights[2] = sorted_weights[3]
                sorted_weights[1] = sorted_weights[3]
                sorted_weights[0] = _polstat_1222(
                    sorted_differences[0], sorted_differences[3], logarithms[0], logarithms[3]
                )
                _check_polstat_weights(sorted_weights, "4=3=2")
        elif abs(sorted_differences[1] - sorted_differences[0]) < threshold:
            sorted_weights[3] = _polstat_1221(
                sorted_differences[3], sorted_differences[1], logarithms[3], logarithms[1]
            )
            sorted_weights[2] = sorted_weights[3]
            sorted_weights[1] = _polstat_1221(
                sorted_differences[1], sorted_differences[3], logarithms[1], logarithms[3]
            )
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
            sorted_weights[3] = _polstat_1222(
                sorted_differences[3], sorted_differences[2], logarithms[3], logarithms[2]
            )
            sorted_weights[2] = _polstat_1211(
                sorted_differences[2], sorted_differences[3], logarithms[2], logarithms[3]
            )
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


def _polstat_1234(
    g1: float, g2: float, g3: float, g4: float, log1: float, log2: float, log3: float, log4: float
) -> float:
    weight_2 = (((log2 - log1) / (g2 - g1) * g2) - 1.0) * g2 / (g2 - g1)
    weight_3 = (((log3 - log1) / (g3 - g1) * g3) - 1.0) * g3 / (g3 - g1)
    weight_4 = (((log4 - log1) / (g4 - g1) * g4) - 1.0) * g4 / (g4 - g1)
    weight_2 = ((weight_2 - weight_3) * g2) / (g2 - g3)
    weight_4 = ((weight_4 - weight_3) * g4) / (g4 - g3)
    return (weight_4 - weight_2) / (g4 - g2)


def _polstat_1231(g1: float, g2: float, g3: float, log1: float, log2: float, log3: float) -> float:
    weight_2 = ((((log2 - log1) / (g2 - g1) * g2) - 1.0) * g2**2 / (g2 - g1) - g1 / 2.0) / (g2 - g1)
    weight_3 = ((((log3 - log1) / (g3 - g1) * g3) - 1.0) * g3**2 / (g3 - g1) - g1 / 2.0) / (g3 - g1)
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
