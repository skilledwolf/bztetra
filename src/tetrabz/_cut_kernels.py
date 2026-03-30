from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def sort4(values, sorted_order, sorted_values):
    for index in range(4):
        sorted_order[index] = index
        sorted_values[index] = values[index]

    for index in range(1, 4):
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
def sort4_shifted(values, shift, sorted_order, sorted_values):
    for index in range(4):
        sorted_order[index] = index
        sorted_values[index] = values[index] - shift

    for index in range(1, 4):
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
def accumulate_small_tetra_weight_sums(
    weights,
    sorted_order,
    case_id,
    sorted_energies,
    strict_energies,
    scale,
    affine,
    coefficients,
):
    volume_factor = small_tetra_volume_and_coefficients(case_id, sorted_energies, strict_energies, affine, coefficients)
    for column in range(4):
        column_sum = 0.0
        for row in range(4):
            column_sum += coefficients[row, column]
        weights[sorted_order[column]] += volume_factor * column_sum * scale


@njit(cache=True)
def accumulate_triangle_weight_sums(
    weights,
    sorted_order,
    case_id,
    shifted_energies,
    strict_energies,
    scale,
    affine,
    coefficients,
):
    volume_factor = triangle_volume_and_coefficients(case_id, shifted_energies, strict_energies, affine, coefficients)
    for column in range(4):
        column_sum = 0.0
        for row in range(3):
            column_sum += coefficients[row, column]
        weights[sorted_order[column]] += volume_factor * column_sum * scale


@njit(cache=True)
def small_tetra_volume_and_coefficients(case_id, sorted_energies, strict_energies, affine, coefficients):
    _strict_sorted_energies4(sorted_energies, strict_energies)
    _fill_simplex_affine4(strict_energies, affine)
    coefficients[:, :] = 0.0

    if case_id == 0:
        volume_factor = affine[1, 0] * affine[2, 0] * affine[3, 0]
        coefficients[0, 0] = 1.0
        coefficients[1, 0] = affine[0, 1]
        coefficients[1, 1] = affine[1, 0]
        coefficients[2, 0] = affine[0, 2]
        coefficients[2, 2] = affine[2, 0]
        coefficients[3, 0] = affine[0, 3]
        coefficients[3, 3] = affine[3, 0]
    elif case_id == 1:
        volume_factor = affine[2, 0] * affine[3, 0] * affine[1, 3]
        coefficients[0, 0] = 1.0
        coefficients[1, 0] = affine[0, 2]
        coefficients[1, 2] = affine[2, 0]
        coefficients[2, 0] = affine[0, 3]
        coefficients[2, 3] = affine[3, 0]
        coefficients[3, 1] = affine[1, 3]
        coefficients[3, 3] = affine[3, 1]
    elif case_id == 2:
        volume_factor = affine[2, 1] * affine[3, 1]
        coefficients[0, 0] = 1.0
        coefficients[1, 1] = 1.0
        coefficients[2, 1] = affine[1, 2]
        coefficients[2, 2] = affine[2, 1]
        coefficients[3, 1] = affine[1, 3]
        coefficients[3, 3] = affine[3, 1]
    elif case_id == 3:
        volume_factor = affine[1, 2] * affine[2, 0] * affine[3, 1]
        coefficients[0, 0] = 1.0
        coefficients[1, 0] = affine[0, 2]
        coefficients[1, 2] = affine[2, 0]
        coefficients[2, 1] = affine[1, 2]
        coefficients[2, 2] = affine[2, 1]
        coefficients[3, 1] = affine[1, 3]
        coefficients[3, 3] = affine[3, 1]
    elif case_id == 4:
        volume_factor = affine[3, 2]
        coefficients[0, 0] = 1.0
        coefficients[1, 1] = 1.0
        coefficients[2, 2] = 1.0
        coefficients[3, 2] = affine[2, 3]
        coefficients[3, 3] = affine[3, 2]
    elif case_id == 5:
        volume_factor = affine[2, 3] * affine[3, 1]
        coefficients[0, 0] = 1.0
        coefficients[1, 1] = 1.0
        coefficients[2, 1] = affine[1, 3]
        coefficients[2, 3] = affine[3, 1]
        coefficients[3, 2] = affine[2, 3]
        coefficients[3, 3] = affine[3, 2]
    else:
        volume_factor = affine[2, 3] * affine[1, 3] * affine[3, 0]
        coefficients[0, 0] = 1.0
        coefficients[1, 0] = affine[0, 3]
        coefficients[1, 3] = affine[3, 0]
        coefficients[2, 1] = affine[1, 3]
        coefficients[2, 3] = affine[3, 1]
        coefficients[3, 2] = affine[2, 3]
        coefficients[3, 3] = affine[3, 2]

    return volume_factor


@njit(cache=True)
def triangle_volume_and_coefficients(case_id, shifted_energies, strict_energies, affine, coefficients):
    _strict_sorted_energies4(shifted_energies, strict_energies)
    _fill_simplex_affine4(strict_energies, affine)
    coefficients[:, :] = 0.0

    if case_id == 0:
        volume_factor = 3.0 * affine[1, 0] * affine[2, 0] / (strict_energies[3] - strict_energies[0])
        coefficients[0, 0] = affine[0, 1]
        coefficients[0, 1] = affine[1, 0]
        coefficients[1, 0] = affine[0, 2]
        coefficients[1, 2] = affine[2, 0]
        coefficients[2, 0] = affine[0, 3]
        coefficients[2, 3] = affine[3, 0]
    elif case_id == 1:
        volume_factor = 3.0 * affine[3, 0] * affine[1, 3] / (strict_energies[2] - strict_energies[0])
        coefficients[0, 0] = affine[0, 2]
        coefficients[0, 2] = affine[2, 0]
        coefficients[1, 0] = affine[0, 3]
        coefficients[1, 3] = affine[3, 0]
        coefficients[2, 1] = affine[1, 3]
        coefficients[2, 3] = affine[3, 1]
    elif case_id == 2:
        volume_factor = 3.0 * affine[1, 2] * affine[3, 1] / (strict_energies[2] - strict_energies[0])
        coefficients[0, 0] = affine[0, 2]
        coefficients[0, 2] = affine[2, 0]
        coefficients[1, 1] = affine[1, 2]
        coefficients[1, 2] = affine[2, 1]
        coefficients[2, 1] = affine[1, 3]
        coefficients[2, 3] = affine[3, 1]
    else:
        volume_factor = 3.0 * affine[0, 3] * affine[1, 3] / (strict_energies[3] - strict_energies[2])
        coefficients[0, 0] = affine[0, 3]
        coefficients[0, 3] = affine[3, 0]
        coefficients[1, 1] = affine[1, 3]
        coefficients[1, 3] = affine[3, 1]
        coefficients[2, 2] = affine[2, 3]
        coefficients[2, 3] = affine[3, 2]

    return volume_factor


@njit(cache=True)
def _strict_sorted_energies4(sorted_energies, adjusted_energies):
    for index in range(4):
        adjusted_energies[index] = sorted_energies[index]

    for index in range(1, 4):
        if adjusted_energies[index] <= adjusted_energies[index - 1]:
            adjusted_energies[index] = np.nextafter(adjusted_energies[index - 1], np.inf)


@njit(cache=True)
def _fill_simplex_affine4(energies, affine):
    affine[:, :] = 0.0
    for column in range(4):
        energy = energies[column]
        for row in range(4):
            if row != column:
                affine[row, column] = -energy / (energies[row] - energy)
