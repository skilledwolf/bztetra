from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def sort3(values, sorted_order, sorted_energies) -> None:
    sorted_order[0] = 0
    sorted_order[1] = 1
    sorted_order[2] = 2
    sorted_energies[0] = values[0]
    sorted_energies[1] = values[1]
    sorted_energies[2] = values[2]

    if sorted_energies[0] > sorted_energies[1]:
        _swap_sorted_entry(sorted_order, sorted_energies, 0, 1)
    if sorted_energies[1] > sorted_energies[2]:
        _swap_sorted_entry(sorted_order, sorted_energies, 1, 2)
    if sorted_energies[0] > sorted_energies[1]:
        _swap_sorted_entry(sorted_order, sorted_energies, 0, 1)


@njit(cache=True)
def fill_occupation_vertex_weights(
    vertex_weights,
    sorted_order,
    sorted_energies,
    fermi_energy,
    strict_energies,
) -> None:
    vertex_weights[:] = 0.0

    if fermi_energy <= sorted_energies[0]:
        return
    if fermi_energy >= sorted_energies[2]:
        for vertex_index in range(3):
            vertex_weights[sorted_order[vertex_index]] = 1.0 / 3.0
        return

    _strict_sorted_energies3(sorted_energies, strict_energies)
    if fermi_energy <= sorted_energies[1]:
        _fill_low_occupation_weights(vertex_weights, sorted_order, strict_energies, fermi_energy)
        return

    _fill_high_occupation_weights(vertex_weights, sorted_order, strict_energies, fermi_energy)


@njit(cache=True)
def fill_dos_vertex_weights(
    vertex_weights,
    sorted_order,
    sorted_energies,
    energy,
    strict_energies,
) -> None:
    vertex_weights[:] = 0.0

    if energy <= sorted_energies[0] or energy >= sorted_energies[2]:
        return

    _strict_sorted_energies3(sorted_energies, strict_energies)
    if energy <= sorted_energies[1]:
        _fill_low_dos_weights(vertex_weights, sorted_order, strict_energies, energy)
        return

    _fill_high_dos_weights(vertex_weights, sorted_order, strict_energies, energy)


@njit(cache=True)
def _fill_low_occupation_weights(vertex_weights, sorted_order, strict_energies, fermi_energy) -> None:
    e0 = strict_energies[0]
    e1 = strict_energies[1]
    e2 = strict_energies[2]
    a = (fermi_energy - e0) / (e1 - e0)
    b = (fermi_energy - e0) / (e2 - e0)

    w1 = a * a * b / 3.0
    w2 = a * b * b / 3.0
    w0 = a * b - w1 - w2

    vertex_weights[sorted_order[0]] = w0
    vertex_weights[sorted_order[1]] = w1
    vertex_weights[sorted_order[2]] = w2


@njit(cache=True)
def _fill_high_occupation_weights(vertex_weights, sorted_order, strict_energies, fermi_energy) -> None:
    e0 = strict_energies[0]
    e1 = strict_energies[1]
    e2 = strict_energies[2]
    a = (e2 - fermi_energy) / (e2 - e0)
    b = (e2 - fermi_energy) / (e2 - e1)

    w0 = 1.0 / 3.0 - a * a * b / 3.0
    w1 = 1.0 / 3.0 - a * b * b / 3.0
    w2 = 1.0 / 3.0 - a * b * (3.0 - a - b) / 3.0

    vertex_weights[sorted_order[0]] = w0
    vertex_weights[sorted_order[1]] = w1
    vertex_weights[sorted_order[2]] = w2


@njit(cache=True)
def _fill_low_dos_weights(vertex_weights, sorted_order, strict_energies, energy) -> None:
    e0 = strict_energies[0]
    e1 = strict_energies[1]
    e2 = strict_energies[2]
    a = (energy - e0) / (e1 - e0)
    b = (energy - e0) / (e2 - e0)
    da = 1.0 / (e1 - e0)
    db = 1.0 / (e2 - e0)

    w1 = (2.0 * a * da * b + a * a * db) / 3.0
    w2 = (da * b * b + 2.0 * a * b * db) / 3.0
    total = da * b + a * db
    w0 = total - w1 - w2

    vertex_weights[sorted_order[0]] = w0
    vertex_weights[sorted_order[1]] = w1
    vertex_weights[sorted_order[2]] = w2


@njit(cache=True)
def _fill_high_dos_weights(vertex_weights, sorted_order, strict_energies, energy) -> None:
    e0 = strict_energies[0]
    e1 = strict_energies[1]
    e2 = strict_energies[2]
    a = (e2 - energy) / (e2 - e0)
    b = (e2 - energy) / (e2 - e1)
    da = 1.0 / (e2 - e0)
    db = 1.0 / (e2 - e1)

    w0 = (2.0 * a * da * b + a * a * db) / 3.0
    w1 = (da * b * b + 2.0 * a * b * db) / 3.0
    total = da * b + a * db
    w2 = total - w0 - w1

    vertex_weights[sorted_order[0]] = w0
    vertex_weights[sorted_order[1]] = w1
    vertex_weights[sorted_order[2]] = w2


@njit(cache=True)
def _strict_sorted_energies3(sorted_energies, strict_energies) -> None:
    strict_energies[0] = sorted_energies[0]
    for index in range(1, 3):
        value = sorted_energies[index]
        if value <= strict_energies[index - 1]:
            value = np.nextafter(strict_energies[index - 1], np.inf)
        strict_energies[index] = value


@njit(cache=True)
def _swap_sorted_entry(sorted_order, sorted_energies, first_index, second_index) -> None:
    order = sorted_order[first_index]
    sorted_order[first_index] = sorted_order[second_index]
    sorted_order[second_index] = order

    energy = sorted_energies[first_index]
    sorted_energies[first_index] = sorted_energies[second_index]
    sorted_energies[second_index] = energy
