from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit


FloatArray = npt.NDArray[np.float64]


@njit(cache=True)
def sort3(
    values: FloatArray,
    sorted_order: npt.NDArray[np.int64],
    sorted_values: FloatArray,
) -> None:
    order0 = 0
    order1 = 1
    order2 = 2
    value0 = values[0]
    value1 = values[1]
    value2 = values[2]

    if value1 < value0:
        value0, value1 = value1, value0
        order0, order1 = order1, order0
    if value2 < value1:
        value1, value2 = value2, value1
        order1, order2 = order2, order1
    if value1 < value0:
        value0, value1 = value1, value0
        order0, order1 = order1, order0

    sorted_order[0] = order0
    sorted_order[1] = order1
    sorted_order[2] = order2
    sorted_values[0] = value0
    sorted_values[1] = value1
    sorted_values[2] = value2


@njit(cache=True)
def strict_sorted_energies3(sorted_values: FloatArray, strict_values: FloatArray) -> None:
    strict_values[0] = sorted_values[0]
    for index in range(1, 3):
        strict_values[index] = sorted_values[index]
        if strict_values[index] <= strict_values[index - 1]:
            strict_values[index] = np.nextafter(strict_values[index - 1], np.inf)


@njit(cache=True)
def occupation_fraction3(
    energy: float,
    sorted_values: FloatArray,
    strict_values: FloatArray,
) -> float:
    if energy <= sorted_values[0]:
        return 0.0
    if energy >= sorted_values[2]:
        return 1.0

    if energy <= sorted_values[1]:
        x = energy - strict_values[0]
        alpha = 1.0 / (strict_values[1] - strict_values[0])
        beta = 1.0 / (strict_values[2] - strict_values[0])
        return alpha * beta * x * x

    t = strict_values[2] - energy
    gamma = 1.0 / (strict_values[2] - strict_values[1])
    delta = 1.0 / (strict_values[2] - strict_values[0])
    return 1.0 - gamma * delta * t * t


@njit(cache=True)
def fill_occupation_vertex_weights3(
    energy: float,
    sorted_values: FloatArray,
    strict_values: FloatArray,
    sorted_weights: FloatArray,
) -> None:
    sorted_weights[:] = 0.0

    if energy <= sorted_values[0]:
        return
    if energy >= sorted_values[2]:
        sorted_weights[:] = 1.0 / 3.0
        return

    if energy <= sorted_values[1]:
        x = energy - strict_values[0]
        alpha = 1.0 / (strict_values[1] - strict_values[0])
        beta = 1.0 / (strict_values[2] - strict_values[0])
        a = alpha * x
        b = beta * x
        sorted_weights[0] = a * b * (3.0 - a - b) / 3.0
        sorted_weights[1] = a * a * b / 3.0
        sorted_weights[2] = a * b * b / 3.0
        return

    t = strict_values[2] - energy
    gamma = 1.0 / (strict_values[2] - strict_values[1])
    delta = 1.0 / (strict_values[2] - strict_values[0])
    c = gamma * t
    d = delta * t
    sorted_weights[0] = 1.0 / 3.0 - c * d * (1.0 - d) / 3.0
    sorted_weights[1] = 1.0 / 3.0 - c * d * (1.0 - c) / 3.0
    sorted_weights[2] = 1.0 / 3.0 - c * d * (1.0 + c + d) / 3.0


@njit(cache=True)
def fill_dos_vertex_weights3(
    energy: float,
    sorted_values: FloatArray,
    strict_values: FloatArray,
    sorted_weights: FloatArray,
) -> None:
    sorted_weights[:] = 0.0

    if energy < sorted_values[0] or energy > sorted_values[2]:
        return

    if energy <= sorted_values[1]:
        x = energy - strict_values[0]
        alpha = 1.0 / (strict_values[1] - strict_values[0])
        beta = 1.0 / (strict_values[2] - strict_values[0])
        sorted_weights[0] = alpha * beta * x * (2.0 - x * (alpha + beta))
        sorted_weights[1] = alpha * alpha * beta * x * x
        sorted_weights[2] = alpha * beta * beta * x * x
        return

    t = strict_values[2] - energy
    gamma = 1.0 / (strict_values[2] - strict_values[1])
    delta = 1.0 / (strict_values[2] - strict_values[0])
    sorted_weights[0] = gamma * delta * t * (2.0 - 3.0 * delta * t) / 3.0
    sorted_weights[1] = gamma * delta * t * (2.0 - 3.0 * gamma * t) / 3.0
    sorted_weights[2] = gamma * delta * t * (2.0 + 3.0 * (gamma + delta) * t) / 3.0
