from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
SmallTetrahedronKind = Literal["a1", "b1", "b2", "b3", "c1", "c2", "c3"]
TriangleKind = Literal["a1", "b1", "b2", "c1"]


@dataclass(frozen=True, slots=True)
class SimplexCut:
    """Coefficient matrix and prefactor for one simplex cut construction.

    The legacy Fortran code returns a scalar ``V`` together with a matrix of
    barycentric coefficients. The Python port keeps that shape but makes the
    result explicit and zero-based.
    """

    volume_factor: float
    coefficients: FloatArray


def simplex_affine_coefficients(energies: npt.ArrayLike) -> FloatArray:
    """Return the legacy ``a(i, j)`` table for four strictly ordered energies."""

    values = _normalize_sorted_energies(energies)
    coefficients = np.full((4, 4), np.nan, dtype=np.float64)
    for column, energy in enumerate(values):
        mask = np.arange(4) != column
        coefficients[mask, column] = -energy / (values[mask] - energy)
    return coefficients


def small_tetrahedron_cut(kind: SmallTetrahedronKind, energies: npt.ArrayLike) -> SimplexCut:
    """Return the small-tetrahedron construction for one legacy case."""

    a = simplex_affine_coefficients(energies)
    coefficients = np.zeros((4, 4), dtype=np.float64)

    if kind == "a1":
        volume_factor = a[1, 0] * a[2, 0] * a[3, 0]
        coefficients[0] = [1.0, 0.0, 0.0, 0.0]
        coefficients[1] = [a[0, 1], a[1, 0], 0.0, 0.0]
        coefficients[2] = [a[0, 2], 0.0, a[2, 0], 0.0]
        coefficients[3] = [a[0, 3], 0.0, 0.0, a[3, 0]]
    elif kind == "b1":
        volume_factor = a[2, 0] * a[3, 0] * a[1, 3]
        coefficients[0] = [1.0, 0.0, 0.0, 0.0]
        coefficients[1] = [a[0, 2], 0.0, a[2, 0], 0.0]
        coefficients[2] = [a[0, 3], 0.0, 0.0, a[3, 0]]
        coefficients[3] = [0.0, a[1, 3], 0.0, a[3, 1]]
    elif kind == "b2":
        volume_factor = a[2, 1] * a[3, 1]
        coefficients[0] = [1.0, 0.0, 0.0, 0.0]
        coefficients[1] = [0.0, 1.0, 0.0, 0.0]
        coefficients[2] = [0.0, a[1, 2], a[2, 1], 0.0]
        coefficients[3] = [0.0, a[1, 3], 0.0, a[3, 1]]
    elif kind == "b3":
        volume_factor = a[1, 2] * a[2, 0] * a[3, 1]
        coefficients[0] = [1.0, 0.0, 0.0, 0.0]
        coefficients[1] = [a[0, 2], 0.0, a[2, 0], 0.0]
        coefficients[2] = [0.0, a[1, 2], a[2, 1], 0.0]
        coefficients[3] = [0.0, a[1, 3], 0.0, a[3, 1]]
    elif kind == "c1":
        volume_factor = a[3, 2]
        coefficients[0] = [1.0, 0.0, 0.0, 0.0]
        coefficients[1] = [0.0, 1.0, 0.0, 0.0]
        coefficients[2] = [0.0, 0.0, 1.0, 0.0]
        coefficients[3] = [0.0, 0.0, a[2, 3], a[3, 2]]
    elif kind == "c2":
        volume_factor = a[2, 3] * a[3, 1]
        coefficients[0] = [1.0, 0.0, 0.0, 0.0]
        coefficients[1] = [0.0, 1.0, 0.0, 0.0]
        coefficients[2] = [0.0, a[1, 3], 0.0, a[3, 1]]
        coefficients[3] = [0.0, 0.0, a[2, 3], a[3, 2]]
    elif kind == "c3":
        volume_factor = a[2, 3] * a[1, 3] * a[3, 0]
        coefficients[0] = [1.0, 0.0, 0.0, 0.0]
        coefficients[1] = [a[0, 3], 0.0, 0.0, a[3, 0]]
        coefficients[2] = [0.0, a[1, 3], 0.0, a[3, 1]]
        coefficients[3] = [0.0, 0.0, a[2, 3], a[3, 2]]
    else:
        raise ValueError(f"unsupported small tetrahedron kind {kind!r}")

    return SimplexCut(float(volume_factor), coefficients)


def triangle_cut(kind: TriangleKind, energies: npt.ArrayLike) -> SimplexCut:
    """Return the triangle construction for one legacy case."""

    values = _normalize_sorted_energies(energies)
    a = simplex_affine_coefficients(values)
    coefficients = np.zeros((3, 4), dtype=np.float64)

    if kind == "a1":
        volume_factor = 3.0 * a[1, 0] * a[2, 0] / (values[3] - values[0])
        coefficients[0] = [a[0, 1], a[1, 0], 0.0, 0.0]
        coefficients[1] = [a[0, 2], 0.0, a[2, 0], 0.0]
        coefficients[2] = [a[0, 3], 0.0, 0.0, a[3, 0]]
    elif kind == "b1":
        volume_factor = 3.0 * a[3, 0] * a[1, 3] / (values[2] - values[0])
        coefficients[0] = [a[0, 2], 0.0, a[2, 0], 0.0]
        coefficients[1] = [a[0, 3], 0.0, 0.0, a[3, 0]]
        coefficients[2] = [0.0, a[1, 3], 0.0, a[3, 1]]
    elif kind == "b2":
        volume_factor = 3.0 * a[1, 2] * a[3, 1] / (values[2] - values[0])
        coefficients[0] = [a[0, 2], 0.0, a[2, 0], 0.0]
        coefficients[1] = [0.0, a[1, 2], a[2, 1], 0.0]
        coefficients[2] = [0.0, a[1, 3], 0.0, a[3, 1]]
    elif kind == "c1":
        volume_factor = 3.0 * a[0, 3] * a[1, 3] / (values[3] - values[2])
        coefficients[0] = [a[0, 3], 0.0, 0.0, a[3, 0]]
        coefficients[1] = [0.0, a[1, 3], 0.0, a[3, 1]]
        coefficients[2] = [0.0, 0.0, a[2, 3], a[3, 2]]
    else:
        raise ValueError(f"unsupported triangle kind {kind!r}")

    return SimplexCut(float(volume_factor), coefficients)


def _normalize_sorted_energies(energies: npt.ArrayLike) -> FloatArray:
    values = np.asarray(energies, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError(f"expected four sorted energies, got shape {values.shape!r}")
    if not np.all(np.isfinite(values)):
        raise ValueError("energies must be finite")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("energies must be strictly increasing for the cut formulas")
    return values
