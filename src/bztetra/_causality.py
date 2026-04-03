from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt
from numba import njit


_OPERATOR_CACHE_ENTRY_LIMIT = 8
_MAX_CACHED_OPERATOR_BYTES = 16 * 1024 * 1024


def _reconstruct_retarded_response_impl(
    omega: npt.ArrayLike,
    imag_response: npt.ArrayLike,
    *,
    static_anchor: npt.ArrayLike | None,
    support_bounds: tuple[float, float] | npt.ArrayLike | None,
    assume_hermitian: bool,
) -> dict[str, object]:
    original_omega = _normalize_real_frequency_grid(omega)
    original_imag = _normalize_imaginary_response(imag_response, original_omega.size)
    anchor = _normalize_static_anchor(static_anchor, original_imag.shape[1:])
    bounds = _normalize_support_bounds(support_bounds)

    if anchor is not None and not assume_hermitian and original_omega[0] > 0.0:
        raise ValueError(
            "non-Hermitian static_anchor requires omega[0] == 0.0 because only the returned "
            "zero-frequency sample is pinned"
        )

    working_omega = original_omega
    working_imag = original_imag
    output_indices = np.arange(original_omega.size, dtype=np.int64)

    if bounds is not None:
        lower_bound, upper_bound = bounds
        if upper_bound > working_omega[-1] + _spacing_tolerance(working_omega[-1], working_omega[-2]):
            raise ValueError(
                "omega grid does not cover the requested support upper bound: "
                f"{working_omega[-1]:.12g} < {upper_bound:.12g}"
            )
        (
            working_omega,
            working_imag,
            output_indices,
            support_was_clipped,
            support_boundary_insertions,
        ) = _clip_to_support_bounds(
            working_omega,
            working_imag,
            output_indices,
            lower_bound,
            upper_bound,
        )
    else:
        support_was_clipped = False
        support_boundary_insertions = 0

    inserted_zero_frequency = False
    if working_omega[0] > 0.0:
        inserted_zero_frequency = True
        zero_row = np.zeros((1,) + working_imag.shape[1:], dtype=np.float64)
        working_omega = np.concatenate((np.array([0.0], dtype=np.float64), working_omega))
        working_imag = np.concatenate((zero_row, working_imag), axis=0)
        output_indices = output_indices + 1

    zero_frequency_adjusted = False
    zero_tolerance = max(1.0, float(np.max(np.abs(working_imag)))) * 1.0e-12
    if (
        assume_hermitian
        and working_omega[0] == 0.0
        and np.max(np.abs(working_imag[0])) > zero_tolerance
    ):
        working_imag = working_imag.copy()
        working_imag[0] = 0.0
        zero_frequency_adjusted = True

    operator, augmented_count, cached_operator = _causality_operator_matrix(
        working_omega,
        assume_hermitian=assume_hermitian,
    )
    flat_imag = np.ascontiguousarray(working_imag.reshape(working_imag.shape[0], -1))
    real_flat = operator @ flat_imag

    static_anchor_applied = False
    if anchor is not None:
        anchor_flat = np.asarray(anchor, dtype=np.float64).reshape(-1)
        if assume_hermitian:
            real_flat += (anchor_flat - real_flat[0])[None, :]
            static_anchor_applied = True
        else:
            real_flat[0] = anchor_flat
            static_anchor_applied = True

    working_real = real_flat.reshape(working_imag.shape)
    spacing = np.diff(working_omega)

    return {
        "omega": original_omega,
        "imag": working_imag[output_indices],
        "real": working_real[output_indices],
        "support_bounds": bounds,
        "minimum_spacing": float(np.min(spacing)),
        "maximum_spacing": float(np.max(spacing)),
        "inserted_zero_frequency": inserted_zero_frequency,
        "zero_frequency_adjusted": zero_frequency_adjusted,
        "support_was_clipped": support_was_clipped,
        "support_boundary_insertions": support_boundary_insertions,
        "static_anchor_applied": static_anchor_applied,
        "augmented_point_count": augmented_count,
        "cached_operator": cached_operator,
    }


def _normalize_real_frequency_grid(omega: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.asarray(omega, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("omega must be a one-dimensional grid")
    if values.size < 2:
        raise ValueError("omega must contain at least two sample points")
    if not np.all(np.isfinite(values)):
        raise ValueError("omega must be finite")
    if np.any(values < 0.0):
        raise ValueError("automatic Kramers-Kronig reconstruction expects omega >= 0")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("omega must be strictly increasing")
    return values


def _normalize_imaginary_response(
    imag_response: npt.ArrayLike,
    omega_count: int,
) -> npt.NDArray[np.float64]:
    values = np.asarray(imag_response)
    if values.shape[0] != omega_count:
        raise ValueError(
            "imag_response must have the same leading axis length as omega, "
            f"got {values.shape[0]} and {omega_count}"
        )
    if np.iscomplexobj(values):
        if not np.allclose(values.imag, 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("imag_response must be real-valued")
        values = values.real
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("imag_response must be finite")
    return values


def _normalize_static_anchor(
    static_anchor: npt.ArrayLike | None,
    trailing_shape: tuple[int, ...],
) -> npt.NDArray[np.float64] | None:
    if static_anchor is None:
        return None

    values = np.asarray(static_anchor)
    if np.iscomplexobj(values):
        if not np.allclose(values.imag, 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("static_anchor must be real-valued")
        values = values.real
    values = np.asarray(values, dtype=np.float64)
    if values.shape != trailing_shape:
        raise ValueError(
            "static_anchor must match imag_response.shape[1:], "
            f"got {values.shape!r} and {trailing_shape!r}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("static_anchor must be finite")
    return values


def _normalize_support_bounds(
    support_bounds: tuple[float, float] | npt.ArrayLike | None,
) -> tuple[float, float] | None:
    if support_bounds is None:
        return None

    values = np.asarray(support_bounds, dtype=np.float64)
    if values.shape != (2,):
        raise ValueError("support_bounds must have shape (2,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("support_bounds must be finite")
    if values[1] <= values[0]:
        raise ValueError("support_bounds must satisfy lower < upper")
    return max(0.0, float(values[0])), float(values[1])


def _clip_to_support_bounds(
    omega: npt.NDArray[np.float64],
    imag_response: npt.NDArray[np.float64],
    output_indices: npt.NDArray[np.int64],
    lower_bound: float,
    upper_bound: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    bool,
    int,
]:
    clipped_omega = omega
    clipped_imag = imag_response
    clipped_indices = output_indices
    support_was_clipped = False
    boundary_insertions = 0

    if lower_bound > clipped_omega[0]:
        clipped_omega, clipped_imag, clipped_indices, inserted = _insert_support_boundary(
            clipped_omega,
            clipped_imag,
            clipped_indices,
            lower_bound,
        )
        support_was_clipped = support_was_clipped or inserted
        boundary_insertions += int(inserted)

    if upper_bound < clipped_omega[-1]:
        clipped_omega, clipped_imag, clipped_indices, inserted = _insert_support_boundary(
            clipped_omega,
            clipped_imag,
            clipped_indices,
            upper_bound,
        )
        support_was_clipped = support_was_clipped or inserted
        boundary_insertions += int(inserted)

    mask = (clipped_omega < lower_bound) | (clipped_omega > upper_bound)
    if np.any(mask):
        clipped_imag = clipped_imag.copy()
        clipped_imag[mask] = 0.0
        support_was_clipped = True

    return clipped_omega, clipped_imag, clipped_indices, support_was_clipped, boundary_insertions


def _insert_support_boundary(
    omega: npt.NDArray[np.float64],
    imag_response: npt.NDArray[np.float64],
    output_indices: npt.NDArray[np.int64],
    boundary: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64], bool]:
    insertion_index = int(np.searchsorted(omega, boundary, side="left"))
    if insertion_index == 0 or insertion_index == omega.size:
        return omega, imag_response, output_indices, False

    if _matches_sample(omega[insertion_index], boundary) or _matches_sample(omega[insertion_index - 1], boundary):
        return omega, imag_response, output_indices, False

    boundary_row = np.zeros((1,) + imag_response.shape[1:], dtype=np.float64)
    augmented_omega = np.insert(omega, insertion_index, boundary)
    augmented_imag = np.concatenate(
        (imag_response[:insertion_index], boundary_row, imag_response[insertion_index:]),
        axis=0,
    )
    augmented_indices = np.where(output_indices >= insertion_index, output_indices + 1, output_indices)
    return augmented_omega, augmented_imag, augmented_indices, True


def _matches_sample(sample: float, boundary: float) -> bool:
    tolerance = _spacing_tolerance(sample, boundary)
    return abs(sample - boundary) <= tolerance


def _spacing_tolerance(left_value: float, right_value: float) -> float:
    scale = max(1.0, abs(left_value), abs(right_value))
    return 1.0e-12 * scale


def _causality_operator_matrix(
    positive_omega: npt.NDArray[np.float64],
    *,
    assume_hermitian: bool,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    operator_bytes = positive_omega.size * positive_omega.size * np.dtype(np.float64).itemsize
    if operator_bytes > _MAX_CACHED_OPERATOR_BYTES:
        operator = _build_causality_operator_matrix_numba(positive_omega, assume_hermitian)
        operator.setflags(write=False)
        return operator, 2 * positive_omega.size + 1, False

    operator, augmented_count = _cached_causality_operator_matrix(
        positive_omega.tobytes(),
        positive_omega.size,
        assume_hermitian,
    )
    return operator, augmented_count, True


@lru_cache(maxsize=_OPERATOR_CACHE_ENTRY_LIMIT)
def _cached_causality_operator_matrix(
    omega_bytes: bytes,
    omega_count: int,
    assume_hermitian: bool,
) -> tuple[npt.NDArray[np.float64], int]:
    positive_omega = np.frombuffer(omega_bytes, dtype=np.float64, count=omega_count).copy()
    operator = _build_causality_operator_matrix_numba(positive_omega, assume_hermitian)
    operator.setflags(write=False)
    return operator, 2 * omega_count + 1


@njit(cache=True)
def _build_causality_operator_matrix_numba(positive_omega, assume_hermitian):
    positive_count = positive_omega.shape[0]
    augmented_omega = _build_augmented_omega_numba(positive_omega)
    augmented_count = augmented_omega.shape[0]
    operator = np.zeros((positive_count, positive_count), dtype=np.float64)
    coefficients = np.empty(augmented_count, dtype=np.float64)

    for output_index in range(positive_count):
        coefficients[:] = 0.0
        full_index = positive_count + output_index
        omega_value = augmented_omega[full_index]

        for interval_index in range(augmented_count - 1):
            if interval_index == full_index - 1 or interval_index == full_index:
                continue
            x0 = augmented_omega[interval_index]
            x1 = augmented_omega[interval_index + 1]
            width = x1 - x0
            log_term = np.log(abs((x1 - omega_value) / (x0 - omega_value)))
            coefficients[interval_index] += ((x1 - omega_value) / width) * log_term - 1.0
            coefficients[interval_index + 1] += 1.0 + ((omega_value - x0) / width) * log_term

        left_width = omega_value - augmented_omega[full_index - 1]
        right_width = augmented_omega[full_index + 1] - omega_value
        coefficients[full_index - 1] += -1.0
        coefficients[full_index] += np.log(right_width / left_width)
        coefficients[full_index + 1] += 1.0

        operator[output_index, 0] = coefficients[positive_count] / np.pi
        for sample_index in range(1, positive_count):
            coefficient = coefficients[positive_count + sample_index]
            if assume_hermitian:
                coefficient -= coefficients[positive_count - sample_index]
            operator[output_index, sample_index] = coefficient / np.pi

    return operator


@njit(cache=True)
def _build_augmented_omega_numba(positive_omega):
    positive_count = positive_omega.shape[0]
    right_spacing = positive_omega[-1] - positive_omega[-2]
    augmented_omega = np.empty(2 * positive_count + 1, dtype=np.float64)

    augmented_omega[0] = -positive_omega[-1] - right_spacing
    for offset in range(positive_count - 1):
        augmented_omega[1 + offset] = -positive_omega[positive_count - 1 - offset]
    augmented_omega[positive_count] = positive_omega[0]
    for offset in range(1, positive_count):
        augmented_omega[positive_count + offset] = positive_omega[offset]
    augmented_omega[2 * positive_count] = positive_omega[-1] + right_spacing

    return augmented_omega
