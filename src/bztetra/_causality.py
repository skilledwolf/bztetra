from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _reconstruct_retarded_response_impl(
    omega: npt.ArrayLike,
    imag_response: npt.ArrayLike,
    *,
    static_anchor: npt.ArrayLike | None,
    support_bounds: tuple[float, float] | npt.ArrayLike | None,
    assume_hermitian: bool,
    padding_tolerance: float,
    max_padding_factor: int,
) -> dict[str, object]:
    original_omega = _normalize_real_frequency_grid(omega)
    original_imag = _normalize_imaginary_response(imag_response, original_omega.size)
    anchor = _normalize_static_anchor(static_anchor, original_imag.shape[1:])
    bounds = _normalize_support_bounds(support_bounds)

    working_omega = original_omega
    working_imag = original_imag
    inserted_zero_frequency = False

    if working_omega[0] > 0.0:
        inserted_zero_frequency = True
        zero_row = np.zeros((1,) + working_imag.shape[1:], dtype=np.float64)
        working_omega = np.concatenate((np.array([0.0], dtype=np.float64), working_omega))
        working_imag = np.concatenate((zero_row, working_imag), axis=0)

    resampled_to_uniform = False
    spacing = float(np.mean(np.diff(working_omega)))

    if bounds is not None:
        lower_bound, upper_bound = bounds
        if upper_bound > working_omega[-1] + max(spacing, 1.0) * 1.0e-9:
            raise ValueError(
                "omega grid does not cover the requested support upper bound: "
                f"{working_omega[-1]:.12g} < {upper_bound:.12g}"
            )
        working_imag, support_was_clipped = _clip_to_support_bounds(
            working_omega,
            working_imag,
            lower_bound,
            upper_bound,
        )
    else:
        support_was_clipped = False

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

    working_real, padded_point_count = _piecewise_linear_principal_value_reconstruction(
        working_omega,
        working_imag,
        static_anchor=anchor,
        assume_hermitian=assume_hermitian,
    )
    padding_factor = 1
    abs_error = 0.0
    rel_error = 0.0

    if inserted_zero_frequency:
        imag_out = working_imag[1:]
        real_out = working_real[1:]
    else:
        imag_out = working_imag
        real_out = working_real

    return {
        "omega": original_omega,
        "imag": imag_out,
        "real": real_out,
        "support_bounds": bounds,
        "uniform_spacing": spacing,
        "resampled_to_uniform": resampled_to_uniform,
        "inserted_zero_frequency": inserted_zero_frequency,
        "zero_frequency_adjusted": zero_frequency_adjusted,
        "support_was_clipped": support_was_clipped,
        "padding_factor": padding_factor,
        "padded_point_count": padded_point_count,
        "estimated_absolute_error": abs_error,
        "estimated_relative_error": rel_error,
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
    lower_bound: float,
    upper_bound: float,
) -> tuple[npt.NDArray[np.float64], bool]:
    clipped = imag_response
    mask = (omega < lower_bound) | (omega > upper_bound)
    if not np.any(mask):
        return clipped, False
    clipped = clipped.copy()
    clipped[mask] = 0.0
    return clipped, True


def _resample_axis0(
    source_x: npt.NDArray[np.float64],
    source_y: npt.NDArray[np.float64],
    target_x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    flat_source = np.ascontiguousarray(source_y.reshape(source_y.shape[0], -1))
    flat_target = np.empty((target_x.size, flat_source.shape[1]), dtype=np.float64)

    for column_index in range(flat_source.shape[1]):
        flat_target[:, column_index] = np.interp(target_x, source_x, flat_source[:, column_index])

    return flat_target.reshape((target_x.size,) + source_y.shape[1:])


def _piecewise_linear_principal_value_reconstruction(
    positive_omega: npt.NDArray[np.float64],
    positive_imag: npt.NDArray[np.float64],
    *,
    static_anchor: npt.NDArray[np.float64] | None,
    assume_hermitian: bool,
) -> tuple[npt.NDArray[np.float64], int]:
    positive_count = positive_imag.shape[0]
    flat_imag = np.ascontiguousarray(positive_imag.reshape(positive_count, -1))
    augmented_omega, symmetric_extension = _build_augmented_symmetric_extension(
        positive_omega,
        flat_imag,
        assume_hermitian=assume_hermitian,
    )
    augmented_count = augmented_omega.size
    positive_start = positive_count
    channel_count = flat_imag.shape[1]
    real_part = np.zeros((positive_count, channel_count), dtype=np.float64)

    for output_index in range(positive_count):
        full_index = positive_start + output_index
        omega_value = augmented_omega[full_index]
        interval_sum = np.zeros(channel_count, dtype=np.float64)

        for interval_index in range(augmented_count - 1):
            if interval_index == full_index - 1 or interval_index == full_index:
                continue
            x0 = augmented_omega[interval_index]
            x1 = augmented_omega[interval_index + 1]
            width = x1 - x0
            slope = (symmetric_extension[interval_index + 1] - symmetric_extension[interval_index]) / width
            intercept = symmetric_extension[interval_index] - slope * x0
            interval_sum += slope * width + (slope * omega_value + intercept) * np.log(
                abs((x1 - omega_value) / (x0 - omega_value))
            )

        if 0 < full_index < augmented_count - 1:
            left_width = omega_value - augmented_omega[full_index - 1]
            right_width = augmented_omega[full_index + 1] - omega_value
            center_value = symmetric_extension[full_index]
            left_slope = (center_value - symmetric_extension[full_index - 1]) / left_width
            right_slope = (symmetric_extension[full_index + 1] - center_value) / right_width
            interval_sum += center_value * np.log(right_width / left_width)
            interval_sum += left_slope * left_width + right_slope * right_width
        elif full_index == augmented_count - 1:
            left_width = omega_value - augmented_omega[full_index - 1]
            left_slope = (symmetric_extension[full_index] - symmetric_extension[full_index - 1]) / left_width
            interval_sum += left_slope * left_width
        else:
            raise RuntimeError("unexpected boundary index during Kramers-Kronig reconstruction")

        real_part[output_index] = interval_sum / np.pi

    if static_anchor is not None:
        anchor_flat = np.asarray(static_anchor, dtype=np.float64).reshape(-1)
        if assume_hermitian:
            real_part += anchor_flat[None, :] - real_part[0:1, :]
        else:
            real_part[0] = anchor_flat

    return real_part.reshape(positive_imag.shape), augmented_count


def _build_augmented_symmetric_extension(
    positive_omega: npt.NDArray[np.float64],
    flat_imag: npt.NDArray[np.float64],
    *,
    assume_hermitian: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    negative_count = flat_imag.shape[0] - 1
    if negative_count < 0:
        raise ValueError("positive_omega must contain at least one sample")

    if flat_imag.shape[0] != positive_omega.size:
        raise ValueError("positive_omega and flat_imag must share the leading axis")

    if assume_hermitian:
        negative = -flat_imag[:0:-1]
    else:
        negative = np.zeros((negative_count, flat_imag.shape[1]), dtype=np.float64)

    full_omega = np.concatenate((-positive_omega[:0:-1], positive_omega))
    full_imag = np.concatenate((negative, flat_imag), axis=0)

    if positive_omega.size == 1:
        right_spacing = 1.0
    else:
        right_spacing = positive_omega[-1] - positive_omega[-2]
    if right_spacing <= 0.0:
        raise ValueError("positive_omega must be strictly increasing")

    augmented_omega = np.concatenate(
        (
            np.array([full_omega[0] - right_spacing], dtype=np.float64),
            full_omega,
            np.array([full_omega[-1] + right_spacing], dtype=np.float64),
        )
    )
    augmented_imag = np.concatenate(
        (
            np.zeros((1, flat_imag.shape[1]), dtype=np.float64),
            full_imag,
            np.zeros((1, flat_imag.shape[1]), dtype=np.float64),
        ),
        axis=0,
    )
    return augmented_omega, augmented_imag
