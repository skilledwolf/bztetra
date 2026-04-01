from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._causality import _reconstruct_retarded_response_impl


@dataclass(slots=True)
class KramersKronigDiagnostics:
    support_bounds: tuple[float, float] | None
    uniform_spacing: float
    resampled_to_uniform: bool
    inserted_zero_frequency: bool
    zero_frequency_adjusted: bool
    support_was_clipped: bool
    padding_factor: int
    padded_point_count: int
    estimated_absolute_error: float
    estimated_relative_error: float


@dataclass(slots=True)
class RetardedResponse:
    omega: npt.NDArray[np.float64]
    imag: npt.NDArray[np.float64]
    real: npt.NDArray[np.float64]
    static_anchor: npt.NDArray[np.float64] | None
    support_bounds: tuple[float, float] | None
    diagnostics: KramersKronigDiagnostics


def reconstruct_retarded_response(
    omega: npt.ArrayLike,
    imag_response: npt.ArrayLike,
    *,
    static_anchor: npt.ArrayLike | None = None,
    support_bounds: tuple[float, float] | npt.ArrayLike | None = None,
    assume_hermitian: bool = False,
    padding_tolerance: float = 5.0e-7,
    max_padding_factor: int = 32,
) -> RetardedResponse:
    """Reconstruct a retarded response from its imaginary part on ω >= 0.

    By default, the unspecified negative-frequency branch is taken to be zero,
    which matches the occupied-to-empty response returned by the current
    `fermi_golden_rule_*` APIs. Set `assume_hermitian=True` to instead extend
    the imaginary part as an odd function, which is appropriate for full
    Hermitian self-responses. A static anchor can be supplied to fix the
    zero-frequency constant, and compact support bounds can be used to clip the
    high-frequency tail automatically before applying the Kramers-Kronig
    transform.
    """

    result = _reconstruct_retarded_response_impl(
        omega,
        imag_response,
        static_anchor=static_anchor,
        support_bounds=support_bounds,
        assume_hermitian=assume_hermitian,
        padding_tolerance=padding_tolerance,
        max_padding_factor=max_padding_factor,
    )
    diagnostics = KramersKronigDiagnostics(
        support_bounds=result["support_bounds"],
        uniform_spacing=float(result["uniform_spacing"]),
        resampled_to_uniform=bool(result["resampled_to_uniform"]),
        inserted_zero_frequency=bool(result["inserted_zero_frequency"]),
        zero_frequency_adjusted=bool(result["zero_frequency_adjusted"]),
        support_was_clipped=bool(result["support_was_clipped"]),
        padding_factor=int(result["padding_factor"]),
        padded_point_count=int(result["padded_point_count"]),
        estimated_absolute_error=float(result["estimated_absolute_error"]),
        estimated_relative_error=float(result["estimated_relative_error"]),
    )
    return RetardedResponse(
        omega=np.asarray(result["omega"], dtype=np.float64),
        imag=np.asarray(result["imag"], dtype=np.float64),
        real=np.asarray(result["real"], dtype=np.float64),
        static_anchor=None
        if static_anchor is None
        else np.asarray(static_anchor, dtype=np.float64),
        support_bounds=result["support_bounds"],
        diagnostics=diagnostics,
    )
