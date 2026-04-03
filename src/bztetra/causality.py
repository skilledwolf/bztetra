from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._causality import _reconstruct_retarded_response_impl


@dataclass(slots=True)
class KramersKronigDiagnostics:
    support_bounds: tuple[float, float] | None
    minimum_spacing: float
    maximum_spacing: float
    inserted_zero_frequency: bool
    zero_frequency_adjusted: bool
    support_was_clipped: bool
    support_boundary_insertions: int
    static_anchor_applied: bool
    augmented_point_count: int
    cached_operator: bool


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
) -> RetardedResponse:
    """Reconstruct a retarded response from its imaginary part on ω >= 0.

    By default, the unspecified negative-frequency branch is taken to be zero,
    which matches the occupied-to-empty response returned by the current
    `fermi_golden_rule_*` APIs. Set `assume_hermitian=True` to instead extend
    the imaginary part as an odd function, which is appropriate for full
    Hermitian self-responses. For the default non-Hermitian branch, a
    `static_anchor` is only valid when `omega[0] == 0.0`, because only the
    returned zero-frequency point is pinned. Compact support bounds clip the
    spectrum on the working piecewise-linear grid, inserting zero-valued edge
    samples when a support boundary falls between two requested frequencies.
    """

    result = _reconstruct_retarded_response_impl(
        omega,
        imag_response,
        static_anchor=static_anchor,
        support_bounds=support_bounds,
        assume_hermitian=assume_hermitian,
    )
    diagnostics = KramersKronigDiagnostics(
        support_bounds=result["support_bounds"],
        minimum_spacing=float(result["minimum_spacing"]),
        maximum_spacing=float(result["maximum_spacing"]),
        inserted_zero_frequency=bool(result["inserted_zero_frequency"]),
        zero_frequency_adjusted=bool(result["zero_frequency_adjusted"]),
        support_was_clipped=bool(result["support_was_clipped"]),
        support_boundary_insertions=int(result["support_boundary_insertions"]),
        static_anchor_applied=bool(result["static_anchor_applied"]),
        augmented_point_count=int(result["augmented_point_count"]),
        cached_operator=bool(result["cached_operator"]),
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
