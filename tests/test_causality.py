from __future__ import annotations

import numpy as np

from bztetra import reconstruct_retarded_response
from bztetra._causality import _cached_causality_operator_matrix
from bztetra._causality import _MAX_CACHED_OPERATOR_BYTES


def _positive_branch_spectrum(omega: np.ndarray, cutoff: float) -> np.ndarray:
    values = np.zeros_like(omega, dtype=np.float64)
    mask = (omega >= 0.0) & (omega < cutoff)
    scaled = omega[mask] / cutoff
    values[mask] = np.pi * omega[mask] * (1.0 - scaled * scaled) ** 2
    return values


def _complex_axis_reference(
    omega: np.ndarray,
    cutoff: float,
    *,
    assume_hermitian: bool,
) -> np.ndarray:
    dense_omega = np.linspace(0.0, cutoff, 200_001, dtype=np.float64)
    spectral_density = _positive_branch_spectrum(dense_omega, cutoff) / np.pi
    eta = 5.0e-4
    reference = np.empty_like(omega, dtype=np.float64)

    for index, omega_value in enumerate(omega):
        kernel = 1.0 / (dense_omega - omega_value - 1j * eta)
        if assume_hermitian:
            kernel = kernel - 1.0 / (dense_omega + omega_value + 1j * eta)
        reference[index] = np.trapezoid(spectral_density * kernel, dense_omega).real

    return reference


def test_reconstruct_retarded_response_matches_principal_value_reference() -> None:
    omega = np.linspace(0.0, 3.0, 301, dtype=np.float64)
    imag_response = _positive_branch_spectrum(omega, cutoff=2.2)
    reference_real = _complex_axis_reference(omega, cutoff=2.2, assume_hermitian=False)

    reconstructed = reconstruct_retarded_response(
        omega,
        imag_response,
        static_anchor=np.array(reference_real[0], dtype=np.float64),
        support_bounds=(0.0, 2.2),
    )

    np.testing.assert_allclose(reconstructed.imag, imag_response, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(reconstructed.real, reference_real, rtol=3.0e-3, atol=3.0e-3)
    assert reconstructed.diagnostics.cached_operator
    assert reconstructed.diagnostics.augmented_point_count >= (2 * omega.size - 1)
    assert reconstructed.diagnostics.static_anchor_applied


def test_reconstruct_retarded_response_supports_hermitian_odd_extension() -> None:
    omega = np.linspace(0.0, 3.0, 301, dtype=np.float64)
    imag_response = _positive_branch_spectrum(omega, cutoff=2.2)

    reconstructed = reconstruct_retarded_response(
        omega,
        imag_response,
        static_anchor=np.array(0.0, dtype=np.float64),
        support_bounds=(0.0, 2.2),
        assume_hermitian=True,
    )

    assert np.isfinite(reconstructed.real).all()
    np.testing.assert_allclose(reconstructed.real[0], 0.0, rtol=0.0, atol=1.0e-12)


def test_reconstruct_retarded_response_supports_nonuniform_grids() -> None:
    omega = np.linspace(0.0, 1.0, 129, dtype=np.float64) ** 1.5 * 3.0
    imag_response = _positive_branch_spectrum(omega, cutoff=2.4)

    reconstructed = reconstruct_retarded_response(
        omega,
        imag_response,
        static_anchor=np.array(0.0, dtype=np.float64),
        support_bounds=(0.0, 2.4),
    )

    np.testing.assert_allclose(reconstructed.omega, omega, rtol=0.0, atol=0.0)
    assert reconstructed.real.shape == omega.shape
    assert reconstructed.diagnostics.minimum_spacing < reconstructed.diagnostics.maximum_spacing


def test_nonhermitian_static_anchor_only_pins_zero_frequency_point() -> None:
    omega = np.linspace(0.0, 3.0, 301, dtype=np.float64)
    imag_response = _positive_branch_spectrum(omega, cutoff=2.2)

    raw = reconstruct_retarded_response(
        omega,
        imag_response,
        support_bounds=(0.0, 2.2),
    )
    anchored = reconstruct_retarded_response(
        omega,
        imag_response,
        static_anchor=np.array(7.5, dtype=np.float64),
        support_bounds=(0.0, 2.2),
    )

    np.testing.assert_allclose(anchored.real[1:], raw.real[1:], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(anchored.real[0], 7.5, rtol=0.0, atol=1.0e-12)


def test_nonhermitian_static_anchor_requires_zero_frequency_sample() -> None:
    omega = np.linspace(0.1, 3.0, 300, dtype=np.float64)
    imag_response = _positive_branch_spectrum(omega, cutoff=2.2)

    try:
        reconstruct_retarded_response(
            omega,
            imag_response,
            static_anchor=np.array(7.5, dtype=np.float64),
            support_bounds=(0.0, 2.2),
        )
    except ValueError as exc:
        assert "omega[0] == 0.0" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected non-Hermitian static_anchor to require an explicit zero-frequency sample")


def test_support_bounds_insert_exact_zero_edges() -> None:
    omega = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    imag_response = np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float64)

    clipped = reconstruct_retarded_response(
        omega,
        imag_response,
        support_bounds=(0.0, 1.5),
    )
    reference = reconstruct_retarded_response(
        np.array([0.0, 1.0, 1.5, 2.0, 3.0], dtype=np.float64),
        np.array([0.0, np.pi, 0.0, 0.0, 0.0], dtype=np.float64),
    )

    np.testing.assert_allclose(clipped.real, reference.real[[0, 1, 3, 4]], rtol=1.0e-12, atol=1.0e-12)
    assert clipped.diagnostics.support_boundary_insertions == 1
    assert clipped.diagnostics.support_was_clipped


def test_operator_cache_skips_large_uncacheable_matrices() -> None:
    _cached_causality_operator_matrix.cache_clear()

    small_omega = np.linspace(0.0, 3.0, 301, dtype=np.float64)
    small_imag = _positive_branch_spectrum(small_omega, cutoff=2.2)
    reconstruct_retarded_response(
        small_omega,
        small_imag,
        support_bounds=(0.0, 2.2),
    )
    assert _cached_causality_operator_matrix.cache_info().currsize == 1

    large_count = int(np.sqrt(_MAX_CACHED_OPERATOR_BYTES / np.dtype(np.float64).itemsize)) + 1
    large_omega = np.linspace(0.0, 5.0, large_count, dtype=np.float64)
    large_imag = np.pi * np.exp(-large_omega)
    reconstructed = reconstruct_retarded_response(
        large_omega,
        large_imag,
        support_bounds=(0.0, 5.0),
    )

    assert not reconstructed.diagnostics.cached_operator
    assert _cached_causality_operator_matrix.cache_info().currsize == 1
